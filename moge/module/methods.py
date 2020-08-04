import multiprocessing
import pytorch_lightning as pl
import torch
from cogdl.models.nn.gtn import GTN
from cogdl.models.nn.han import HAN
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from torch.nn import functional as F
from torch_geometric.nn import MetaPath2Vec

from moge.generator.datasets import HeteroNetDataset
from .metrics import Metrics
from .trainer import _fix_dp_return_type
from .latte import LATTELayer, LATTE
from .classifier import DenseClassification, MulticlassClassification
from .losses import ClassificationLoss
from .utils import filter_samples

class MetricsComparison(pl.LightningModule):
    def __init__(self):
        super(MetricsComparison, self).__init__()

    def name(self):
        if hasattr(self, "_name"):
            return self._name
        else:
            return self.__class__.__name__

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean().item()
        logs = self.train_metrics.compute_metrics()
        logs = _fix_dp_return_type(logs, device=outputs[0]["loss"].device)

        logs.update({"loss": avg_loss})
        self.train_metrics.reset_metrics()
        return {"log": logs}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean().item()
        logs = self.valid_metrics.compute_metrics()
        logs = _fix_dp_return_type(logs, device=outputs[0]["val_loss"].device)

        logs.update({"val_loss": avg_loss})
        self.valid_metrics.reset_metrics()
        return {"progress_bar": logs,
                "log": logs}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).sum().item()
        if hasattr(self, "test_metrics"):
            logs = self.test_metrics.compute_metrics()
            self.test_metrics.reset_metrics()
        else:
            logs = {}
        logs.update({"test_loss": avg_loss})

        return {"progress_bar": logs,
                "log": logs}


class LATTENodeClassifier(MetricsComparison):
    def __init__(self, hparams, dataset: HeteroNetDataset, metrics=["accuracy"], collate_fn="LATTENode_batch") -> None:
        super(LATTENodeClassifier, self).__init__()
        self.head_node_type = dataset.head_node_type
        self.dataset = dataset
        self.multilabel = dataset.multilabel
        self._name = f"LATTE-{hparams.t_order}{' proximity' if hparams.use_proximity_loss else ''}"
        num_class = dataset.n_classes
        self.collate_fn = collate_fn

        self.latte = LATTE(embedding_dim=hparams.embedding_dim, t_order=hparams.t_order,
                           num_nodes_dict=dataset.num_nodes_dict,
                           node_attr_shape=dataset.node_attr_shape, metapaths=dataset.get_metapaths(),
                           neg_sampling_ratio=hparams.neg_sampling_ratio,
                           use_proximity_loss=hparams.use_proximity_loss)
        hparams.embedding_dim = hparams.embedding_dim * hparams.t_order
        # self.classifier = DenseClassification(hparams)
        # self.classifier = MulticlassClassification(num_feature=hparams.embedding_dim, num_class=hparams.n_classes)
        self.classifier = torch.nn.Linear(hparams.embedding_dim, dataset.n_classes)
        self.criterion = ClassificationLoss(n_classes=dataset.n_classes,
                                            # class_weight=dataset.class_weight,
                                            loss_type=hparams.loss_type,
                                            multilabel=dataset.multilabel)

        self.train_metrics = Metrics(prefix="", loss_type=hparams.loss_type, n_classes=num_class,
                                     multilabel=dataset.multilabel, metrics=metrics)
        self.valid_metrics = Metrics(prefix="val_", loss_type=hparams.loss_type, n_classes=num_class,
                                     multilabel=dataset.multilabel, metrics=metrics)
        self.test_metrics = Metrics(prefix="test_", loss_type=hparams.loss_type, n_classes=num_class,
                                    multilabel=dataset.multilabel, metrics=metrics)
        self.hparams = hparams

    def forward(self, x_dict, global_node_index, edge_index_dict):
        embeddings, proximity_loss = self.latte.forward(x_dict, global_node_index, edge_index_dict)
        y_hat = self.classifier.forward(embeddings[self.head_node_type])
        return y_hat, proximity_loss

    def loss(self, y_hat, y):
        if not self.multilabel:
            loss = self.criterion(y_hat, y)
        else:
            loss = F.binary_cross_entropy_with_logits(y_hat, y.type_as(y_hat))
        return loss

    def training_step(self, batch, batch_nb):
        X, y, weights = batch
        y_hat, proximity_loss = self.forward(X["x_dict"], X["global_node_index"], X["edge_index_dict"])
        y_hat, y = filter_samples(Y_hat=y_hat, Y=y, weights=weights)

        self.train_metrics.update_metrics(y_pred=y_hat, y_true=y, weights=None)
        loss = self.loss(y_hat, y)

        logs = None
        if self.hparams.use_proximity_loss:
            loss = loss + proximity_loss
            logs = {"proximity_loss": proximity_loss}

        outputs = {'loss': loss}
        if logs is not None:
            outputs.update({'progress_bar': logs, "logs": logs})
        return outputs

    def validation_step(self, batch, batch_nb):
        X, y, weights = batch
        y_hat, proximity_loss = self.forward(X["x_dict"], X["global_node_index"], X["edge_index_dict"])
        y_hat, y = filter_samples(Y_hat=y_hat, Y=y, weights=weights)
        self.valid_metrics.update_metrics(y_pred=y_hat, y_true=y, weights=None)

        val_loss = self.loss(y_hat, y)
        if self.hparams.use_proximity_loss:
            val_loss = val_loss + proximity_loss

        return {"val_loss": val_loss}

    def test_step(self, batch, batch_nb):
        X, y, weights = batch
        y_hat, proximity_loss = self.forward(X["x_dict"], X["global_node_index"], X["edge_index_dict"])
        y_hat, y = filter_samples(Y_hat=y_hat, Y=y, weights=weights)
        self.test_metrics.update_metrics(y_pred=y_hat, y_true=y, weights=None)

        test_loss = self.loss(y_hat, y)
        if self.hparams.use_proximity_loss:
            test_loss = test_loss + proximity_loss

        return {"test_loss": test_loss}

    def train_dataloader(self):
        return self.dataset.train_dataloader(collate_fn=self.collate_fn,
                                             batch_size=self.hparams.batch_size,
                                             num_workers=int(0.8 * multiprocessing.cpu_count()))

    def val_dataloader(self, batch_size=None):
        return self.dataset.val_dataloader(collate_fn=self.collate_fn,
                                           batch_size=self.hparams.batch_size * 2,
                                           num_workers=int(0.2 * multiprocessing.cpu_count()))

    def test_dataloader(self, batch_size=None):
        return self.dataset.test_dataloader(collate_fn=self.collate_fn,
                                            batch_size=self.hparams.batch_size * 2,
                                            num_workers=int(0.2 * multiprocessing.cpu_count()))

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.hparams.lr)

class GTN(GTN, MetricsComparison):
    def __init__(self, hparams, dataset: HeteroNetDataset, metrics=["precision"]):
        num_edge = len(dataset.edge_index_dict)
        num_layers = len(dataset.edge_index_dict)
        num_class = dataset.n_classes
        self.multilabel = dataset.multilabel
        self.collate_fn = hparams.collate_fn
        self.val_collate_fn = hparams.val_collate_fn
        num_nodes = dataset.num_nodes_dict[dataset.head_node_type]

        if hasattr(dataset, "x"):
            w_in = dataset.in_features
        else:
            w_in = hparams.embedding_dim

        w_out = hparams.embedding_dim
        num_channels = hparams.num_channels
        super().__init__(num_edge, num_channels, w_in, w_out, num_class, num_nodes, num_layers)

        if not hasattr(dataset, "x"):
            self.embedding = torch.nn.Embedding(num_embeddings=num_nodes, embedding_dim=hparams.embedding_dim,
                                                sparse=True)

        self.train_metricss = Metrics(prefix=None, loss_type=hparams.loss_type, n_classes=num_class,
                                      multilabel=dataset.multilabel, metrics=metrics)
        self.valid_metrics = Metrics(prefix="val_", loss_type=hparams.loss_type, n_classes=num_class,
                                     multilabel=dataset.multilabel, metrics=metrics)
        self.hparams = hparams
        self.dataset = dataset
        self.head_node_type = self.dataset.head_node_type

    def forward(self, A, X, x_idx):
        if X is None and "batch" in self.collate_fn:
            X = self.embedding.weight[x_idx]
        elif X is None:
            X = self.embedding.weight

        Ws = []
        for i in range(self.num_layers):
            if i == 0:
                H, W = self.layers[i](A)
            else:
                H = self.normalization(H)
                H, W = self.layers[i](A, H)
            Ws.append(W)
        for i in range(self.num_channels):
            if i == 0:
                edge_index, edge_weight = H[i][0], H[i][1]
                X_ = self.gcn(X, edge_index=edge_index.detach(), edge_weight=edge_weight)
                X_ = F.relu(X_)
            else:
                edge_index, edge_weight = H[i][0], H[i][1]
                X_ = torch.cat((X_, F.relu(self.gcn(X, edge_index=edge_index.detach(), edge_weight=edge_weight))),
                               dim=1)
        X_ = self.linear1(X_)
        X_ = F.relu(X_)
        # X_ = F.dropout(X_, p=0.5)
        if "batch" in self.collate_fn:
            y = self.linear2(X_)
        else:
            y = self.linear2(X_[x_idx])
        return y

    def loss(self, y_hat, y):
        if not self.multilabel:
            loss = self.cross_entropy_loss(y_hat, y)
        else:
            loss = F.binary_cross_entropy_with_logits(y_hat, y.type_as(y_hat))

        return loss

    def training_step(self, batch, batch_nb):
        X, y, weights = batch
        y_hat = self.forward(X["adj"], X["x"], X["idx"])
        y_hat, y = filter_samples(Y_hat=y_hat, Y=y, weights=weights)
        self.train_metricss.update_metrics(y_pred=y_hat, y_true=y, weights=None)
        loss = self.loss(y_hat, y)
        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        X, y, weights = batch

        y_hat = self.forward(X["adj"], X["x"], X["idx"])
        y_hat, y = filter_samples(Y_hat=y_hat, Y=y, weights=weights)
        self.valid_metrics.update_metrics(y_pred=y_hat, y_true=y, weights=None)
        loss = self.loss(y_hat, y)

        return {"val_loss": loss}

    def test_step(self, batch, batch_nb):
        X, y, weights = batch
        y_hat = self.forward(X["adj"], X["x"], X["idx"])
        y_hat, y = filter_samples(Y_hat=y_hat, Y=y, weights=weights)
        loss = self.loss(y_hat, y)

        return {"test_loss": loss}

    def train_dataloader(self):
        return self.dataset.train_dataloader(collate_fn=self.collate_fn, batch_size=self.hparams.batch_size)

    def val_dataloader(self):
        return self.dataset.val_dataloader(collate_fn=self.val_collate_fn, batch_size=self.hparams.batch_size * 2)

    def test_dataloader(self):
        return self.dataset.test_dataloader(collate_fn=self.val_collate_fn, batch_size=self.hparams.batch_size * 2)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


class HAN(HAN, MetricsComparison):
    def __init__(self, hparams, dataset: HeteroNetDataset, metrics=["precision"]):
        num_edge = len(dataset.edge_index_dict)
        num_layers = len(dataset.edge_index_dict)
        num_class = dataset.n_classes
        self.collate_fn = hparams.collate_fn
        self.val_collate_fn = hparams.val_collate_fn
        self.multilabel = dataset.multilabel
        num_nodes = dataset.num_nodes_dict[dataset.head_node_type]

        if hasattr(dataset, "x"):
            w_in = dataset.in_features
        else:
            w_in = hparams.embedding_dim

        w_out = hparams.embedding_dim

        super().__init__(num_edge, w_in, w_out, num_class, num_nodes, num_layers)

        if not hasattr(dataset, "x"):
            self.embedding = torch.nn.Embedding(num_embeddings=num_nodes, embedding_dim=hparams.embedding_dim)

        self.train_metricss = Metrics(prefix=None, loss_type=hparams.loss_type, n_classes=num_class,
                                      multilabel=dataset.multilabel, metrics=metrics)
        self.valid_metrics = Metrics(prefix="val_", loss_type=hparams.loss_type, n_classes=num_class,
                                     multilabel=dataset.multilabel, metrics=metrics)
        self.hparams = hparams
        self.dataset = dataset
        self.head_node_type = self.dataset.head_node_type

    def forward(self, A, X, x_idx):
        if X is None and "batch" in self.collate_fn:
            X = self.embedding.weight[x_idx]
        elif X is None:
            X = self.embedding.weight

        for i in range(self.num_layers):
            X = self.layers[i](X, A)

        if "batch" in self.collate_fn:
            y = self.linear(X)
        else:
            y = self.linear(X[x_idx])
        return y

    def loss(self, y_hat, y):
        if not self.multilabel:
            loss = self.cross_entropy_loss(y_hat, y)
        else:
            loss = F.binary_cross_entropy_with_logits(y_hat, y.type_as(y_hat))
        return loss

    def training_step(self, batch, batch_nb):
        X, y, weights = batch
        y_hat = self.forward(X["adj"], X["x"], X["idx"])
        y_hat, y = filter_samples(Y_hat=y_hat, Y=y, weights=weights)
        self.train_metricss.update_metrics(y_pred=y_hat, y_true=y, weights=None)
        loss = self.loss(y_hat, y)
        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        X, y, weights = batch

        y_hat = self.forward(X["adj"], X["x"], X["idx"])
        y_hat, y = filter_samples(Y_hat=y_hat, Y=y, weights=weights)
        self.valid_metrics.update_metrics(y_pred=y_hat, y_true=y, weights=None)
        loss = self.loss(y_hat, y)

        return {"val_loss": loss}

    def test_step(self, batch, batch_nb):
        X, y, weights = batch
        y_hat = self.forward(X["adj"], X["x"], X["idx"])
        y_hat, y = filter_samples(Y_hat=y_hat, Y=y, weights=weights)
        loss = self.loss(y_hat, y)

        return {"test_loss": loss}

    def train_dataloader(self):
        return self.dataset.train_dataloader(collate_fn=self.collate_fn, batch_size=self.hparams.batch_size)

    def val_dataloader(self):
        return self.dataset.val_dataloader(collate_fn=self.val_collate_fn, batch_size=self.hparams.batch_size * 2)

    def test_dataloader(self):
        return self.dataset.test_dataloader(collate_fn=self.val_collate_fn, batch_size=self.hparams.batch_size)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


class MetaPath2Vec(MetaPath2Vec, MetricsComparison):
    def __init__(self, hparams, dataset: HeteroNetDataset, metrics=None):
        # Hparams
        self.train_ratio = hparams.train_ratio
        self.batch_size = hparams.batch_size
        self.sparse = hparams.sparse

        embedding_dim = hparams.embedding_dim
        walk_length = hparams.walk_length
        context_size = hparams.context_size
        walks_per_node = hparams.walks_per_node
        num_negative_samples = hparams.num_negative_samples

        # Dataset
        self.dataset = dataset
        num_nodes_dict = None
        metapath = self.dataset.metapaths
        self.head_node_type = self.dataset.head_node_type
        edge_index_dict = dataset.edge_index_dict

        super().__init__(edge_index_dict, embedding_dim, metapath, walk_length, context_size, walks_per_node,
                         num_negative_samples, num_nodes_dict, self.sparse)
        self.hparams = hparams

    def training_step(self, batch, batch_nb):

        pos_rw, neg_rw = batch
        loss = self.loss(pos_rw, neg_rw)
        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        pos_rw, neg_rw = batch
        loss = self.loss(pos_rw, neg_rw)
        return {"val_loss": loss}

    def test_step(self, batch, batch_nb):
        pos_rw, neg_rw = batch
        loss = self.loss(pos_rw, neg_rw)
        return {"test_loss": loss}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).sum().item()
        accuracy = self.node_classification(training=True)

        return {"progress_bar": {"accuracy": accuracy},
                "log": {"loss": avg_loss, "accuracy": accuracy}}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).sum().item()
        logs = {"val_loss": avg_loss,
                "val_accuracy": self.node_classification(training=False)}
        return {"progress_bar": logs,
                "log": logs}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).sum().item()
        logs = {"test_loss": avg_loss,
                "test_accuracy": self.node_classification(training=False)}
        return {"progress_bar": logs,
                "log": logs}

    def node_classification(self, training=True):
        if training:
            z = self.forward(self.head_node_type,
                             batch=self.dataset.y_index_dict[self.head_node_type][self.dataset.training_idx])
            y = self.dataset.y_dict[self.head_node_type][self.dataset.training_idx]

            perm = torch.randperm(z.size(0))
            train_perm = perm[:int(z.size(0) * self.dataset.train_ratio)]
            test_perm = perm[int(z.size(0) * self.dataset.train_ratio):]

            if y.dim() > 1 and y.size(1) > 1:
                clf = OneVsRestClassifier(LogisticRegression(solver="lbfgs", multi_class="auto", max_iter=150))
            else:
                clf = LogisticRegression(solver="lbfgs", multi_class="auto", max_iter=150)

            clf.fit(z[train_perm].detach().cpu().numpy(),
                    y[train_perm].detach().cpu().numpy())

            accuracy = clf.score(z[test_perm].detach().cpu().numpy(),
                                 y[test_perm].detach().cpu().numpy())
        else:
            z_train = self.forward(self.head_node_type,
                                   batch=self.dataset.y_index_dict[self.head_node_type][self.dataset.training_idx])
            y_train = self.dataset.y_dict[self.head_node_type][self.dataset.training_idx]

            z_val = self.forward(self.head_node_type,
                                 batch=self.dataset.y_index_dict[self.head_node_type][self.dataset.validation_idx])
            y_val = self.dataset.y_dict[self.head_node_type][self.dataset.validation_idx]

            if y_train.dim() > 1 and y_train.size(1) > 1:
                clf = OneVsRestClassifier(LogisticRegression(solver="lbfgs", multi_class="auto", max_iter=150))
            else:
                clf = LogisticRegression(solver="lbfgs", multi_class="auto", max_iter=150)

            clf.fit(z_train.detach().cpu().numpy(),
                    y_train.detach().cpu().numpy())

            accuracy = clf.score(z_val.detach().cpu().numpy(),
                                 y_val.detach().cpu().numpy())

        return accuracy

    def train_dataloader(self):
        return self.dataset.train_dataloader(collate_fn=self.sample, batch_size=self.hparams.batch_size)

    def val_dataloader(self):
        return self.dataset.val_dataloader(collate_fn=self.sample, batch_size=self.hparams.batch_size)

    def test_dataloader(self):
        return self.dataset.test_dataloader(collate_fn=self.sample, batch_size=self.hparams.batch_size)

    def configure_optimizers(self):
        if self.sparse:
            return torch.optim.SparseAdam(self.parameters(), lr=self.hparams.lr)
        else:
            return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
