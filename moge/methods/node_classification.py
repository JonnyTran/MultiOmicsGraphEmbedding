import multiprocessing
import itertools
import pytorch_lightning as pl
import pandas as pd
import torch
# from cogdl.models.nn.pyg_gtn import GTN
# from cogdl.models.nn.pyg_han import HAN
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from torch.nn import functional as F
from torch_geometric.nn import MetaPath2Vec

from torch.optim.lr_scheduler import ReduceLROnPlateau

from moge.generator.datasets import HeteroNetDataset
from moge.module.metrics import Metrics
from moge.module.trainer import _fix_dp_return_type
from moge.module.latte import LATTE
from moge.module.classifier import MulticlassClassification, DenseClassification
from moge.module.losses import ClassificationLoss
from moge.module.utils import filter_samples, preprocess_input, pad_tensors

class NodeClfMetrics(pl.LightningModule):
    def __init__(self, hparams, dataset, metrics):
        super(NodeClfMetrics, self).__init__()
        self.train_metrics = Metrics(prefix="", loss_type=hparams.loss_type, n_classes=dataset.n_classes,
                                     multilabel=dataset.multilabel, metrics=metrics)
        self.valid_metrics = Metrics(prefix="val_", loss_type=hparams.loss_type, n_classes=dataset.n_classes,
                                     multilabel=dataset.multilabel, metrics=metrics)
        self.test_metrics = Metrics(prefix="test_", loss_type=hparams.loss_type, n_classes=dataset.n_classes,
                                    multilabel=dataset.multilabel, metrics=metrics)
        self.hparams = hparams

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
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean().item()
        if hasattr(self, "test_metrics"):
            logs = self.test_metrics.compute_metrics()
            self.test_metrics.reset_metrics()
        else:
            logs = {}
        logs.update({"test_loss": avg_loss})

        return {"progress_bar": logs,
                "log": logs}

    def print_pred_class_counts(self, y_hat, y, multilabel, n_top_class=8):
        if multilabel:
            y_pred_dict = pd.Series(y_hat.sum(1).detach().cpu().type(torch.int).numpy()).value_counts().to_dict()
            y_true_dict = pd.Series(y.sum(1).detach().cpu().type(torch.int).numpy()).value_counts().to_dict()
            print(f"y_pred {len(y_pred_dict)} classes",
                  {str(k): v for k, v in itertools.islice(y_pred_dict.items(), n_top_class)})
            print("y_true classes", {str(k): v for k, v in itertools.islice(y_true_dict.items(), n_top_class)})
        else:
            y_pred_dict = pd.Series(y_hat.argmax(1).detach().cpu().type(torch.int).numpy()).value_counts().to_dict()
            y_true_dict = pd.Series(y.detach().cpu().type(torch.int).numpy()).value_counts().to_dict()
            print(f"y_pred {len(y_pred_dict)} classes",
                  {str(k): v for k, v in itertools.islice(y_pred_dict.items(), n_top_class)})
            print("y_true classes", {str(k): v for k, v in itertools.islice(y_true_dict.items(), n_top_class)})


class LATTENodeClassifier(NodeClfMetrics):
    def __init__(self, hparams, dataset: HeteroNetDataset, metrics=["accuracy"], collate_fn="neighbor_sampler") -> None:
        super(LATTENodeClassifier, self).__init__(hparams=hparams, dataset=dataset, metrics=metrics)
        self.head_node_type = dataset.head_node_type
        self.dataset = dataset
        self.multilabel = dataset.multilabel
        self._name = f"LATTE-{hparams.t_order}{' proximity' if hparams.use_proximity_loss else ''}"
        self.collate_fn = collate_fn

        self.latte = LATTE(in_channels_dict=dataset.node_attr_shape, embedding_dim=hparams.embedding_dim,
                           t_order=hparams.t_order, num_nodes_dict=dataset.num_nodes_dict,
                           metapaths=dataset.get_metapaths(), activation=hparams.activation,
                           use_proximity_loss=hparams.use_proximity_loss, neg_sampling_ratio=hparams.neg_sampling_ratio)
        hparams.embedding_dim = hparams.embedding_dim * hparams.t_order
        self.classifier = DenseClassification(hparams)
        # self.classifier = MulticlassClassification(num_feature=hparams.embedding_dim,
        #                                            num_class=hparams.n_classes,
        #                                            loss_type=hparams.loss_type)
        self.criterion = ClassificationLoss(n_classes=dataset.n_classes,
                                            class_weight=dataset.class_weight if hasattr(dataset,
                                                                                         "class_weight") and hparams.use_class_weights else None,
                                            loss_type=hparams.loss_type,
                                            multilabel=dataset.multilabel)

    def forward(self, x_dict, global_node_index, edge_index_dict):
        embeddings, proximity_loss, _ = self.latte.forward(x_dict, global_node_index, edge_index_dict)
        y_hat = self.classifier.forward(embeddings[self.head_node_type])
        return y_hat, proximity_loss

    def training_step(self, batch, batch_nb):
        X, y, weights = batch
        y_hat, proximity_loss = self.forward(X["x_dict"], X["global_node_index"], X["edge_index_dict"])
        y_hat, y = filter_samples(Y_hat=y_hat, Y=y, weights=weights)
        loss = self.criterion(y_hat, y)

        self.train_metrics.update_metrics(y_hat, y, weights=None)

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
        # print({k: {j: l.shape for j, l in v.items()} for k, v in X.items()})
        # print("y", y.shape)
        y_hat, proximity_loss = self.forward(X["x_dict"], X["global_node_index"], X["edge_index_dict"])
        y_hat, y = filter_samples(Y_hat=y_hat, Y=y, weights=weights)
        val_loss = self.criterion(y_hat, y)

        self.valid_metrics.update_metrics(y_hat, y, weights=None)

        if self.hparams.use_proximity_loss:
            val_loss = val_loss + proximity_loss

        return {"val_loss": val_loss}

    def test_step(self, batch, batch_nb):
        X, y, weights = batch
        y_hat, proximity_loss = self.forward(X["x_dict"], X["global_node_index"], X["edge_index_dict"])
        y_hat, y = filter_samples(Y_hat=y_hat, Y=y, weights=weights)
        test_loss = self.criterion(y_hat, y)

        if batch_nb == 0:
            self.print_pred_class_counts(y_hat, y, multilabel=self.dataset.multilabel)

        self.test_metrics.update_metrics(y_hat, y, weights=None)

        if self.hparams.use_proximity_loss:
            test_loss = test_loss + proximity_loss

        return {"test_loss": test_loss}

    @staticmethod
    def multiplex_collate_fn(node_types, layers):
        def collate_fn(batch):
            y_all, idx_all = [], []
            node_type_concat = dict()
            layer_concat = dict()
            for node_type in node_types:
                node_type_concat[node_type] = []
            for layer in layers:
                layer_concat[layer] = []

            for X, y, idx in batch:
                for node_type in node_types:
                    node_type_concat[node_type].append(torch.tensor(X[node_type]))
                for layer in layers:
                    layer_concat[layer].append(torch.tensor(X[layer]))
                y_all.append(torch.tensor(y))
                idx_all.append(torch.tensor(idx))

            X_all = {}
            for node_type in node_types:
                X_all[node_type] = torch.cat(node_type_concat[node_type])
            for layer in layers:
                X_all[layer] = pad_tensors(layer_concat[layer])

            return X_all, torch.cat(y_all), torch.cat(idx_all)

        return collate_fn

    def train_dataloader(self):
        return self.dataset.train_dataloader(collate_fn=self.collate_fn,
                                             batch_size=self.hparams.batch_size,
                                             num_workers=int(0.8 * multiprocessing.cpu_count()),
                                             t_order=self.hparams.t_order)

    def val_dataloader(self, batch_size=None):
        return self.dataset.val_dataloader(collate_fn=self.collate_fn,
                                           batch_size=self.hparams.batch_size * 2,
                                           num_workers=int(0.2 * multiprocessing.cpu_count()),
                                           t_order=self.hparams.t_order)

    def test_dataloader(self, batch_size=None):
        return self.dataset.test_dataloader(collate_fn=self.collate_fn,
                                            batch_size=self.hparams.batch_size * 2,
                                            num_workers=int(0.2 * multiprocessing.cpu_count()),
                                            t_order=self.hparams.t_order)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

        # optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=self.named_parameters())
        scheduler = ReduceLROnPlateau(optimizer)

        return [optimizer], [scheduler]


class GTN(NodeClfMetrics):
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
        super().__init__(num_edge, None, num_channels)

        if not hasattr(dataset, "x"):
            self.embedding = torch.nn.Embedding(num_embeddings=num_nodes, embedding_dim=hparams.embedding_dim,
                                                sparse=True)

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
        self.train_metricss.update_metrics(y_hat, y, weights=None)
        loss = self.loss(y_hat, y)
        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        X, y, weights = batch

        y_hat = self.forward(X["adj"], X["x"], X["idx"])
        y_hat, y = filter_samples(Y_hat=y_hat, Y=y, weights=weights)
        loss = self.loss(y_hat, y)
        self.valid_metrics.update_metrics(y_hat, y, weights=None)

        return {"val_loss": loss}

    def test_step(self, batch, batch_nb):
        X, y, weights = batch
        y_hat = self.forward(X["adj"], X["x"], X["idx"])
        loss = self.loss(y_hat, y)
        y_hat, y = filter_samples(Y_hat=y_hat, Y=y, weights=weights)

        return {"test_loss": loss}

    def train_dataloader(self):
        return self.dataset.train_dataloader(collate_fn=self.collate_fn, batch_size=self.hparams.batch_size)

    def val_dataloader(self):
        return self.dataset.val_dataloader(collate_fn=self.val_collate_fn, batch_size=self.hparams.batch_size * 2)

    def test_dataloader(self):
        return self.dataset.test_dataloader(collate_fn=self.val_collate_fn, batch_size=self.hparams.batch_size * 2)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


class HAN(NodeClfMetrics):
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

        super().__init__(num_edge, None, w_in)

        if not hasattr(dataset, "x"):
            self.embedding = torch.nn.Embedding(num_embeddings=num_nodes, embedding_dim=hparams.embedding_dim)

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
        self.train_metrics.update_metrics(y_hat, y, weights=None)
        loss = self.loss(y_hat, y)
        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        X, y, weights = batch

        y_hat = self.forward(X["adj"], X["x"], X["idx"])
        y_hat, y = filter_samples(Y_hat=y_hat, Y=y, weights=weights)
        self.valid_metrics.update_metrics(y_hat, y, weights=None)
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


class MetaPath2Vec(MetaPath2Vec, NodeClfMetrics):
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

        super().__init__(edge_index_dict, None, embedding_dim)
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
