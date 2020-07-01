import pytorch_lightning as pl
import torch
from cogdl.models.nn.gtn import GTN
from cogdl.models.nn.han import HAN
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from torch.nn import functional as F
from torch_geometric.nn import MetaPath2Vec

from moge.generator.datasets import HeterogeneousNetworkDataset
from .metrics import Metrics
from .trainer import _fix_dp_return_type


class MetricsComparison(pl.LightningModule):
    def __init__(self):
        super(MetricsComparison, self).__init__()

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["batch_loss"] for x in outputs]).mean().item()
        logs = self.training_metrics.compute_metrics()
        logs = _fix_dp_return_type(logs, device=outputs[0]["batch_loss"].device)

        logs.update({"loss": avg_loss})
        self.training_metrics.reset_metrics()
        return {"progress_bar": logs, "log": logs}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean().item()
        logs = self.validation_metrics.compute_metrics()
        logs = _fix_dp_return_type(logs, device=outputs[0]["val_loss"].device)

        logs.update({"val_loss": avg_loss})
        self.validation_metrics.reset_metrics()
        return {"progress_bar": logs,
                "log": logs}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).sum().item()
        logs = {"test_loss": avg_loss}
        return {"progress_bar": logs,
                "log": logs}


class GTN(GTN, MetricsComparison):
    def __init__(self, hparams, dataset: HeterogeneousNetworkDataset, metrics=["precision"]):
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
        # for i, l in enumerate(self.layers):
        #     self.layers[i] = self.layers[i].cuda(i % 3 + 1)

        if not hasattr(dataset, "x"):
            self.embedding = torch.nn.Embedding(num_embeddings=num_nodes, embedding_dim=hparams.embedding_dim,
                                                sparse=True)

        self.training_metrics = Metrics(loss_type=hparams.loss_type, n_classes=num_class, metrics=metrics, prefix=None,
                                        multilabel=dataset.multilabel)
        self.validation_metrics = Metrics(loss_type=hparams.loss_type, n_classes=num_class, metrics=metrics,
                                          prefix="val_", multilabel=dataset.multilabel)
        self.hparams = hparams
        self.data = dataset
        self.head_node_type = self.data.head_node_type

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
        self.training_metrics.update_metrics(Y_hat=y_hat, Y=y, weights=weights)
        loss = self.loss(y_hat, y)
        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        X, y, weights = batch
        y_hat = self.forward(X["adj"], X["x"], X["idx"])
        self.validation_metrics.update_metrics(Y_hat=y_hat, Y=y, weights=weights)
        loss = self.loss(y_hat, y)

        return {"val_loss": loss}

    def test_step(self, batch, batch_nb):
        X, y, weights = batch
        y_hat = self.forward(X["adj"], X["x"], X["idx"])
        loss = self.loss(y_hat, y)

        return {"test_loss": loss}

    def train_dataloader(self):
        return self.data.train_dataloader(collate_fn=self.collate_fn, batch_size=self.hparams.batch_size)

    def val_dataloader(self):
        return self.data.val_dataloader(collate_fn=self.val_collate_fn, batch_size=self.hparams.batch_size)

    def test_dataloader(self):
        return self.data.test_dataloader(collate_fn=self.val_collate_fn, batch_size=self.hparams.batch_size)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


class HAN(HAN, MetricsComparison):
    def __init__(self, hparams, dataset: HeterogeneousNetworkDataset, metrics=["precision"]):
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

        self.training_metrics = Metrics(loss_type=hparams.loss_type, n_classes=num_class, metrics=metrics, prefix=None,
                                        multilabel=dataset.multilabel)
        self.validation_metrics = Metrics(loss_type=hparams.loss_type, n_classes=num_class, metrics=metrics,
                                          prefix="val_",
                                          multilabel=dataset.multilabel)
        self.hparams = hparams
        self.data = dataset
        self.head_node_type = self.data.head_node_type

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
        self.training_metrics.update_metrics(Y_hat=y_hat, Y=y, weights=weights)
        loss = self.loss(y_hat, y)
        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        X, y, weights = batch
        y_hat = self.forward(X["adj"], X["x"], X["idx"])
        self.validation_metrics.update_metrics(Y_hat=y_hat, Y=y, weights=weights)
        loss = self.loss(y_hat, y)

        return {"val_loss": loss}

    def test_step(self, batch, batch_nb):
        X, y, weights = batch
        y_hat = self.forward(X["adj"], X["x"], X["idx"])
        loss = self.loss(y_hat, y)

        return {"test_loss": loss}

    def train_dataloader(self):
        return self.data.train_dataloader(collate_fn=self.collate_fn, batch_size=self.hparams.batch_size)

    def val_dataloader(self):
        return self.data.val_dataloader(collate_fn=self.val_collate_fn, batch_size=self.hparams.batch_size)

    def test_dataloader(self):
        return self.data.test_dataloader(collate_fn=self.val_collate_fn, batch_size=self.hparams.batch_size)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


class MetaPath2Vec(MetaPath2Vec, MetricsComparison):
    def __init__(self, hparams, dataset: HeterogeneousNetworkDataset, metrics=None):
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
        self.data = dataset
        num_nodes_dict = None
        metapath = self.data.metapath
        self.head_node_type = self.data.head_node_type
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
                             batch=self.data.y_index_dict[self.head_node_type][self.data.training_idx])
            y = self.data.y_dict[self.head_node_type][self.data.training_idx]

            perm = torch.randperm(z.size(0))
            train_perm = perm[:int(z.size(0) * self.data.train_ratio)]
            test_perm = perm[int(z.size(0) * self.data.train_ratio):]

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
                                   batch=self.data.y_index_dict[self.head_node_type][self.data.training_idx])
            y_train = self.data.y_dict[self.head_node_type][self.data.training_idx]

            z_val = self.forward(self.head_node_type,
                                 batch=self.data.y_index_dict[self.head_node_type][self.data.validation_idx])
            y_val = self.data.y_dict[self.head_node_type][self.data.validation_idx]

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
        return self.data.train_dataloader(collate_fn=self.sample, batch_size=self.hparams.batch_size)

    def val_dataloader(self):
        return self.data.val_dataloader(collate_fn=self.sample, batch_size=self.hparams.batch_size)

    def test_dataloader(self):
        return self.data.test_dataloader(collate_fn=self.sample, batch_size=self.hparams.batch_size)

    def configure_optimizers(self):
        if self.sparse:
            return torch.optim.SparseAdam(self.parameters(), lr=self.hparams.lr)
        else:
            return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
