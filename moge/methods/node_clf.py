import multiprocessing
import itertools
import numpy as np
import pytorch_lightning as pl
import pandas as pd
import torch
from cogdl.models.nn.pyg_gtn import GTN as Gtn
from cogdl.models.nn.pyg_han import HAN as Han
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import precision_recall_fscore_support
from torch.nn import functional as F
from torch_geometric.nn import MetaPath2Vec as Metapath2vec
from torch_geometric.utils import remove_self_loops, add_self_loops

from torch.optim.lr_scheduler import ReduceLROnPlateau

from moge.generator.sampler.datasets import HeteroNetDataset
from moge.module.metrics import Metrics
from moge.module.trainer import _fix_dp_return_type
from moge.module.latte import LATTE
from moge.module.classifier import DenseClassification, MulticlassClassification
from moge.module.losses import ClassificationLoss
from moge.module.utils import filter_samples, pad_tensors, tensor_sizes


class NodeClfMetrics(pl.LightningModule):
    def __init__(self, hparams, dataset, metrics, *args, **kwargs):
        super(NodeClfMetrics, self).__init__(*args, **kwargs)
        self.train_metrics = Metrics(prefix="", loss_type=hparams.loss_type, n_classes=dataset.n_classes,
                                     multilabel=dataset.multilabel, metrics=metrics)
        self.valid_metrics = Metrics(prefix="val_", loss_type=hparams.loss_type, n_classes=dataset.n_classes,
                                     multilabel=dataset.multilabel, metrics=metrics)
        self.test_metrics = Metrics(prefix="test_", loss_type=hparams.loss_type, n_classes=dataset.n_classes,
                                    multilabel=dataset.multilabel, metrics=metrics)
        hparams.name = self.name()
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
        print({k: np.around(v.item(), decimals=3) for k, v in logs.items()})

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
            print(f"y_true {len(y_true_dict)} classes",
                  {str(k): v for k, v in itertools.islice(y_true_dict.items(), n_top_class)})
        else:
            y_pred_dict = pd.Series(y_hat.argmax(1).detach().cpu().type(torch.int).numpy()).value_counts().to_dict()
            y_true_dict = pd.Series(y.detach().cpu().type(torch.int).numpy()).value_counts().to_dict()
            print(f"y_pred {len(y_pred_dict)} classes",
                  {str(k): v for k, v in itertools.islice(y_pred_dict.items(), n_top_class)})
            print(f"y_true {len(y_true_dict)} classes",
                  {str(k): v for k, v in itertools.islice(y_true_dict.items(), n_top_class)})


class LATTENodeClassifier(NodeClfMetrics):
    def __init__(self, hparams, dataset: HeteroNetDataset, metrics=["accuracy"], collate_fn="neighbor_sampler") -> None:
        super(LATTENodeClassifier, self).__init__(hparams=hparams, dataset=dataset, metrics=metrics)
        self.head_node_type = dataset.head_node_type
        self.dataset = dataset
        self.multilabel = dataset.multilabel
        self.y_types = list(dataset.y_dict.keys())
        self._name = f"LATTE-{hparams.t_order}{' proximity' if hparams.use_proximity_loss else ''}"
        self.collate_fn = collate_fn

        self.latte = LATTE(in_channels_dict=dataset.node_attr_shape, embedding_dim=hparams.embedding_dim,
                           t_order=hparams.t_order, num_nodes_dict=dataset.num_nodes_dict,
                           metapaths=dataset.get_metapaths(), activation=hparams.activation,
                           attn_heads=hparams.attn_heads, attn_activation=hparams.attn_activation,
                           attn_dropout=hparams.attn_dropout, use_proximity_loss=hparams.use_proximity_loss,
                           neg_sampling_ratio=hparams.neg_sampling_ratio)
        hparams.embedding_dim = hparams.embedding_dim * hparams.t_order
        self.classifier = DenseClassification(hparams)
        # self.classifier = MulticlassClassification(num_feature=hparams.embedding_dim,
        #                                            num_class=hparams.n_classes,
        #                                            loss_type=hparams.loss_type)
        self.criterion = ClassificationLoss(n_classes=dataset.n_classes,
                                            class_weight=dataset.class_weight if hasattr(dataset, "class_weight") and \
                                                                                 hparams.use_class_weights else None,
                                            loss_type=hparams.loss_type,
                                            multilabel=dataset.multilabel)

    def forward(self, X: dict, **kwargs):
        embeddings, proximity_loss, _ = self.latte.forward(X["x_dict"], X["global_node_index"], X["edge_index_dict"],
                                                           **kwargs)

        y_hat = self.classifier.forward(embeddings[self.head_node_type])
        return y_hat, proximity_loss

    def training_step(self, batch, batch_nb):
        X, y, weights = batch
        y_hat, proximity_loss = self.forward(X)

        if isinstance(y, dict) and len(y) > 1:
            y = y[self.head_node_type]

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
        y_hat, proximity_loss = self.forward(X)
        if isinstance(y, dict) and len(y) > 1:
            y = y[self.head_node_type]
        y_hat, y = filter_samples(Y_hat=y_hat, Y=y, weights=weights)
        val_loss = self.criterion(y_hat, y)

        self.valid_metrics.update_metrics(y_hat, y, weights=None)

        if self.hparams.use_proximity_loss:
            val_loss = val_loss + proximity_loss

        return {"val_loss": val_loss}

    def test_step(self, batch, batch_nb):
        X, y, weights = batch
        y_hat, proximity_loss = self.forward(X, save_betas=True)
        if isinstance(y, dict) and len(y) > 1:
            y = y[self.head_node_type]
        y_hat, y = filter_samples(Y_hat=y_hat, Y=y, weights=weights)
        test_loss = self.criterion(y_hat, y)

        if batch_nb == 0:
            self.print_pred_class_counts(y_hat, y, multilabel=self.dataset.multilabel)

        self.test_metrics.update_metrics(y_hat, y, weights=None)

        if self.hparams.use_proximity_loss:
            test_loss = test_loss + proximity_loss

        return {"test_loss": test_loss}

    def train_dataloader(self):
        return self.dataset.train_dataloader(collate_fn=self.collate_fn,
                                             batch_size=self.hparams.batch_size,
                                             num_workers=int(0.4 * multiprocessing.cpu_count()),
                                             t_order=self.hparams.t_order)

    def val_dataloader(self, batch_size=None):
        return self.dataset.val_dataloader(collate_fn=self.collate_fn,
                                           batch_size=self.hparams.batch_size,
                                           num_workers=max(1, int(0.1 * multiprocessing.cpu_count())),
                                           t_order=self.hparams.t_order)

    def test_dataloader(self, batch_size=None):
        return self.dataset.test_dataloader(collate_fn=self.collate_fn,
                                            batch_size=self.hparams.batch_size,
                                            num_workers=max(1, int(0.1 * multiprocessing.cpu_count())),
                                            t_order=self.hparams.t_order)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.hparams.lr,  # momentum=self.hparams.momentum,
                                     weight_decay=self.hparams.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer)

        return [optimizer], [scheduler]


class GTN(NodeClfMetrics, Gtn):
    def __init__(self, hparams, dataset: HeteroNetDataset, metrics=["precision"]):
        num_edge = len(dataset.edge_index_dict)
        num_layers = len(dataset.edge_index_dict)
        num_class = dataset.n_classes
        self.multilabel = dataset.multilabel
        self.head_node_type = dataset.head_node_type
        self.collate_fn = hparams.collate_fn
        self.val_collate_fn = hparams.val_collate_fn
        num_nodes = dataset.num_nodes_dict[dataset.head_node_type]

        if dataset.in_features:
            w_in = dataset.in_features
        else:
            w_in = hparams.embedding_dim

        w_out = hparams.embedding_dim
        num_channels = hparams.num_channels
        super().__init__(hparams, dataset, metrics, num_edge, num_channels, w_in, w_out, num_class, num_nodes,
                         num_layers)

        if not hasattr(dataset, "x"):
            if num_nodes > 10000:
                self.embedding = {self.head_node_type: torch.nn.Embedding(num_embeddings=num_nodes,
                                                                          embedding_dim=hparams.embedding_dim).cpu()}
            else:
                self.embedding = torch.nn.Embedding(num_embeddings=num_nodes, embedding_dim=hparams.embedding_dim)

        self.dataset = dataset
        self.head_node_type = self.dataset.head_node_type

    def forward(self, A, X, x_idx):
        if X is None:
            if isinstance(self.embedding, dict):
                X = self.embedding[self.head_node_type].weight[x_idx].to(self.layers[0].device)
            else:
                X = self.embedding.weight[x_idx]

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
        if x_idx is not None and X_.size(0) > x_idx.size(0):
            y = self.linear2(X_[x_idx])
        else:
            y = self.linear2(X_)

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
        loss = self.loss(y_hat, y)
        self.valid_metrics.update_metrics(y_hat, y, weights=None)

        return {"val_loss": loss}

    def test_step(self, batch, batch_nb):
        X, y, weights = batch
        y_hat = self.forward(X["adj"], X["x"], X["idx"])
        loss = self.loss(y_hat, y)
        y_hat, y = filter_samples(Y_hat=y_hat, Y=y, weights=weights)
        self.test_metrics.update_metrics(y_hat, y, weights=None)

        return {"test_loss": loss}

    def train_dataloader(self):
        return self.dataset.train_dataloader(collate_fn=self.collate_fn, batch_size=self.hparams.batch_size)

    def val_dataloader(self):
        return self.dataset.val_dataloader(collate_fn=self.val_collate_fn, batch_size=self.hparams.batch_size * 2)

    def test_dataloader(self):
        return self.dataset.test_dataloader(collate_fn=self.val_collate_fn, batch_size=self.hparams.batch_size * 2)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


class HAN(NodeClfMetrics, Han):
    def __init__(self, hparams, dataset: HeteroNetDataset, metrics=["precision"]):
        num_edge = len(dataset.edge_index_dict)
        num_layers = len(dataset.edge_index_dict)
        num_class = dataset.n_classes
        self.collate_fn = hparams.collate_fn
        self.val_collate_fn = hparams.val_collate_fn
        self.multilabel = dataset.multilabel
        num_nodes = dataset.num_nodes_dict[dataset.head_node_type]

        if dataset.in_features:
            w_in = dataset.in_features
        else:
            w_in = hparams.embedding_dim

        w_out = hparams.embedding_dim

        super().__init__(hparams, dataset, metrics, num_edge=num_edge, w_in=w_in, w_out=w_out, num_class=num_class,
                         num_nodes=num_nodes, num_layers=num_layers)

        if not hasattr(dataset, "x"):
            if num_nodes > 10000:
                self.embedding = {self.head_node_type: torch.nn.Embedding(num_embeddings=num_nodes,
                                                                          embedding_dim=hparams.embedding_dim).cpu()}
            else:
                self.embedding = torch.nn.Embedding(num_embeddings=num_nodes, embedding_dim=hparams.embedding_dim)

        self.hparams = hparams
        self.dataset = dataset
        self.head_node_type = self.dataset.head_node_type

    def forward(self, A, X, x_idx):
        if X is None:
            if isinstance(self.embedding, dict):
                X = self.embedding[self.head_node_type].weight[x_idx].to(self.layers[0].device)
            else:
                X = self.embedding.weight[x_idx]


        for i in range(self.num_layers):
            X = self.layers[i].forward(X, A, )

        if x_idx is not None and X.size(0) > x_idx.size(0):
            y = self.linear(X[x_idx])
        else:
            y = self.linear(X)
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
        self.test_metrics.update_metrics(y_hat, y, weights=None)
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


class MetaPath2Vec(Metapath2vec, pl.LightningModule):
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
        metapaths = self.dataset.get_metapaths()
        self.head_node_type = self.dataset.head_node_type
        edge_index_dict = dataset.edge_index_dict
        first_node_type = metapaths[0][0]

        for metapath in reversed(metapaths):
            if metapath[-1] == first_node_type:
                last_metapath = metapath
                break
        metapaths.pop(metapaths.index(last_metapath))
        metapaths.append(last_metapath)
        print("metapaths", metapaths)

        if dataset.use_reverse:
            dataset.add_reverse_edge_index(dataset.edge_index_dict)

        super(MetaPath2Vec, self).__init__(edge_index_dict, embedding_dim,
                                           metapaths, walk_length, context_size,
                                           walks_per_node,
                                           num_negative_samples, num_nodes_dict,
                                           hparams.sparse)

        hparams.name = self.name()
        self.hparams = hparams

    def name(self):
        if hasattr(self, "_name"):
            return self._name
        else:
            return self.__class__.__name__

    def forward(self, node_type, batch=None):
        return super().forward(node_type, batch=batch)

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
        results = self.classification_results(training=True)

        return {"progress_bar": results,
                "log": {"loss": avg_loss, **results}}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).sum().item()
        logs = {"val_loss": avg_loss}
        logs.update({"val_" + k: v for k, v in self.classification_results(training=False).items()})
        return {"progress_bar": logs, "log": logs}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).sum().item()
        logs = {"test_loss": avg_loss}
        logs.update({"test_" + k: v for k, v in self.classification_results(training=False).items()})
        return {"progress_bar": logs, "log": logs}

    def classification_results(self, training=True):
        if training:
            z = self.forward(self.head_node_type,
                             batch=self.dataset.training_idx)
            y = self.dataset.y_dict[self.head_node_type][self.dataset.training_idx]

            perm = torch.randperm(z.size(0))
            train_perm = perm[:int(z.size(0) * self.dataset.train_ratio)]
            test_perm = perm[int(z.size(0) * self.dataset.train_ratio):]

            if y.dim() > 1 and y.size(1) > 1:
                multilabel = True
                clf = OneVsRestClassifier(LogisticRegression(solver="lbfgs", multi_class="auto", max_iter=150))
            else:
                multilabel = False
                clf = LogisticRegression(solver="lbfgs", multi_class="auto", max_iter=150)

            clf.fit(z[train_perm].detach().cpu().numpy(),
                    y[train_perm].detach().cpu().numpy())

            y_pred = clf.predict(z[test_perm].detach().cpu().numpy())
            y_test = y[test_perm].detach().cpu().numpy()
        else:
            z_train = self.forward(self.head_node_type,
                                   batch=self.dataset.training_idx)
            y_train = self.dataset.y_dict[self.head_node_type][self.dataset.training_idx]

            z = self.forward(self.head_node_type,
                             batch=self.dataset.validation_idx)
            y = self.dataset.y_dict[self.head_node_type][self.dataset.validation_idx]

            if y_train.dim() > 1 and y_train.size(1) > 1:
                multilabel = True
                clf = OneVsRestClassifier(LogisticRegression(solver="lbfgs", multi_class="auto", max_iter=150))
            else:
                multilabel = False
                clf = LogisticRegression(solver="lbfgs", multi_class="auto", max_iter=150)

            clf.fit(z_train.detach().cpu().numpy(), y_train.detach().cpu().numpy())
            y_pred = clf.predict(z.detach().cpu().numpy())
            y_test = y.detach().cpu().numpy()

        result = dict(zip(["precision", "recall", "f1"],
                          precision_recall_fscore_support(y_test, y_pred, average="micro")))
        result["acc" if not multilabel else "accuracy"] = result["precision"]
        return result

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
