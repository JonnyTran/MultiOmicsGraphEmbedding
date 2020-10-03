import multiprocessing
import itertools
import pytorch_lightning as pl
import pandas as pd
import torch
from cogdl.models.nn.pyg_gtn import GTN as Gtn
from cogdl.models.nn.pyg_han import HAN as Han
from cogdl.models.emb.hin2vec import Hin2vec, Hin2vec_layer, RWgraph, tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import precision_recall_fscore_support
from torch.nn import functional as F
from torch_geometric.nn import MetaPath2Vec as Metapath2vec

from torch.optim.lr_scheduler import ReduceLROnPlateau

from moge.generator.sampler.datasets import HeteroNetDataset
from moge.module.metrics import Metrics
from moge.module.PyG.latte import LATTE
from moge.module.classifier import DenseClassification
from moge.module.losses import ClassificationLoss
from moge.module.utils import filter_samples


class NodeClfMetrics(pl.LightningModule):
    def __init__(self, hparams, dataset, metrics, *args):
        super().__init__(*args)

        self.train_metrics = Metrics(prefix="", loss_type=hparams.loss_type, n_classes=dataset.n_classes,
                                     multilabel=dataset.multilabel, metrics=metrics)
        self.valid_metrics = Metrics(prefix="val_", loss_type=hparams.loss_type, n_classes=dataset.n_classes,
                                     multilabel=dataset.multilabel, metrics=metrics)
        self.test_metrics = Metrics(prefix="test_", loss_type=hparams.loss_type, n_classes=dataset.n_classes,
                                    multilabel=dataset.multilabel, metrics=metrics)
        hparams.name = self.name()
        hparams.inductive = dataset.inductive
        self.hparams = hparams

    def name(self):
        if hasattr(self, "_name"):
            return self._name
        else:
            return self.__class__.__name__

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean().item()
        logs = self.train_metrics.compute_metrics()
        # logs = _fix_dp_return_type(logs, device=outputs[0]["loss"].device)

        logs.update({"loss": avg_loss})
        self.train_metrics.reset_metrics()
        return {"log": logs}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean().item()
        logs = self.valid_metrics.compute_metrics()
        # logs = _fix_dp_return_type(logs, device=outputs[0]["val_loss"].device)
        # print({k: np.around(v.item(), decimals=3) for k, v in logs.items()})

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

    def get_n_params(self):
        size = 0
        for name, param in dict(self.named_parameters()).items():
            nn = 1
            for s in list(param.size()):
                nn = nn * s
            size += nn
        return size


class LATTENodeClassifier(NodeClfMetrics):
    def __init__(self, hparams, dataset: HeteroNetDataset, metrics=["accuracy"], collate_fn="neighbor_sampler") -> None:
        super(LATTENodeClassifier, self).__init__(hparams=hparams, dataset=dataset, metrics=metrics)
        self.head_node_type = dataset.head_node_type
        self.dataset = dataset
        self.multilabel = dataset.multilabel
        self.y_types = list(dataset.y_dict.keys())
        self._name = f"LATTE-{hparams.t_order}{' proximity' if hparams.use_proximity else ''}"
        self.collate_fn = collate_fn

        self.latte = LATTE(t_order=hparams.t_order, embedding_dim=hparams.embedding_dim,
                           in_channels_dict=dataset.node_attr_shape, num_nodes_dict=dataset.num_nodes_dict,
                           metapaths=dataset.get_metapaths(), activation=hparams.activation,
                           attn_heads=hparams.attn_heads, attn_activation=hparams.attn_activation,
                           attn_dropout=hparams.attn_dropout, use_proximity=hparams.use_proximity,
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
        self.hparams.n_params = self.get_n_params()

    def forward(self, input: dict, **kwargs):
        embeddings, proximity_loss, _ = self.latte.forward(X=input["x_dict"], edge_index_dict=input["edge_index_dict"],
                                                           global_node_idx=input["global_node_index"], **kwargs)
        y_hat = self.classifier.forward(embeddings[self.head_node_type])
        return y_hat, proximity_loss

    def training_step(self, batch, batch_nb):
        X, y, weights = batch
        y_hat, proximity_loss = self.forward(X)

        if isinstance(y, dict) and len(y) > 1:
            y = y[self.head_node_type]

        y_hat, y = filter_samples(Y_hat=y_hat, Y=y, weights=weights)
        loss = self.criterion.forward(y_hat, y)

        self.train_metrics.update_metrics(y_hat, y, weights=None)

        logs = None
        if self.hparams.use_proximity:
            loss = loss + proximity_loss
            logs = {"proximity_loss": proximity_loss}

        outputs = {'loss': loss}
        if logs is not None:
            outputs.update({'progress_bar': logs, "logs": logs})
        return outputs

    def validation_step(self, batch, batch_nb):
        X, y, weights = batch
        y_hat, proximity_loss = self.forward(X)

        if isinstance(y, dict) and len(y) > 1:
            y = y[self.head_node_type]

        y_hat, y = filter_samples(Y_hat=y_hat, Y=y, weights=weights)
        val_loss = self.criterion.forward(y_hat, y)
        # if batch_nb == 0:
        #     self.print_pred_class_counts(y_hat, y, multilabel=self.dataset.multilabel)

        self.valid_metrics.update_metrics(y_hat, y, weights=None)

        if self.hparams.use_proximity:
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

        if self.hparams.use_proximity:
            test_loss = test_loss + proximity_loss

        return {"test_loss": test_loss}

    def train_dataloader(self):
        return self.dataset.train_dataloader(collate_fn=self.collate_fn,
                                             batch_size=self.hparams.batch_size,
                                             num_workers=int(0.4 * multiprocessing.cpu_count()))

    def val_dataloader(self, batch_size=None):
        return self.dataset.valid_dataloader(collate_fn=self.collate_fn,
                                             batch_size=self.hparams.batch_size,
                                             num_workers=max(1, int(0.1 * multiprocessing.cpu_count())))

    def valtrain_dataloader(self):
        return self.dataset.valtrain_dataloader(collate_fn=self.collate_fn,
                                                batch_size=self.hparams.batch_size,
                                                num_workers=max(1, int(0.1 * multiprocessing.cpu_count())))

    def test_dataloader(self, batch_size=None):
        return self.dataset.test_dataloader(collate_fn=self.collate_fn,
                                            batch_size=self.hparams.batch_size,
                                            num_workers=max(1, int(0.1 * multiprocessing.cpu_count())))

    def configure_optimizers(self):
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'alpha_activation']
        optimizer_grouped_parameters = [
            {'params': [p for name, p in param_optimizer if not any(key in name for key in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for name, p in param_optimizer if any(key in name for key in no_decay)], 'weight_decay': 0.0}
        ]

        # optimizer = torch.optim.AdamW(optimizer_grouped_parameters, eps=1e-06, lr=self.hparams.lr)

        optimizer = torch.optim.Adam(optimizer_grouped_parameters,
                                     lr=self.hparams.lr,  # momentum=self.hparams.momentum,
                                     weight_decay=self.hparams.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer)

        return [optimizer], [scheduler]


class GTN(Gtn, pl.LightningModule):
    def __init__(self, hparams, dataset: HeteroNetDataset, metrics=["precision"]):
        num_edge = len(dataset.edge_index_dict)
        num_layers = hparams.num_layers
        num_class = dataset.n_classes
        self.multilabel = dataset.multilabel
        self.head_node_type = dataset.head_node_type
        self.collate_fn = hparams.collate_fn
        num_nodes = dataset.num_nodes_dict[dataset.head_node_type]

        if dataset.in_features:
            w_in = dataset.in_features
        else:
            w_in = hparams.embedding_dim

        w_out = hparams.embedding_dim
        num_channels = hparams.num_channels
        super(GTN, self).__init__(num_edge, num_channels, w_in, w_out, num_class, num_nodes, num_layers)

        if not hasattr(dataset, "x") and not hasattr(dataset, "x_dict"):
            if num_nodes > 10000:
                self.embedding = {self.head_node_type: torch.nn.Embedding(num_embeddings=num_nodes,
                                                                          embedding_dim=hparams.embedding_dim).cpu()}
            else:
                self.embedding = torch.nn.Embedding(num_embeddings=num_nodes, embedding_dim=hparams.embedding_dim)

        self.dataset = dataset
        self.head_node_type = self.dataset.head_node_type
        hparams.n_params = self.get_n_params()
        self.train_metrics = Metrics(prefix="", loss_type=hparams.loss_type, n_classes=dataset.n_classes,
                                     multilabel=dataset.multilabel, metrics=metrics)
        self.valid_metrics = Metrics(prefix="val_", loss_type=hparams.loss_type, n_classes=dataset.n_classes,
                                     multilabel=dataset.multilabel, metrics=metrics)
        self.test_metrics = Metrics(prefix="test_", loss_type=hparams.loss_type, n_classes=dataset.n_classes,
                                    multilabel=dataset.multilabel, metrics=metrics)
        hparams.name = self.name()
        hparams.inductive = dataset.inductive
        self.hparams = hparams

    def name(self):
        if hasattr(self, "_name"):
            return self._name
        else:
            return self.__class__.__name__

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean().item()
        logs = self.train_metrics.compute_metrics()
        # logs = _fix_dp_return_type(logs, device=outputs[0]["loss"].device)

        logs.update({"loss": avg_loss})
        self.train_metrics.reset_metrics()
        return {"log": logs}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean().item()
        logs = self.valid_metrics.compute_metrics()
        # logs = _fix_dp_return_type(logs, device=outputs[0]["val_loss"].device)
        # print({k: np.around(v.item(), decimals=3) for k, v in logs.items()})

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

    def get_n_params(self):
        size = 0
        for name, param in dict(self.named_parameters()).items():
            nn = 1
            for s in list(param.size()):
                nn = nn * s
            size += nn
        return size

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
        y_hat, y = filter_samples(Y_hat=y_hat, Y=y, weights=weights)
        loss = self.loss(y_hat, y)
        self.test_metrics.update_metrics(y_hat, y, weights=None)

        return {"test_loss": loss}

    def train_dataloader(self):
        return self.dataset.train_dataloader(collate_fn=self.collate_fn, batch_size=self.hparams.batch_size)

    def val_dataloader(self):
        return self.dataset.valid_dataloader(collate_fn=self.collate_fn, batch_size=self.hparams.batch_size)

    def valtrain_dataloader(self):
        return self.dataset.valtrain_dataloader(collate_fn=self.collate_fn,
                                                batch_size=self.hparams.batch_size,
                                                num_workers=max(1, int(0.1 * multiprocessing.cpu_count())))

    def test_dataloader(self):
        return self.dataset.test_dataloader(collate_fn=self.collate_fn, batch_size=self.hparams.batch_size)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


class HAN(Han, pl.LightningModule):
    def __init__(self, hparams, dataset: HeteroNetDataset, metrics=["precision"]):
        num_edge = len(dataset.edge_index_dict)
        num_layers = hparams.num_layers
        num_class = dataset.n_classes
        self.collate_fn = hparams.collate_fn
        self.multilabel = dataset.multilabel
        num_nodes = dataset.num_nodes_dict[dataset.head_node_type]

        if dataset.in_features:
            w_in = dataset.in_features
        else:
            w_in = hparams.embedding_dim

        w_out = hparams.embedding_dim

        super(HAN, self).__init__(num_edge=num_edge, w_in=w_in, w_out=w_out, num_class=num_class,
                                  num_nodes=num_nodes, num_layers=num_layers)

        if not hasattr(dataset, "x") and not hasattr(dataset, "x_dict"):
            if num_nodes > 10000:
                self.embedding = {dataset.head_node_type: torch.nn.Embedding(num_embeddings=num_nodes,
                                                                             embedding_dim=hparams.embedding_dim).cpu()}
            else:
                self.embedding = torch.nn.Embedding(num_embeddings=num_nodes, embedding_dim=hparams.embedding_dim)

        self.dataset = dataset
        self.head_node_type = self.dataset.head_node_type
        hparams.n_params = self.get_n_params()
        self.train_metrics = Metrics(prefix="", loss_type=hparams.loss_type, n_classes=dataset.n_classes,
                                     multilabel=dataset.multilabel, metrics=metrics)
        self.valid_metrics = Metrics(prefix="val_", loss_type=hparams.loss_type, n_classes=dataset.n_classes,
                                     multilabel=dataset.multilabel, metrics=metrics)
        self.test_metrics = Metrics(prefix="test_", loss_type=hparams.loss_type, n_classes=dataset.n_classes,
                                    multilabel=dataset.multilabel, metrics=metrics)
        hparams.name = self.name()
        hparams.inductive = dataset.inductive
        self.hparams = hparams

    def name(self):
        if hasattr(self, "_name"):
            return self._name
        else:
            return self.__class__.__name__

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean().item()
        logs = self.train_metrics.compute_metrics()
        # logs = _fix_dp_return_type(logs, device=outputs[0]["loss"].device)

        logs.update({"loss": avg_loss})
        self.train_metrics.reset_metrics()
        return {"log": logs}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean().item()
        logs = self.valid_metrics.compute_metrics()
        # logs = _fix_dp_return_type(logs, device=outputs[0]["val_loss"].device)
        # print({k: np.around(v.item(), decimals=3) for k, v in logs.items()})

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

    def get_n_params(self):
        size = 0
        for name, param in dict(self.named_parameters()).items():
            nn = 1
            for s in list(param.size()):
                nn = nn * s
            size += nn
        return size

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
        return self.dataset.valid_dataloader(collate_fn=self.collate_fn, batch_size=self.hparams.batch_size)

    def valtrain_dataloader(self):
        return self.dataset.valtrain_dataloader(collate_fn=self.collate_fn,
                                                batch_size=self.hparams.batch_size)

    def test_dataloader(self):
        return self.dataset.test_dataloader(collate_fn=self.collate_fn, batch_size=self.hparams.batch_size)

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
        hparams.n_params = self.get_n_params()
        hparams.inductive = dataset.inductive
        self.hparams = hparams

    def get_n_params(self):
        size = 0
        for name, param in dict(self.named_parameters()).items():
            nn = 1
            for s in list(param.size()):
                nn = nn * s
            size += nn
        return size

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
        if self.current_epoch % 10 == 0:
            results = self.classification_results(training=True)
        else:
            results = {}

        return {"progress_bar": results,
                "log": {"loss": avg_loss, **results}}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).sum().item()
        logs = {"val_loss": avg_loss}
        if self.current_epoch % 5 == 0:
            logs.update({"val_" + k: v for k, v in self.classification_results(training=False).items()})
        return {"progress_bar": logs, "log": logs}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).sum().item()
        logs = {"test_loss": avg_loss}
        logs.update({"test_" + k: v for k, v in self.classification_results(training=False, testing=True).items()})
        return {"progress_bar": logs, "log": logs}

    def classification_results(self, training=True, testing=False):
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
                             batch=self.dataset.validation_idx if not testing else self.dataset.testing_idx)
            y = self.dataset.y_dict[self.head_node_type][
                self.dataset.validation_idx if not testing else self.dataset.testing_idx]

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

    def pos_sample(self, batch):
        # device = self.embedding.weight.device

        batch = batch.repeat(self.walks_per_node)

        rws = [batch]
        for i in range(self.walk_length):
            keys = self.metapath[i % len(self.metapath)]
            adj = self.adj_dict[keys]
            batch = adj.sample(num_neighbors=1, subset=batch).squeeze()
            rws.append(batch)

        rw = torch.stack(rws, dim=-1)
        rw.add_(self.offset.view(1, -1))

        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.context_size])
        return torch.cat(walks, dim=0)

    def neg_sample(self, batch):
        batch = batch.repeat(self.walks_per_node * self.num_negative_samples)

        rws = [batch]
        for i in range(self.walk_length):
            keys = self.metapath[i % len(self.metapath)]
            batch = torch.randint(0, self.num_nodes_dict[keys[-1]],
                                  (batch.size(0),), dtype=torch.long)
            rws.append(batch)

        rw = torch.stack(rws, dim=-1)
        rw.add_(self.offset.view(1, -1))

        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.context_size])
        return torch.cat(walks, dim=0)

    def sample(self, batch):
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch)
        return self.pos_sample(batch), self.neg_sample(batch)

    def train_dataloader(self):
        return self.dataset.train_dataloader(collate_fn=self.sample, batch_size=self.hparams.batch_size)

    def val_dataloader(self):
        return self.dataset.valid_dataloader(collate_fn=self.sample, batch_size=self.hparams.batch_size)

    def valtrain_dataloader(self):
        return self.dataset.valtrain_dataloader(collate_fn=self.sample,
                                                batch_size=self.hparams.batch_size)

    def test_dataloader(self):
        return self.dataset.test_dataloader(collate_fn=self.sample, batch_size=self.hparams.batch_size)

    def configure_optimizers(self):
        if self.sparse:
            return torch.optim.SparseAdam(self.parameters(), lr=self.hparams.lr)
        else:
            return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


class HIN2Vec(Hin2vec):
    def __init__(self, hparams, dataset: HeteroNetDataset, metrics=None):
        self.train_ratio = hparams.train_ratio
        self.batch_size = hparams.batch_size
        self.sparse = hparams.sparse

        embedding_dim = hparams.embedding_dim
        walk_length = hparams.walk_length
        hop = hparams.context_size
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

        super().__init__(embedding_dim, walk_length, walks_per_node, hparams.batch_size, hop, num_negative_samples,
                         1000, hparams.lr, cpu=True)

    def train(self, G, node_type):
        self.num_node = G.number_of_nodes()
        rw = RWgraph(G, node_type)
        walks = rw._simulate_walks(self.walk_length, self.walk_num)
        pairs, relation = rw.data_preparation(walks, self.hop, self.negative)

        self.num_relation = len(relation)
        model = Hin2vec_layer(self.num_node, self.num_relation, self.hidden_dim, self.cpu)
        self.model = model.to(self.device)

        num_batch = int(len(pairs) / self.batch_size)
        print_num_batch = 100
        print("number of batch", num_batch)

        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        epoch_iter = tqdm(range(self.epoches))
        for epoch in epoch_iter:
            loss_n, pred, label = [], [], []
            for i in range(num_batch):
                batch_pairs = torch.from_numpy(pairs[i * self.batch_size:(i + 1) * self.batch_size])
                batch_pairs = batch_pairs.to(self.device)
                batch_pairs = batch_pairs.T
                x, y, r, l = batch_pairs[0], batch_pairs[1], batch_pairs[2], batch_pairs[3]
                opt.zero_grad()
                logits, loss = self.model.forward(x, y, r, l)

                loss_n.append(loss.item())
                label.append(l)
                pred.extend(logits)
                if i % print_num_batch == 0 and i != 0:
                    label = torch.cat(label).to(self.device)
                    pred = torch.stack(pred, dim=0)
                    pred = pred.max(1)[1]
                    acc = pred.eq(label).sum().item() / len(label)
                    epoch_iter.set_description(
                        f"Epoch: {i:03d}, Loss: {sum(loss_n) / print_num_batch:.5f}, Acc: {acc:.5f}"
                    )
                    loss_n, pred, label = [], [], []

                loss.backward()
                opt.step()

        embedding = self.model.get_emb()
        return embedding.cpu().detach().numpy()

    def get_n_params(self):
        size = 0
        for name, param in dict(self.named_parameters()).items():
            nn = 1
            for s in list(param.size()):
                nn = nn * s
            size += nn
        return size

    def train_dataloader(self):
        return self.dataset.train_dataloader(collate_fn=self.sample, batch_size=self.hparams.batch_size)

    def val_dataloader(self):
        return self.dataset.valid_dataloader(collate_fn=self.sample, batch_size=self.hparams.batch_size)

    def valtrain_dataloader(self):
        return self.dataset.valtrain_dataloader(collate_fn=self.sample,
                                                batch_size=self.hparams.batch_size)

    def test_dataloader(self):
        return self.dataset.test_dataloader(collate_fn=self.sample, batch_size=self.hparams.batch_size)

    def configure_optimizers(self):
        if self.sparse:
            return torch.optim.SparseAdam(self.parameters(), lr=self.hparams.lr)
        else:
            return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
