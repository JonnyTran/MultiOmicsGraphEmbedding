import logging
from typing import Dict

import pytorch_lightning as pl
import torch
import torch_sparse.sample
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.multiclass import OneVsRestClassifier
from torch.nn import functional as F
from torch import nn
from torch_geometric.nn import MetaPath2Vec as Metapath2vec
from torch.utils.data import DataLoader

from moge.data.network import HeteroNetDataset
from moge.module.PyG.latte import LATTE
from moge.module.classifier import DenseClassification
from moge.module.losses import ClassificationLoss
from moge.module.trainer import NodeClfTrainer, print_pred_class_counts
from moge.module.utils import tensor_sizes


class LATTENodeClf(NodeClfTrainer):
    def __init__(self, hparams, dataset: HeteroNetDataset, metrics=["accuracy"], collate_fn="neighbor_sampler") -> None:
        super(LATTENodeClf, self).__init__(hparams=hparams, dataset=dataset, metrics=metrics)
        self.head_node_type = dataset.head_node_type
        self.node_types = list(dataset.num_nodes_dict.keys())
        self.dataset = dataset
        self.multilabel = dataset.multilabel
        self.y_types = list(dataset.y_dict.keys())
        self._name = f"LATTE-{hparams.t_order}"
        self.collate_fn = collate_fn

        self.embedder = LATTE(n_layers=hparams.n_layers,
                              t_order=min(hparams.t_order, hparams.n_layers),
                              embedding_dim=hparams.embedding_dim,
                              num_nodes_dict=dataset.num_nodes_dict,
                              metapaths=dataset.get_metapaths(khop=True if "khop" in collate_fn else None),
                              activation=hparams.activation,
                              attn_heads=hparams.attn_heads,
                              attn_activation=hparams.attn_activation,
                              attn_dropout=hparams.attn_dropout,
                              use_proximity=hparams.use_proximity if hasattr(hparams, "use_proximity") else False,
                              neg_sampling_ratio=hparams.neg_sampling_ratio if hasattr(hparams,
                                                                                       "neg_sampling_ratio") else None,
                              edge_sampling=hparams.edge_sampling if hasattr(hparams, "edge_sampling") else False,
                              layer_pooling=hparams.layer_pooling,
                              hparams=hparams)

        self.embeddings = self.initialize_embeddings(hparams.embedding_dim,
                                                     dataset.num_nodes_dict,
                                                     dataset.node_attr_shape,
                                                     pretrain_embeddings=hparams.node_emb_init if "node_emb_init" in hparams else None,
                                                     freeze=hparams.freeze_embeddings if "freeze_embeddings" in hparams else True)

        # node types that needs a projection to align to the embedding_dim
        self.proj_ntypes = [ntype for ntype in self.node_types \
                            if (ntype in dataset.node_attr_shape
                                and dataset.node_attr_shape[ntype] != hparams.embedding_dim) \
                            or (self.embeddings and ntype in self.embeddings and
                                self.embeddings[ntype].weight.size(1) != hparams.embedding_dim)]

        self.feature_projection = nn.ModuleDict({
            ntype: nn.Linear(
                in_features=dataset.node_attr_shape[ntype] \
                    if not self.embeddings or ntype not in self.embeddings \
                    else self.embeddings[ntype].weight.size(1),
                out_features=hparams.embedding_dim) \
            for ntype in self.proj_ntypes})

        if hparams.batchnorm:
            self.batchnorm = nn.ModuleDict({
                ntype: nn.BatchNorm1d(hparams.embedding_dim) \
                for ntype in self.proj_ntypes
            })

        # self.dropout = hparams.dropout if hasattr(hparams, "dropout") else 0.0

        if hparams.nb_cls_dense_size >= 0:
            if hparams.layer_pooling == "concat":
                hparams.embedding_dim = hparams.embedding_dim * hparams.n_layers
                logging.info("embedding_dim {}".format(hparams.embedding_dim))
            elif hparams.layer_pooling == "rel_concat":
                hparams.embedding_dim = hparams.embedding_dim * self.embedder.layers[-1].t_order

            self.classifier = DenseClassification(hparams)
        else:
            assert hparams.layer_pooling != "concat", "Layer pooling cannot be concat when output of network is a GNN"

        self.criterion = ClassificationLoss(n_classes=dataset.n_classes,
                                            loss_type=hparams.loss_type,
                                            class_weight=dataset.class_weight if hasattr(dataset, "class_weight") and \
                                                                                 hparams.use_class_weights else None,
                                            multilabel=dataset.multilabel,
                                            reduction="mean" if "reduction" not in hparams else hparams.reduction)

        self.hparams.n_params = self.get_n_params()
        self.lr = self.hparams.lr

        self.val_moving_loss = torch.tensor([2.5, ] * 5, dtype=torch.float)

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        for ntype in self.feature_projection:
            nn.init.xavier_normal_(self.feature_projection[ntype].weight, gain=gain)

    def initialize_embeddings(self, embedding_dim, num_nodes_dict, in_channels_dict,
                              pretrain_embeddings: Dict[str, torch.Tensor],
                              freeze=True):
        # If some node type are not attributed, instantiate nn.Embedding for them
        if isinstance(in_channels_dict, dict):
            non_attr_node_types = (num_nodes_dict.keys() - in_channels_dict.keys())
        else:
            non_attr_node_types = []


        if non_attr_node_types:
            module_dict = {}

            for ntype in non_attr_node_types:
                if pretrain_embeddings is None or ntype not in pretrain_embeddings:
                    print("Initialized trainable embeddings", ntype)
                    module_dict[ntype] = nn.Embedding(num_embeddings=num_nodes_dict[ntype],
                                                      embedding_dim=embedding_dim,
                                                      scale_grad_by_freq=True,
                                                      sparse=False)
                else:
                    print(f"Pretrained embeddings freeze={freeze}", ntype)
                    max_norm = pretrain_embeddings[ntype].norm(dim=1).mean()
                    module_dict[ntype] = nn.Embedding.from_pretrained(pretrain_embeddings[ntype],
                                                                      freeze=freeze,
                                                                      scale_grad_by_freq=True,
                                                                      max_norm=max_norm)

            embeddings = nn.ModuleDict(module_dict)
        else:
            embeddings = None

        return embeddings

    def transform_inp_feats(self, node_feats: Dict[str, torch.Tensor], global_node_idx: Dict[str, torch.Tensor]):
        h_dict = {}

        for ntype in global_node_idx:
            if global_node_idx[ntype].numel() == 0: continue

            if ntype not in node_feats:
                node_feats[ntype] = self.embeddings[ntype](global_node_idx[ntype]).to(self.device)

            # project to embedding_dim if node features are not same same dimension
            if ntype in self.proj_ntypes:
                h_dict[ntype] = self.feature_projection[ntype](node_feats[ntype])

                if hasattr(self, "batchnorm"):
                    h_dict[ntype] = self.batchnorm[ntype](h_dict[ntype])

                h_dict[ntype] = F.relu(h_dict[ntype])
                # if self.dropout:
                #     h_dict[ntype] = F.dropout(h_dict[ntype], p=self.dropout, training=self.training)

            else:
                # Skips projection
                h_dict[ntype] = node_feats[ntype]


        return h_dict

    def forward(self, X: dict, **kwargs):
        if not self.training:
            self._node_ids = X["global_node_index"]

        h_out = self.transform_inp_feats(X["x_dict"], global_node_idx=X["global_node_index"][0])

        embeddings, proximity_loss, edge_index_dict = self.embedder(h_out,
                                                                    X["edge_index"],
                                                                    X["sizes"],
                                                                    X["global_node_index"], **kwargs)
        y_hat = self.classifier(embeddings[self.head_node_type]) \
            if hasattr(self, "classifier") else embeddings[self.head_node_type]

        return y_hat, proximity_loss, edge_index_dict

    def on_test_epoch_start(self) -> None:
        for l in range(self.embedder.n_layers):
            self.embedder.layers[l]._betas = {}
            self.embedder.layers[l]._beta_avg = {}
            self.embedder.layers[l]._beta_std = {}
        super().on_test_epoch_start()

    def on_validation_epoch_end(self) -> None:
        for l in range(self.embedder.n_layers):
            self.embedder.layers[l]._betas = {}
            self.embedder.layers[l]._beta_avg = {}
            self.embedder.layers[l]._beta_std = {}
        super().on_validation_epoch_end()

    def on_predict_epoch_start(self) -> None:
        for l in range(self.embedder.n_layers):
            self.embedder.layers[l]._betas = {}
            self.embedder.layers[l]._beta_avg = {}
            self.embedder.layers[l]._beta_std = {}
        super().on_predict_epoch_start()

    def training_step(self, batch, batch_nb):
        X, y_true, weights = batch
        y_pred, proximity_loss, _ = self.forward(X)

        # y_pred, y_true, weights = filter_samples_weights(Y_hat=y_pred, Y=y_true, weights=weights)
        loss = self.criterion.forward(y_pred, y_true, weights=weights)
        self.train_metrics.update_metrics(y_pred, y_true, weights=weights)

        if batch_nb % 100 == 0:
            logs = self.train_metrics.compute_metrics()
            self.log("loss", loss, logger=True, on_step=True)
        else:
            logs = {}

        if proximity_loss is not None:
            loss = loss + proximity_loss
            logs.update({"proximity_loss": proximity_loss})

        self.log_dict(logs, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_nb):
        X, y_true, weights = batch

        y_pred, proximity_loss, _ = self.forward(X, save_betas=False)

        # y_pred, y_true, weights = filter_samples_weights(Y_hat=y_pred, Y=y_true, weights=weights)
        val_loss = self.criterion.forward(y_pred, y_true, weights=weights)
        self.valid_metrics.update_metrics(y_pred, y_true, weights=weights)

        if proximity_loss is not None:
            val_loss = val_loss + proximity_loss

        self.log("val_loss", val_loss)

        return val_loss

    def test_step(self, batch, batch_nb):
        X, y_true, weights = batch
        y_pred, proximity_loss, _ = self.forward(X, save_betas=True)

        # y_pred, y_true, weights = filter_samples_weights(Y_hat=y_pred, Y=y_true, weights=weights)
        test_loss = self.criterion(y_pred, y_true, weights=weights)

        if batch_nb == 0:
            print_pred_class_counts(y_pred, y_true, multilabel=self.dataset.multilabel)

        self.test_metrics.update_metrics(y_pred, y_true, weights=weights)

        if proximity_loss is not None:
            test_loss = test_loss + proximity_loss

        self.log("test_loss", test_loss)

        return test_loss

    def predict_step(self, batch, batch_idx: int, dataloader_idx=None):
        X, y_true, weights = batch
        y_pred, proximity_loss = self.forward(X, save_betas=True)

        predict_loss = self.criterion(y_pred, y_true)
        self.test_metrics.update_metrics(y_pred, y_true)

        if self.hparams.use_proximity:
            predict_loss = predict_loss + proximity_loss

        self.log("predict_loss", predict_loss)

        return predict_loss

    def configure_optimizers(self):
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'alpha_activation', 'batchnorm', 'layernorm', "activation",
                    'LayerNorm.bias', 'LayerNorm.weight',
                    'BatchNorm.bias', 'BatchNorm.weight']

        optimizer_grouped_parameters = [
            {'params': [p for name, p in param_optimizer if not any(key in name for key in no_decay) \
                        and "embeddings" not in name],
             'weight_decay': self.hparams.weight_decay},
            {'params': [p for name, p in param_optimizer if any(key in name for key in no_decay)], 'weight_decay': 0.0},
            {'params': [p for name, p in param_optimizer if "embeddings" in name], 'lr': self.lr / 5,
             'weight_decay': 0.0},
        ]

        # print("weight_decay", sorted({name for name, p in param_optimizer if not any(key in name for key in no_decay)}))
        # print("no weight_decay", sorted({name for name, p in param_optimizer if any(key in name for key in no_decay)}))
        # print("embeddings", [name for name, p in param_optimizer if "embeddings" in name])
        optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=self.lr)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=self.num_training_steps,
                                                               eta_min=self.lr / 100
                                                               )

        return {"optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "val_loss"
                }


class MetaPath2Vec(Metapath2vec, pl.LightningModule):
    def __init__(self, hparams, dataset: HeteroNetDataset, metrics=None):
        # Hparams
        self.batch_size = hparams.batch_size
        self.sparse = hparams.sparse

        embedding_dim = hparams.embedding_dim
        walk_length = hparams.walk_length
        context_size = hparams.context_size
        walks_per_node = hparams.walks_per_node
        num_negative_samples = hparams.num_negative_samples

        # Dataset
        self.dataset = dataset

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

        num_nodes_dict = dataset.get_num_nodes_dict(dataset.edge_index_dict)

        super(MetaPath2Vec, self).__init__(edge_index_dict=edge_index_dict, embedding_dim=embedding_dim,
                                           metapath=metapaths, walk_length=walk_length, context_size=context_size,
                                           walks_per_node=walks_per_node,
                                           num_negative_samples=num_negative_samples, num_nodes_dict=num_nodes_dict,
                                           sparse=hparams.sparse)

        hparams.name = self.name()
        hparams.n_params = self.get_n_params()
        hparams.inductive = dataset.inductive
        self._set_hparams(hparams)

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
        self.log("loss", loss, logger=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_nb):
        pos_rw, neg_rw = batch
        val_loss = self.loss(pos_rw, neg_rw)
        self.log("val_loss", val_loss, on_step=True)
        return val_loss

    def test_step(self, batch, batch_nb):
        pos_rw, neg_rw = batch
        test_loss = self.loss(pos_rw, neg_rw)
        self.log("test_loss", test_loss, on_step=True)
        return test_loss

    def training_epoch_end(self, outputs):
        if self.current_epoch % 10 == 0:
            results = self.classification_results(training=True)
        else:
            results = {}

        logs = results
        self.log_dict(logs, prog_bar=True)
        return None

    def validation_epoch_end(self, outputs):
        logs = {}
        if self.current_epoch % 5 == 0:
            logs.update({"val_" + k: v for k, v in self.classification_results(training=False).items()})
        self.log_dict(logs, prog_bar=True)
        return None

    def test_epoch_end(self, outputs):
        logs = {}
        logs.update({"test_" + k: v for k, v in self.classification_results(training=False, testing=True).items()})
        self.log_dict(logs, prog_bar=True)
        return None

    def sample(self, src: torch_sparse.SparseTensor, num_neighbors: int,
               subset=None) -> torch.Tensor:

        rowptr, col, _ = src.csr()
        rowcount = src.storage.rowcount()

        if subset is not None:
            rowcount = rowcount[subset]
            rowptr = rowptr[subset]

        rand = torch.rand((rowcount.size(0), num_neighbors), device=col.device)
        rand.mul_(rowcount.to(rand.dtype).view(-1, 1))
        rand = rand.to(torch.long)
        rand.add_(rowptr.view(-1, 1))

        rand = torch.clamp(rand, min=0, max=min(col.size(0) - 1, rand.squeeze(-1).max()))

        return col[rand]

    def pos_sample(self, batch):
        batch = batch.repeat(self.walks_per_node)

        rws = [batch]
        for i in range(self.walk_length):
            keys = self.metapath[i % len(self.metapath)]
            adj = self.adj_dict[keys]
            batch = self.sample(adj, num_neighbors=1, subset=batch).squeeze()
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

    def collate_fn(self, batch):
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch)
        return self.pos_sample(batch), self.neg_sample(batch)

    def train_dataloader(self, ):
        loader = torch.utils.data.DataLoader(range(self.dataset.num_nodes_dict[self.dataset.head_node_type]),
                                             batch_size=self.hparams.batch_size,
                                             shuffle=True, num_workers=0,
                                             collate_fn=self.collate_fn,
                                             )
        return loader

    def val_dataloader(self, ):
        loader = torch.utils.data.DataLoader(self.dataset.validation_idx,
                                             batch_size=self.hparams.batch_size,
                                             shuffle=False, num_workers=0,
                                             collate_fn=self.collate_fn, )
        return loader

    def test_dataloader(self, ):
        loader = torch.utils.data.DataLoader(self.dataset.testing_idx,
                                             batch_size=self.hparams.batch_size,
                                             shuffle=False, num_workers=0,
                                             collate_fn=self.collate_fn, )
        return loader

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

    def configure_optimizers(self):
        if self.sparse:
            optimizer = torch.optim.SparseAdam(self.parameters(), lr=self.hparams.lr)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

        return {"optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "val_loss"}

