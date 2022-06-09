import logging
import math
import traceback
from typing import Dict, Iterable, Union, Tuple

import pandas as pd
import pytorch_lightning as pl
import torch
import torch_sparse.sample
import tqdm
from fairscale.nn import auto_wrap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.multiclass import OneVsRestClassifier
from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch_geometric.nn import MetaPath2Vec as Metapath2vec

from moge.dataset import HeteroNodeClfDataset
from moge.dataset.graph import HeteroGraphDataset
from moge.model.PyG.latte import LATTE
from moge.model.PyG.latte_flat import LATTE
from moge.model.classifier import DenseClassification, LabelGraphNodeClassifier
from moge.model.encoder import LSTMSequenceEncoder, HeteroSequenceEncoder, HeteroNodeEncoder
from moge.model.losses import ClassificationLoss
from moge.model.metrics import Metrics
from moge.model.trainer import NodeClfTrainer, print_pred_class_counts
from moge.model.utils import filter_samples_weights, process_tensor_dicts, activation, select_batch


class LATTENodeClf(NodeClfTrainer):
    def __init__(self, hparams, dataset: HeteroGraphDataset, metrics=["accuracy"],
                 collate_fn="neighbor_sampler") -> None:
        super(LATTENodeClf, self).__init__(hparams=hparams, dataset=dataset, metrics=metrics)
        self.head_node_type = dataset.head_node_type
        self.node_types = list(dataset.num_nodes_dict.keys())
        self.dataset = dataset
        self.multilabel = dataset.multilabel
        self.y_types = list(dataset.y_dict.keys())
        self._name = f"LATTE-{hparams.t_order}"
        self.collate_fn = collate_fn

        if "fanouts" in hparams and isinstance(hparams.fanouts,
                                               Iterable) and self.dataset.neighbor_sizes != hparams.fanouts:
            self.set_fanouts(self.dataset, hparams.fanouts)
            hparams.n_layers = len(dataset.neighbor_sizes)

        assert hparams.n_layers == len(dataset.neighbor_sizes)

        self.embedder = LATTE(n_layers=hparams.n_layers,
                              t_order=min(hparams.t_order, hparams.n_layers),
                              embedding_dim=hparams.embedding_dim,
                              num_nodes_dict=dataset.num_nodes_dict,
                              metapaths=dataset.get_metapaths(khop=True if "khop" in collate_fn else None),
                              activation=hparams.activation,
                              attn_heads=hparams.attn_heads,
                              attn_activation=hparams.attn_activation,
                              attn_dropout=hparams.attn_dropout,
                              layer_pooling=hparams.layer_pooling,
                              use_proximity=hparams.use_proximity \
                                  if hasattr(hparams, "use_proximity") else False,
                              neg_sampling_ratio=hparams.neg_sampling_ratio \
                                  if hasattr(hparams, "neg_sampling_ratio") else None,
                              edge_sampling=hparams.edge_sampling if hasattr(hparams, "edge_sampling") else False,
                              hparams=hparams)

        if "vocab" not in hparams or hparams.vocab is None:
            self.embeddings = self.initialize_embeddings(hparams.embedding_dim,
                                                         dataset.num_nodes_dict,
                                                         dataset.node_attr_shape,
                                                         pretrain_embeddings=hparams.node_emb_init if "node_emb_init" in hparams else None,
                                                         freeze=hparams.freeze_embeddings if "freeze_embeddings" in hparams else True)

            # node types that needs a projection to align to the embedding_dim
            self.proj_ntypes = [ntype for ntype in self.node_types \
                                if (ntype in dataset.node_attr_shape and
                                    dataset.node_attr_shape[ntype] != hparams.embedding_dim) \
                                or (self.embeddings and ntype in self.embeddings and
                                    self.embeddings[ntype].weight.size(1) != hparams.embedding_dim)]

            self.feature_projection = nn.ModuleDict({
                ntype: nn.Linear(
                    in_features=dataset.node_attr_shape[ntype] \
                        if not self.embeddings or ntype not in self.embeddings \
                        else self.embeddings[ntype].weight.size(1),
                    out_features=hparams.embedding_dim) \
                for ntype in self.proj_ntypes})

            if hparams.batchnorm_l:
                self.batchnorm = nn.ModuleDict({
                    ntype: nn.BatchNorm1d(hparams.embedding_dim) \
                    for ntype in self.proj_ntypes
                })

            self.dropout = hparams.dropout if hasattr(hparams, "dropout") else 0.0

        else:
            self.sequence_encoders = nn.ModuleDict({
                ntype: LSTMSequenceEncoder(vocab_size=len(vocab.vocab), embedding_dim=hparams.embedding_dim) \
                for ntype, vocab in hparams.vocab.items()})

        if hparams.nb_cls_dense_size >= 0:
            if hparams.layer_pooling == "concat":
                hparams.embedding_dim = hparams.embedding_dim * hparams.n_layers
                logging.info("embedding_dim {}".format(hparams.embedding_dim))
            elif hparams.layer_pooling == "order_concat":
                hparams.embedding_dim = hparams.embedding_dim * self.embedder.layers[-1].t_order

            if "cls_graph" in hparams and hparams.cls_graph:
                self.classifier = LabelGraphNodeClassifier(hparams)
            else:
                self.classifier = DenseClassification(hparams)
        else:
            assert "concat" not in hparams.layer_pooling, "Layer pooling cannot be `concat` or `rel_concat` when output of network is a GNN"

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

        if hasattr(self, "feature_projection"):
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

    def forward(self, inputs: Dict[str, Union[Tensor, Dict[str, Tensor]]], **kwargs):
        if not self.training:
            self._node_ids = inputs["global_node_index"]

        if "x_dict" in inputs or hasattr(self, "embeddings"):
            h_out = self.transform_inp_feats(inputs["x_dict"], global_node_idx=inputs["global_node_index"][0])

        elif "sequence" in inputs:
            h_out = {ntype: self.sequence_encoders[ntype](inputs["sequence"][ntype], inputs["seq_len"][ntype]) \
                     for ntype in inputs["sequence"]}

        embeddings = self.embedder(h_out,
                                   inputs["edge_index"],
                                   inputs["sizes"],
                                   inputs["global_node_index"], **kwargs)

        if isinstance(self.head_node_type, str):
            y_hat = self.classifier(embeddings[self.head_node_type]) \
                if hasattr(self, "classifier") else embeddings[self.head_node_type]

        elif isinstance(self.head_node_type, Iterable):
            if hasattr(self, "classifier"):
                y_hat = {ntype: self.classifier(emb) for ntype, emb in embeddings.items()}
            else:
                y_hat = embeddings

        return y_hat

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
        y_pred = self.forward(X)

        y_pred, y_true, weights = process_tensor_dicts(y_pred, y_true, weights)
        y_pred, y_true, weights = filter_samples_weights(Y_hat=y_pred, Y=y_true, weights=weights)
        if y_true.size(0) == 0: return torch.tensor(0.0, requires_grad=True)

        loss = self.criterion.forward(y_pred, y_true, weights=weights)
        self.train_metrics.update_metrics(y_pred, y_true, weights=weights)

        if batch_nb % 100 == 0:
            logs = self.train_metrics.compute_metrics()
            self.log("loss", loss, logger=True, on_step=True)
        else:
            logs = {}

        self.log_dict(logs, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_nb):
        X, y_true, weights = batch
        y_pred = self.forward(X, save_betas=False)

        y_pred, y_true, weights = process_tensor_dicts(y_pred, y_true, weights)
        y_pred, y_true, weights = filter_samples_weights(Y_hat=y_pred, Y=y_true, weights=weights)
        if y_true.size(0) == 0: return torch.tensor(0.0, requires_grad=True)

        val_loss = self.criterion.forward(y_pred, y_true, weights=weights)
        self.valid_metrics.update_metrics(y_pred, y_true, weights=weights)

        self.log("val_loss", val_loss)

        return val_loss

    def test_step(self, batch, batch_nb):
        X, y_true, weights = batch
        y_pred = self.forward(X, save_betas=True)

        y_pred, y_true, weights = process_tensor_dicts(y_pred, y_true, weights)
        y_pred, y_true, weights = filter_samples_weights(Y_hat=y_pred, Y=y_true, weights=weights)
        if y_true.size(0) == 0: return torch.tensor(0.0, requires_grad=True)

        test_loss = self.criterion(y_pred, y_true, weights=weights)

        if batch_nb == 0:
            print_pred_class_counts(y_pred, y_true, multilabel=self.dataset.multilabel)

        self.test_metrics.update_metrics(y_pred, y_true, weights=weights)

        self.log("test_loss", test_loss)

        return test_loss

    def predict_step(self, batch, batch_idx: int, dataloader_idx=None):
        X, y_true, weights = batch
        y_pred = self.forward(X, save_betas=True)

        predict_loss = self.criterion(y_pred, y_true)
        self.test_metrics.update_metrics(y_pred, y_true)

        self.log("predict_loss", predict_loss)

        return predict_loss

    def predict(self, dataloader, node_names=None, filter_nan_labels=True, **kwargs):
        y_true = []
        y_pred = []

        for X_test, y_test, w_test in tqdm.tqdm(dataloader):
            y_test_pred, _, edge_index = self.forward(X_test, save_betas=True)

            y_test_pred, y_test, w_test = process_tensor_dicts(y_test_pred, y_test, w_test)
            mask_idx = filter_samples_weights(Y_hat=y_test_pred, Y=y_test, weights=w_test, return_index=True)

            y_test = y_test[mask_idx]
            y_test_pred = activation(y_test_pred, loss_type=self.hparams["loss_type"])[mask_idx]
            w_test = w_test[mask_idx]

            node_ids = X_test["global_node_index"][-1][self.head_node_type].numpy()[mask_idx]
            if node_names is not None:
                node_ids = node_names[node_ids]

            y_test = pd.DataFrame(y_test.numpy(), index=node_ids, columns=self.dataset.classes)
            y_test_pred = pd.DataFrame(y_test_pred.detach().cpu().numpy(), index=node_ids, columns=self.dataset.classes)

            y_true.append(y_test)
            y_pred.append(y_test_pred)

        y_true = pd.concat(y_true, axis=0)
        y_pred = pd.concat(y_pred, axis=0)

        if filter_nan_labels:
            mask_labels = y_true.columns[y_true.sum(0) > 0]
            y_true = y_true.filter(mask_labels, axis=1)
            y_pred = y_pred.filter(mask_labels, axis=1)

        return y_true, y_pred

    def configure_optimizers(self):
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'alpha_activation', 'batchnorm', 'layernorm', "activation", "embeddings",
                    'LayerNorm.bias', 'LayerNorm.weight',
                    'BatchNorm.bias', 'BatchNorm.weight']

        optimizer_grouped_parameters = [
            {'params': [p for name, p in param_optimizer \
                        if not any(key in name for key in no_decay) \
                        and "embeddings" not in name],
             'weight_decay': self.hparams.weight_decay},
            {'params': [p for name, p in param_optimizer if any(key in name for key in no_decay)],
             'weight_decay': 0.0},
        ]

        # print("weight_decay", sorted({name for name, p in param_optimizer if not any(key in name for key in no_decay)}))
        # print("no weight_decay", sorted({name for name, p in param_optimizer if any(key in name for key in no_decay)}))

        # optimizer_grouped_parameters.append({'params': [p for name, p in param_optimizer if "embeddings" in name],
        #                                      # 'lr': self.lr / 2,
        #                                      'weight_decay': 0.0},)
        # print("embeddings", [name for name, p in param_optimizer if "embeddings" in name])

        optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=self.lr)

        extra = {}
        if "lr_annealing" in self.hparams and self.hparams.lr_annealing == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                   T_max=self.num_training_steps,
                                                                   eta_min=self.lr / 100
                                                                   )
            extra = {"lr_scheduler": scheduler, "monitor": "val_loss"}
            print("Using CosineAnnealingLR", scheduler.state_dict())

        elif "lr_annealing" in self.hparams and self.hparams.lr_annealing == "restart":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                             T_0=50, T_mult=1,
                                                                             eta_min=self.lr / 100)
            extra = {"lr_scheduler": scheduler, "monitor": "val_loss"}
            print("Using CosineAnnealingWarmRestarts", scheduler.state_dict())

        return {"optimizer": optimizer, **extra}



class MetaPath2Vec(Metapath2vec, pl.LightningModule):
    def __init__(self, hparams, dataset: HeteroGraphDataset, metrics=None):
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


class LATTEFlatNodeClf(NodeClfTrainer):
    def __init__(self, hparams, dataset: HeteroNodeClfDataset, metrics=["accuracy"], collate_fn=None) -> None:
        super().__init__(hparams=hparams, dataset=dataset, metrics=metrics)
        self.head_node_type = dataset.head_node_type
        self.node_types = dataset.node_types
        self.dataset = dataset
        self.multilabel = dataset.multilabel
        self.y_types = list(dataset.y_dict.keys())
        self._name = f"LATTE-{hparams.n_layers}-{hparams.t_order}"
        self.collate_fn = collate_fn

        # Node attr input
        if hasattr(dataset, 'seq_tokenizer'):
            self.seq_encoder = HeteroSequenceEncoder(hparams, dataset)

        if not hasattr(self, "seq_encoder") or len(self.seq_encoder.seq_encoders.keys()) < len(self.node_types):
            self.encoder = HeteroNodeEncoder(hparams, dataset)

        # Graph embedding
        self.embedder = LATTE(n_layers=hparams.n_layers,
                              t_order=min(hparams.t_order, hparams.n_layers),
                              embedding_dim=hparams.embedding_dim,
                              num_nodes_dict=dataset.num_nodes_dict,
                              metapaths=dataset.get_metapaths(khop=None),
                              layer_pooling=hparams.layer_pooling,
                              activation=hparams.activation,
                              attn_heads=hparams.attn_heads,
                              attn_activation=hparams.attn_activation,
                              attn_dropout=hparams.attn_dropout,
                              use_proximity=hparams.use_proximity if hasattr(hparams, "use_proximity") else False,
                              neg_sampling_ratio=hparams.neg_sampling_ratio \
                                  if hasattr(hparams, "neg_sampling_ratio") else None,
                              edge_sampling=hparams.edge_sampling if hasattr(hparams, "edge_sampling") else False,
                              hparams=hparams)

        # Output layer
        if "cls_graph" in hparams and hparams.cls_graph is not None:
            self.classifier = LabelGraphNodeClassifier(hparams)

        elif hparams.nb_cls_dense_size >= 0:
            if hparams.layer_pooling == "concat":
                hparams.embedding_dim = hparams.embedding_dim * hparams.t_order
                logging.info("embedding_dim {}".format(hparams.embedding_dim))

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

        self.val_moving_loss = torch.tensor([3.0, ] * 5, dtype=torch.float)

    def configure_sharded_model(self):
        # modules are sharded across processes
        # as soon as they are wrapped with ``wrap`` or ``auto_wrap``.
        # During the forward/backward passes, weights get synced across processes
        # and de-allocated once computation is complete, saving memory.

        # Wraps the layer in a Fully Sharded Wrapper automatically
        if hasattr(self, "seq_encoder"):
            self.seq_encoder = auto_wrap(self.seq_encoder)
        if hasattr(self, "encoder"):
            self.encoder = auto_wrap(self.encoder)

    def forward(self, inputs: Dict[str, Union[Tensor, Dict[Union[str, Tuple[str]], Union[Tensor, int]]]], **kwargs):
        if not self.training:
            self._node_ids = inputs["global_node_index"]

        h_out = {}
        if 'sequences' in inputs and hasattr(self, "seq_encoder"):
            h_out.update(self.seq_encoder.forward(inputs['sequences'],
                                                  minibatch=math.sqrt(self.hparams.batch_size // 4)))

        if len(h_out) < len(inputs["global_node_index"].keys()):
            embs = self.encoder.forward(inputs["x_dict"], global_node_idx=inputs["global_node_index"])
            h_out.update({ntype: emb for ntype, emb in embs.items() if ntype not in h_out})

        embeddings = self.embedder.forward(h_dict=h_out,
                                           edge_index_dict=inputs["edge_index_dict"],
                                           global_node_idx=inputs["global_node_index"],
                                           sizes=inputs["sizes"],
                                           **kwargs)

        if hasattr(self, "classifier"):
            y_hat = self.classifier.forward(embeddings[self.head_node_type])
        else:
            y_hat = embeddings[self.head_node_type]

        return y_hat

    def training_step(self, batch, batch_nb):
        X, y_true, weights = batch
        y_pred = self.forward(X)

        y_pred, y_true, weights = process_tensor_dicts(y_pred, y_true, weights)
        y_pred, y_true, weights = filter_samples_weights(Y_hat=y_pred, Y=y_true, weights=weights)
        if y_true.size(0) == 0: return torch.tensor(0.0, requires_grad=False)

        loss = self.criterion.forward(y_pred, y_true, weights=weights)

        self.update_node_clf_metrics(self.train_metrics, y_pred, y_true, weights)

        return loss

    def validation_step(self, batch, batch_nb):
        X, y_true, weights = batch
        y_pred = self.forward(X)

        y_pred, y_true, weights = select_batch(X['batch_size'], y_pred, y_true, weights)
        y_pred, y_true, weights = filter_samples_weights(Y_hat=y_pred, Y=y_true, weights=weights)
        if y_true.size(0) == 0:
            return torch.tensor(0.0, requires_grad=False)

        val_loss = self.criterion.forward(y_pred, y_true, weights=weights)
        self.update_node_clf_metrics(self.valid_metrics, y_pred, y_true, weights)

        self.log("val_loss", val_loss)

        return val_loss

    def update_node_clf_metrics(self, metrics: Union[Metrics, Dict[str, Metrics]],
                                y_pred: Tensor, y_true: Tensor, weights: Tensor):
        if isinstance(metrics, dict):
            y_pred_dict = self.dataset.split_labels_by_go_namespace(y_pred)
            y_true_dict = self.dataset.split_labels_by_go_namespace(y_true)

            for namespace in y_true_dict.keys():
                go_type = "BPO" if namespace == 'biological_process' else \
                    "CCO" if namespace == 'cellular_component' else \
                        "MFO" if namespace == 'molecular_function' else None

                metrics[go_type].update_metrics(y_pred_dict[namespace], y_true_dict[namespace], weights=weights)

        else:
            metrics.update_metrics(y_pred, y_true, weights=weights)

    def on_validation_end(self) -> None:
        super().on_validation_end()
        if self.current_epoch % 20 == 1:
            self.plot_sankey_flow(layer=-1)

    def test_step(self, batch, batch_nb):
        X, y_true, weights = batch
        y_pred = self.forward(X, save_betas=False)

        y_pred, y_true, weights = select_batch(X['batch_size'], y_pred, y_true, weights)
        y_pred, y_true, weights = filter_samples_weights(Y_hat=y_pred, Y=y_true, weights=weights)
        if y_true.size(0) == 0: return torch.tensor(0.0, requires_grad=False)

        test_loss = self.criterion(y_pred, y_true, weights=weights)

        if batch_nb == 0:
            print_pred_class_counts(y_pred, y_true, multilabel=self.dataset.multilabel)

        self.update_node_clf_metrics(self.test_metrics, y_pred, y_true, weights)

        self.log("test_loss", test_loss)

        return test_loss

    def on_test_end(self):
        try:
            if self.wandb_experiment is not None:
                X, y, _ = self.dataset.get_full_graph()
                embs, edge_pred_dict = self.cpu().forward(X, y, save_betas=True)

                self.predict_umap(X, embs, log_table=True)
                self.plot_sankey_flow(layer=-1)
                self.cleanup_artifacts()

        except Exception as e:
            traceback.print_exc()

        finally:
            super().on_test_end()

    def configure_optimizers(self):
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'alpha_activation', 'batchnorm', 'layernorm', "activation", "embeddings",
                    'LayerNorm.bias', 'LayerNorm.weight',
                    'BatchNorm.bias', 'BatchNorm.weight']

        optimizer_grouped_parameters = [
            {'params': [p for name, p in param_optimizer \
                        if not any(key in name for key in no_decay) \
                        and "embeddings" not in name],
             'weight_decay': self.hparams.weight_decay if isinstance(self.hparams.weight_decay, float) else 0.0},
            {'params': [p for name, p in param_optimizer if any(key in name for key in no_decay)],
             'weight_decay': 0.0},
        ]

        optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=self.lr)

        extra = {}
        if "lr_annealing" in self.hparams and self.hparams.lr_annealing == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                   T_max=self.num_training_steps,
                                                                   eta_min=self.lr / 100
                                                                   )
            extra = {"lr_scheduler": scheduler, "monitor": "val_loss"}
            print("Using CosineAnnealingLR", scheduler.state_dict())

        elif "lr_annealing" in self.hparams and self.hparams.lr_annealing == "restart":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                             T_0=50, T_mult=1,
                                                                             eta_min=self.lr / 100)
            extra = {"lr_scheduler": scheduler, "monitor": "val_loss"}
            print("Using CosineAnnealingWarmRestarts", scheduler.state_dict())

        return {"optimizer": optimizer, **extra}

    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if self.trainer.max_steps:
            return self.trainer.max_steps

        limit_batches = self.trainer.limit_train_batches
        batches = len(self.train_dataloader())
        batches = min(batches, limit_batches) if isinstance(limit_batches, int) else int(limit_batches * batches)

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_accum = self.trainer.accumulate_grad_batches * num_devices
        return (batches // effective_accum) * self.trainer.max_epochs
