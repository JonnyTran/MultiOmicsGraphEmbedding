import logging
from typing import List, Dict, Tuple, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn as nn, Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch_sparse.tensor import SparseTensor

from moge.dataset.graph import HeteroGraphDataset
from moge.model.classifier import DenseClassification
from moge.model.losses import ClassificationLoss
from moge.model.sampling import negative_sample
from moge.model.trainer import NodeClfTrainer, print_pred_class_counts
from moge.model.transformers.encoder import SequenceEncoder
from moge.model.utils import tensor_sizes, filter_samples_weights


class LATTEFlatNodeClf(NodeClfTrainer):
    def __init__(self, hparams, dataset: HeteroGraphDataset, metrics=["accuracy"],
                 collate_fn="neighbor_sampler") -> None:
        super().__init__(hparams=hparams, dataset=dataset, metrics=metrics)
        self.head_node_type = dataset.head_node_type
        self.node_types = dataset.node_types
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

        # Node feature projection
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

            if hparams.batchnorm:
                self.batchnorm = nn.ModuleDict({
                    ntype: nn.BatchNorm1d(hparams.embedding_dim) \
                    for ntype in self.proj_ntypes
                })

            self.dropout = hparams.dropout if hasattr(hparams, "dropout") else 0.0

        else:
            self.sequence_encoders = nn.ModuleDict({
                ntype: SequenceEncoder(vocab_size=len(vocab.vocab), embed_dim=hparams.embedding_dim) \
                for ntype, vocab in hparams.vocab.items()})

        # Last layer
        if hparams.nb_cls_dense_size >= 0:
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
            h_out = self.transform_inp_feats(inputs["x_dict"], global_node_idx=inputs["global_node_index"])

        elif "sequence" in inputs:
            h_out = {ntype: self.sequence_encoders[ntype](inputs["sequence"][ntype], inputs["seq_len"][ntype]) \
                     for ntype in inputs["sequence"]}

        embeddings, proximity_loss, edge_index_dict = self.embedder(h_out,
                                                                    inputs["edge_index_dict"],
                                                                    inputs["global_node_index"], **kwargs)

        y_hat = self.classifier(embeddings[self.head_node_type]) \
            if hasattr(self, "classifier") else embeddings[self.head_node_type]

        return y_hat, proximity_loss, edge_index_dict

    def training_step(self, batch, batch_nb):
        X, y_true, weights = batch
        y_pred, proximity_loss = self.forward(X)

        y_pred, y_true, weights = filter_samples_weights(Y_hat=y_pred, Y=y_true, weights=weights)
        loss = self.criterion.forward(y_pred, y_true, weights=weights)

        self.train_metrics.update_metrics(y_pred, y_true, weights=weights)

        if batch_nb % 100 == 0:
            logs = self.train_metrics.compute_metrics()
            self.log("loss", loss, logger=True, on_step=True)
        else:
            logs = {}

        if self.hparams.use_proximity:
            loss = loss + proximity_loss
            logs.update({"proximity_loss": proximity_loss})

        self.log_dict(logs, prog_bar=True, logger=True, on_step=True)

        return loss

    def validation_step(self, batch, batch_nb):
        X, y_true, weights = batch
        print(tensor_sizes(X))
        y_pred, proximity_loss = self.forward(X)

        y_pred, y_true, weights = filter_samples_weights(Y_hat=y_pred, Y=y_true, weights=weights)
        val_loss = self.criterion.forward(y_pred, y_true, weights=weights)
        self.valid_metrics.update_metrics(y_pred, y_true)

        if self.hparams.use_proximity:
            val_loss = val_loss + proximity_loss

        self.log("val_loss", val_loss)

        return val_loss

    def test_step(self, batch, batch_nb):
        X, y_true, weights = batch
        y_pred, proximity_loss = self.forward(X, save_betas=True)

        y_pred, y_true, weights = filter_samples_weights(Y_hat=y_pred, Y=y_true, weights=weights)
        test_loss = self.criterion(y_pred, y_true, weights=weights)

        if batch_nb == 0:
            print_pred_class_counts(y_pred, y_true, multilabel=self.dataset.multilabel)

        self.test_metrics.update_metrics(y_pred, y_true, weights=weights)

        if self.hparams.use_proximity:
            test_loss = test_loss + proximity_loss

        self.log("test_loss", test_loss)

        return test_loss

    def configure_optimizers(self):
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'alpha_activation', 'embedding', 'layernorm']
        optimizer_grouped_parameters = [
            {'params': [p for name, p in param_optimizer if not any(key in name for key in no_decay)],
             'weight_decay': self.hparams.weight_decay},
            {'params': [p for name, p in param_optimizer if any(key in name for key in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=self.lr)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_training_steps,
                                                               eta_min=self.lr / 100)

        return {"optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "val_loss"}

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


class LATTE(nn.Module):
    def __init__(self, n_layers: int, t_order: int, embedding_dim: int, num_nodes_dict: Dict[str, int],
                 metapaths: List[Tuple[str, str, str]], layer_pooling,
                 activation: str = "relu", attn_heads: int = 1, attn_activation="sharpening", attn_dropout: float = 0.5,
                 use_proximity=True, neg_sampling_ratio=2.0, edge_sampling=True,
                 hparams=None):
        super().__init__()
        self.metapaths = metapaths
        self.node_types = list(num_nodes_dict.keys())
        self.embedding_dim = embedding_dim

        self.t_order = t_order
        self.n_layers = n_layers

        self.neg_sampling_ratio = neg_sampling_ratio
        self.edge_sampling = edge_sampling
        self.use_proximity = use_proximity
        self.layer_pooling = layer_pooling

        # align the dimension of different types of nodes
        if hparams.batchnorm:
            self.batchnorm = nn.ModuleDict({
                ntype: nn.BatchNorm1d(embedding_dim) for ntype in num_nodes_dict
            })
        self.dropout = hparams.dropout if hasattr(hparams, "dropout") else 0.0

        layers = []
        for l in range(n_layers):
            is_last_layer = l + 1 == n_layers
            is_output_layer = hparams.nb_cls_dense_size < 0
            print(l, metapaths)

            layer = LATTEConv(input_dim=embedding_dim,
                              output_dim=hparams.n_classes if is_last_layer and is_output_layer else embedding_dim,
                              num_nodes_dict=num_nodes_dict, metapaths=metapaths, layer=l,
                              activation=None if is_last_layer and is_output_layer else activation,
                              layernorm=False if not hasattr(hparams, "layernorm") or (
                                      is_last_layer and is_output_layer) else hparams.layernorm,
                              attn_heads=attn_heads, attn_activation=attn_activation, attn_dropout=attn_dropout,
                              use_proximity=use_proximity, neg_sampling_ratio=neg_sampling_ratio, )
            layers.append(layer)

        self.layers: List[LATTEConv] = nn.ModuleList(layers)

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')

    def forward(self, h_dict: Dict[str, Tensor],
                edge_index_dict: Dict[Tuple, Tensor],
                global_node_idx: Dict[str, Tensor],
                save_betas=False):
        """
        Args:
            h_dict: Dict of <ntype>:<tensor size (batch_size, in_channels)>. If nodes are not attributed, then pass an empty dict.
            global_node_idx: Dict of <ntype>:<int tensor size (batch_size,)>
            edge_index_dict: Dict of <metapath>:<tensor size (2, num_edge_index)>
            save_betas: whether to save _beta values for batch
        Returns:
            embedding_output, proximity_loss, edge_pred_dict
        """
        proximity_loss = torch.tensor(0.0, device=self.device) if self.use_proximity else None

        h_layers = {ntype: [] for ntype in global_node_idx}
        for t in range(self.t_order):
            if t == 0:
                h_dict, t_loss, edge_pred_dict = self.layers[t].forward(x_l=h_dict,
                                                                        edge_index_dict=edge_index_dict,
                                                                        global_node_idx=global_node_idx,
                                                                        save_betas=save_betas)
            else:
                h_dict, t_loss, _ = self.layers[t].forward(x_l=h_dict,
                                                           edge_index_dict=edge_index_dict,
                                                           global_node_idx=global_node_idx,
                                                           save_betas=save_betas)

            for ntype in global_node_idx:
                h_layers[ntype].append(h_dict[ntype])

            if self.use_proximity:
                proximity_loss += t_loss

        if self.layer_pooling in ["last", "order_concat"] or self.n_layers == 1:
            out = h_dict

        elif self.layer_pooling == "max":
            out = {node_type: torch.stack(h_list, dim=1) for node_type, h_list in h_layers.items() \
                   if len(h_list)}
            out = {ntype: h_s.max(1).values for ntype, h_s in out.items()}

        elif self.layer_pooling == "mean":
            out = {node_type: torch.stack(h_list, dim=1) for node_type, h_list in h_layers.items() \
                   if len(h_list)}
            out = {ntype: torch.mean(h_s, dim=1) for ntype, h_s in out.items()}

        elif self.layer_pooling == "concat":
            out = {node_type: torch.cat(h_list, dim=1) for node_type, h_list in h_layers.items() \
                   if len(h_list)}
        else:
            raise Exception("`layer_pooling` should be either ['last', 'max', 'mean', 'concat']")

        return out, proximity_loss, edge_pred_dict

    @staticmethod
    def join_metapaths(metapath_A, metapath_B):
        metapaths = []
        for relation_a in metapath_A:
            for relation_b in metapath_B:
                if relation_a[-1] == relation_b[0]:
                    new_relation = relation_a + relation_b[1:]
                    metapaths.append(new_relation)
        return metapaths

    @staticmethod
    def get_edge_index_values(edge_index_tup: [tuple, torch.Tensor]):
        if isinstance(edge_index_tup, tuple):
            edge_index = edge_index_tup[0]
            edge_values = edge_index[1]

        elif isinstance(edge_index_tup, torch.Tensor) and edge_index_tup.size(1) > 0:
            edge_index = edge_index_tup
            edge_values = torch.ones(edge_index_tup.size(1), dtype=torch.float64, device=edge_index_tup.device)
        else:
            return None, None

        if edge_values.dtype != torch.float:
            edge_values = edge_values.to(torch.float)

        return edge_index, edge_values

    @staticmethod
    def join_edge_indexes(edge_index_dict_A, edge_index_dict_B, global_node_idx, edge_sampling=False):
        output_edge_index = {}
        for metapath_a, edge_index_a in edge_index_dict_A.items():
            if is_negative(metapath_a): continue
            edge_index_a, values_a = LATTE.get_edge_index_values(edge_index_a)
            if edge_index_a is None: continue

            for metapath_b, edge_index_b in edge_index_dict_B.items():
                if metapath_a[-1] != metapath_b[0] or is_negative(metapath_b): continue

                new_metapath = metapath_a + metapath_b[1:]
                edge_index_b, values_b = LATTE.get_edge_index_values(edge_index_b)
                if edge_index_b is None: continue

                try:
                    new_edge_index, new_values = adamic_adar(indexA=edge_index_a, valueA=values_a,
                                                             indexB=edge_index_b, valueB=values_b,
                                                             m=global_node_idx[metapath_a[0]].size(0),
                                                             k=global_node_idx[metapath_b[0]].size(0),
                                                             n=global_node_idx[metapath_b[-1]].size(0),
                                                             coalesced=True,
                                                             sampling=edge_sampling
                                                             )
                    if new_edge_index.size(1) == 0: continue
                    output_edge_index[new_metapath] = (new_edge_index, new_values)

                except Exception as e:
                    print(f"{e} \n {metapath_a}: {edge_index_a.size(1)}, {metapath_b}: {edge_index_b.size(1)}")
                    print("\t", {"m": global_node_idx[metapath_a[0]].size(0),
                                 "k": global_node_idx[metapath_a[-1]].size(0),
                                 "n": global_node_idx[metapath_b[-1]].size(0), })
                    continue

        return output_edge_index

    def get_attn_activation_weights(self, t):
        return dict(zip(self.layers[t].metapaths, self.layers[t].alpha_activation.detach().numpy().tolist()))

    def get_relation_weights(self, t):
        return self.layers[t].get_relation_weights()


class LATTEConv(MessagePassing, pl.LightningModule):
    def __init__(self, input_dim: int, output_dim: int, num_nodes_dict: Dict[str, int], metapaths: List, layer: int,
                 activation: str = "relu", attn_heads=4, attn_activation="LeakyReLU", attn_dropout=0.2,
                 layernorm=False,
                 use_proximity=False, neg_sampling_ratio=1.0) -> None:
        super().__init__(aggr="add", flow="source_to_target", node_dim=0)
        self.layer = layer
        self.node_types = list(num_nodes_dict.keys())
        self.metapaths = list(metapaths)
        self.num_nodes_dict = num_nodes_dict
        self.embedding_dim = output_dim
        self.use_proximity = use_proximity
        self.neg_sampling_ratio = neg_sampling_ratio
        self.attn_heads = attn_heads
        self.attn_dropout = attn_dropout

        if activation == "sigmoid":
            self.activation = F.sigmoid
        elif activation == "tanh":
            self.activation = F.tanh
        elif activation == "relu":
            self.activation = F.relu
        else:
            print(f"Embedding activation arg `{activation}` did not match, so uses linear activation.")

        if layernorm:
            self.layernorm = torch.nn.ModuleDict({
                node_type: nn.LayerNorm(output_dim) \
                for node_type in self.node_types})

        self.linear = nn.ModuleDict(
            {node_type: nn.Linear(input_dim, output_dim, bias=True) \
             for node_type in self.node_types})  # W.shape (F x F)

        self.out_channels = self.embedding_dim // attn_heads
        self.attn_l = nn.Parameter(torch.Tensor(len(self.metapaths), attn_heads, self.out_channels))
        self.attn_r = nn.Parameter(torch.Tensor(len(self.metapaths), attn_heads, self.out_channels))

        self.rel_attn_l = nn.ParameterDict({
            ntype: nn.Parameter(torch.Tensor(attn_heads, self.out_channels)) \
            for ntype in self.node_types})
        self.rel_attn_r = nn.ParameterDict({
            ntype: nn.Parameter(torch.Tensor(attn_heads, self.out_channels)) \
            for ntype in self.node_types})

        if attn_activation == "sharpening":
            self.alpha_activation = nn.Parameter(torch.Tensor(len(self.metapaths)).fill_(1.0))
        elif attn_activation == "PReLU":
            self.alpha_activation = nn.PReLU(init=0.2)
        elif attn_activation == "LeakyReLU":
            self.alpha_activation = nn.LeakyReLU(negative_slope=0.2)
        else:
            print(f"WARNING: alpha_activation `{attn_activation}` did not match, so used linear activation")
            self.alpha_activation = None

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu', 0.2)
        for i, metapath in enumerate(self.metapaths):
            nn.init.xavier_normal_(self.attn_l[i], gain=gain)
            nn.init.xavier_normal_(self.attn_r[i], gain=gain)

        gain = nn.init.calculate_gain('relu')
        for node_type in self.linear:
            nn.init.xavier_normal_(self.linear[node_type].weight, gain=gain)

        gain = nn.init.calculate_gain('leaky_relu', 0.2)
        for ntype, rel_attn in self.rel_attn_l.items():
            nn.init.xavier_normal_(rel_attn, gain=gain)
        for ntype, rel_attn in self.rel_attn_r.items():
            nn.init.xavier_normal_(rel_attn, gain=gain)

    def get_h_dict(self, x_dict, global_node_idx):
        h_dict = {}
        for ntype in global_node_idx:
            if global_node_idx[ntype].numel() == 0: continue
            h_dict[ntype] = self.linear[ntype].forward(x_dict[ntype])

            h_dict[ntype] = h_dict[ntype].view(-1, self.attn_heads, self.out_channels)

        return h_dict

    def get_alphas(self, edge_index_dict, h_dict):
        alpha_l, alpha_r = {}, {}

        for i, metapath in enumerate(self.metapaths):
            if metapath not in edge_index_dict or edge_index_dict[metapath] is None or \
                    edge_index_dict[metapath].numel() == 0:
                continue
            head, tail = metapath[0], metapath[-1]

            alpha_l[metapath] = (h_dict[head] * self.attn_l[i]).sum(dim=-1)
            alpha_r[metapath] = (h_dict[tail] * self.attn_r[i]).sum(dim=-1)
        return alpha_l, alpha_r

    def get_beta_weights(self, node_emb, rel_embs, ntype):
        alpha_l = (node_emb * self.rel_attn_l[ntype]).sum(dim=-1)
        alpha_r = (rel_embs * self.rel_attn_r[ntype][:, None, :]).sum(dim=-1)

        beta = alpha_l[:, :, None] + alpha_r
        beta = F.leaky_relu(beta, negative_slope=0.2)
        beta = F.softmax(beta, dim=2)
        beta = F.dropout(beta, p=self.attn_dropout, training=self.training)
        return beta

    def forward(self, x_l, edge_index_dict, global_node_idx, save_betas=False):
        """
        Args:
            x_l: a dict of node attributes indexed ntype
            global_node_idx: A dict of index values indexed by ntype in this mini-batch sampling
            edge_index_dict: Sparse adjacency matrices for each metapath relation. A dict of edge_index indexed by metapath
            x_r: Context embedding of the previous order, required for t >= 2. Default: None (if first order). A dict of (ntype: tensor)
        Returns:
             output_emb, loss
        """
        print(self.layer, tensor_sizes(x_l))

        h_dict = self.get_h_dict(x_l, global_node_idx)
        # Compute node-level attention coefficients
        alpha_l, alpha_r = self.get_alphas(edge_index_dict, h_dict)

        # For each metapath in a node_type, use GAT message passing to aggregate h_j neighbors
        out = {}
        beta = {}
        for ntype in global_node_idx:
            if global_node_idx[ntype].size(0) == 0: continue
            print(">", ntype)
            out[ntype] = self.agg_relation_neighbors(ntype=ntype, alpha_l=alpha_l, alpha_r=alpha_r,
                                                     h_dict=h_dict, edge_index_dict=edge_index_dict,
                                                     global_node_idx=global_node_idx)
            out[ntype][:, :, -1, :] = h_dict[ntype]
            print("\n Layer", self.layer, ntype, tensor_sizes(out))

            beta[ntype] = self.get_beta_weights(h_dict[ntype], out[ntype], ntype=ntype)

            # Soft-select the relation-specific embeddings by a weighted average with beta[node_type]
            out[ntype] = out[ntype] * beta[ntype].unsqueeze(-1)
            out[ntype] = out[ntype].sum(2).view(out[ntype].size(0), self.embedding_dim)

            if hasattr(self, "layernorm"):
                out[ntype] = self.layernorm[ntype](out[ntype])

            if hasattr(self, "activation"):
                out[ntype] = self.activation(out[ntype])

            beta[ntype] = beta[ntype].mean(1)

        if not self.training: self.save_relation_weights(beta, global_node_idx)

        proximity_loss, edge_pred_dict = None, None
        if self.use_proximity:
            proximity_loss, edge_pred_dict = self.proximity_loss(edge_index_dict,
                                                                 alpha_l=alpha_l, alpha_r=alpha_r,
                                                                 global_node_idx=global_node_idx)
        return out, proximity_loss, edge_pred_dict

    def agg_relation_neighbors(self, ntype, alpha_l, alpha_r, h_dict, edge_index_dict, global_node_idx):
        # Initialize embeddings, size: (num_nodes, num_relations, embedding_dim)
        emb_relations = torch.zeros(
            size=(global_node_idx[ntype].size(0),
                  self.attn_heads,
                  self.num_tail_relations(ntype),
                  self.out_channels)).type_as(self.attn_l)

        for i, metapath in enumerate(self.get_tail_relations(ntype)):
            if metapath not in edge_index_dict or edge_index_dict[metapath] == None: continue
            head, tail = metapath[0], metapath[-1]
            num_node_head, num_node_tail = global_node_idx[head].size(0), global_node_idx[tail].size(0)

            edge_index, values = LATTE.get_edge_index_values(edge_index_dict[metapath])
            if edge_index is None or edge_index.size(1) == 0: continue

            print("\n", metapath)
            print("x_dict", tensor_sizes({tail: h_dict[tail], head: h_dict[head]}))
            print("global_node_idx", tensor_sizes(global_node_idx))
            print("edge_index", tensor_sizes(edge_index), edge_index.max(1).values)
            print("alpha", tensor_sizes({tail: alpha_r[metapath], head: alpha_l[metapath]}))
            print({"num_node_tail": num_node_tail, "num_node_head": num_node_head})

            # Propapate flows from target nodes to source nodes
            out = self.propagate(
                edge_index=edge_index,
                x=(h_dict[head], h_dict[tail]),
                alpha=(alpha_l[metapath], alpha_r[metapath]),
                size=(num_node_head, num_node_tail),
                metapath_idx=self.metapaths.index(metapath))

            emb_relations[:, :, i, :] = out
            print(ntype, out.shape, emb_relations.shape)
        return emb_relations

    def message(self, x_j, alpha_j, alpha_i, index, ptr, size_i, metapath_idx):
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        alpha = self.attn_activation(alpha, metapath_idx)
        alpha = softmax(alpha, index=index, ptr=ptr, num_nodes=size_i)
        alpha = F.dropout(alpha, p=self.attn_dropout, training=self.training)

        return x_j * alpha.unsqueeze(-1)

    def predict_scores(self, edge_index, alpha_l, alpha_r, metapath, logits=False):
        assert metapath in self.metapaths, f"If metapath `{metapath}` is tag_negative()'ed, then pass it with untag_negative()"

        # e_pred = self.attn_q[self.metapaths.index(metapath)].forward(
        #     torch.cat([alpha_l[metapath][edge_index[0]], alpha_r[metapath][edge_index[1]]], dim=1)).squeeze(-1)

        e_pred = self.attn_activation(alpha_l[metapath][edge_index[0]] + alpha_r[metapath][edge_index[1]],
                                      metapath_id=self.metapaths.index(metapath)).squeeze(-1)
        if logits:
            return e_pred
        else:
            return F.sigmoid(e_pred)

    def proximity_loss(self, edge_index_dict, alpha_l, alpha_r, global_node_idx):
        """
        For each relation/metapath type given in `edge_index_dict`, this function both predict link scores and computes
        the NCE loss for both positive and negative (sampled) links. For each relation type in `edge_index_dict`, if the
        negative metapath is not included, then the function automatically samples for random negative edges. And, if it
        is included, then computes the NCE loss over the given negative edges. This function returns the scores of the
        predicted positive and negative edges.

        :param edge_index_dict (dict): Dict of <relation/metapath>: <Tensor(2, num_edges)>
        :param alpha_l (dict): Dict of <ntype>:<alpha_l tensor>
        :param alpha_r (dict): Dict of <ntype>:<alpha_r tensor>
        :param global_node_idx (dict): Dict of <ntype>:<Tensor(node_idx,)>
        :return loss, edge_pred_dict: NCE loss. edge_pred_dict will contain both positive relations of shape (num_edges,) and negative relations of shape (num_edges*num_neg_edges, )
        """
        loss = torch.tensor(0.0, dtype=torch.float, device=self.conv[self.node_types[0]].weight.device)
        edge_pred_dict = {}
        for metapath, edge_index in edge_index_dict.items():
            # KL Divergence over observed positive edges or negative edges (if included)
            if isinstance(edge_index, tuple):  # Weighted edges
                edge_index, values = edge_index
            else:
                values = 1.0
            if edge_index is None: continue

            if not is_negative(metapath):
                e_pred_logits = self.predict_scores(edge_index, alpha_l, alpha_r, metapath, logits=True)
                loss += -torch.mean(values * F.logsigmoid(e_pred_logits), dim=-1)
            elif is_negative(metapath):
                e_pred_logits = self.predict_scores(edge_index, alpha_l, alpha_r, untag_negative(metapath), logits=True)
                loss += -torch.mean(F.logsigmoid(-e_pred_logits), dim=-1)

            edge_pred_dict[metapath] = F.sigmoid(e_pred_logits.detach())

            # Only need to sample for negative edges if negative metapath is not included
            if not is_negative(metapath) and tag_negative(metapath) not in edge_index_dict:
                neg_edge_index = negative_sample(edge_index,
                                                 M=global_node_idx[metapath[0]].size(0),
                                                 N=global_node_idx[metapath[-1]].size(0),
                                                 n_sample_per_edge=self.neg_sampling_ratio)
                if neg_edge_index is None or neg_edge_index.size(1) <= 1: continue

                e_neg_logits = self.predict_scores(neg_edge_index, alpha_l, alpha_r, metapath, logits=True)
                loss += -torch.mean(F.logsigmoid(-e_neg_logits), dim=-1)
                edge_pred_dict[tag_negative(metapath)] = F.sigmoid(e_neg_logits.detach())

        loss = torch.true_divide(loss, max(len(edge_index_dict) * 2, 1))
        return loss, edge_pred_dict

    def attn_activation(self, alpha, metapath_id):
        if isinstance(self.alpha_activation, torch.Tensor):
            return self.alpha_activation[metapath_id] * alpha
        elif isinstance(self.alpha_activation, nn.Module):
            return self.alpha_activation(alpha)
        else:
            return alpha

    def get_head_relations(self, head_node_type, to_str=False) -> list:
        relations = [".".join(metapath) if to_str and isinstance(metapath, tuple) else metapath \
                     for metapath in self.metapaths if metapath[0] == head_node_type]
        return relations

    def get_tail_relations(self, head_node_type, to_str=False) -> list:
        relations = [".".join(metapath) if to_str and isinstance(metapath, tuple) else metapath \
                     for metapath in self.metapaths if metapath[-1] == head_node_type]
        return relations

    def num_head_relations(self, node_type) -> int:
        """
        Return the number of metapaths with head node type equals to :param ntype: and plus one for none-selection.
        """
        relations = self.get_head_relations(node_type)
        return len(relations) + 1

    def num_tail_relations(self, ntype):
        relations = self.get_tail_relations(ntype)
        return len(relations) + 1

    def save_relation_weights(self, beta, global_node_idx):
        # Only save relation weights if beta has weights for all node_types in the global_node_idx batch
        if len(beta) < len(self.node_types): return

        self._betas = {}
        self._beta_avg = {}
        self._beta_std = {}
        for node_type in beta:
            relations = self.get_head_relations(node_type, True) + [node_type, ]

            with torch.no_grad():
                self._betas[node_type] = pd.DataFrame(beta[node_type].squeeze(-1).cpu().numpy(),
                                                      columns=relations,
                                                      index=global_node_idx[node_type].cpu().numpy())

                _beta_avg = np.around(beta[node_type].mean(dim=0).squeeze(-1).cpu().numpy(), decimals=3)
                _beta_std = np.around(beta[node_type].std(dim=0).squeeze(-1).cpu().numpy(), decimals=2)
                self._beta_avg[node_type] = {metapath: _beta_avg[i] for i, metapath in
                                             enumerate(relations)}
                self._beta_std[node_type] = {metapath: _beta_std[i] for i, metapath in
                                             enumerate(relations)}

    def save_attn_weights(self, node_type, attn_weights, node_idx):
        if not hasattr(self, "_betas"):
            self._betas = {}
        if not hasattr(self, "_beta_avg"):
            self._beta_avg = {}
        if not hasattr(self, "_beta_std"):
            self._beta_std = {}

        betas = attn_weights.sum(1)

        relations = self.get_head_relations(node_type, True) + [node_type, ]

        with torch.no_grad():
            self._betas[node_type] = pd.DataFrame(betas.cpu().numpy(),
                                                  columns=relations,
                                                  index=node_idx.cpu().numpy())

            _beta_avg = np.around(betas.mean(dim=0).cpu().numpy(), decimals=3)
            _beta_std = np.around(betas.std(dim=0).cpu().numpy(), decimals=2)
            self._beta_avg[node_type] = {metapath: _beta_avg[i] for i, metapath in
                                         enumerate(relations)}
            self._beta_std[node_type] = {metapath: _beta_std[i] for i, metapath in
                                         enumerate(relations)}

    def get_relation_weights(self):
        """
        Get the mean and std of relation attention weights for all nodes
        :return:
        """
        return {(metapath if "." in metapath or len(metapath) > 1 else node_type): (avg, std) \
                for node_type in self._beta_avg for (metapath, avg), (relation_b, std) in
                zip(self._beta_avg[node_type].items(), self._beta_std[node_type].items())}


def tag_negative(metapath):
    if isinstance(metapath, tuple):
        return metapath + ("neg",)
    elif isinstance(metapath, str):
        return metapath + "_neg"
    else:
        return "neg"


def untag_negative(metapath):
    if isinstance(metapath, tuple) and metapath[-1] == "neg":
        return metapath[:-1]
    elif isinstance(metapath, str):
        return metapath.strip("_neg")
    else:
        return metapath


def is_negative(metapath):
    if isinstance(metapath, tuple) and "neg" in metapath:
        return True
    elif isinstance(metapath, str) and "_neg" in metapath:
        return True
    else:
        return False


def adamic_adar(indexA, valueA, indexB, valueB, m, k, n, coalesced=False, sampling=True):
    A = SparseTensor(row=indexA[0], col=indexA[1], value=valueA,
                     sparse_sizes=(m, k), is_sorted=not coalesced)
    B = SparseTensor(row=indexB[0], col=indexB[1], value=valueB,
                     sparse_sizes=(k, n), is_sorted=not coalesced)

    deg_A = A.storage.colcount()
    deg_B = B.storage.rowcount()
    deg_normalized = 1.0 / (deg_A + deg_B).to(torch.float)
    deg_normalized[deg_normalized == float('inf')] = 0.0

    D = SparseTensor(row=torch.arange(deg_normalized.size(0), device=valueA.device),
                     col=torch.arange(deg_normalized.size(0), device=valueA.device),
                     value=deg_normalized.type_as(valueA),
                     sparse_sizes=(deg_normalized.size(0), deg_normalized.size(0)))

    out = A @ D @ B
    row, col, values = out.coo()

    num_samples = min(int(valueA.numel()), int(valueB.numel()), values.numel())
    if sampling and values.numel() > num_samples:
        idx = torch.multinomial(values, num_samples=num_samples,
                                replacement=False)
        row, col, values = row[idx], col[idx], values[idx]

    return torch.stack([row, col], dim=0), values
