import copy
import logging
from pprint import pprint
from typing import List, Dict, Tuple, Union, Optional, Any

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from colorhash import ColorHash
from fairscale.nn import auto_wrap
from moge.dataset import HeteroNodeClfDataset
from moge.model.PyG import filter_metapaths
from moge.model.PyG.utils import join_metapaths, get_edge_index_values, join_edge_indexes
from moge.model.classifier import DenseClassification, LabelGraphNodeClassifier
from moge.model.encoder import HeteroNodeEncoder, HeteroSequenceEncoder
from moge.model.losses import ClassificationLoss
from moge.model.trainer import NodeClfTrainer, print_pred_class_counts
from moge.model.utils import filter_samples_weights, process_tensor_dicts, select_batch
from torch import nn as nn, Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


class LATTEFlatNodeClf(NodeClfTrainer):
    def __init__(self, hparams, dataset: HeteroNodeClfDataset, metrics=["accuracy"],
                 collate_fn=None) -> None:
        super().__init__(hparams=hparams, dataset=dataset, metrics=metrics)
        self.head_node_type = dataset.head_node_type
        self.node_types = dataset.node_types
        self.dataset = dataset
        self.multilabel = dataset.multilabel
        self.y_types = list(dataset.y_dict.keys())
        self._name = f"LATTE-{hparams.n_layers}-{hparams.t_order}th_Link"
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

        if 'sequences' in inputs and hasattr(self, "seq_encoder"):
            h_out = self.seq_encoder.forward(inputs['sequences'])
        else:
            h_out = {}

        if len(h_out) < len(inputs["global_node_index"].keys()):
            h_out = {**h_out, **self.encoder.forward(inputs["x_dict"], global_node_idx=inputs["global_node_index"])}

        embeddings, proximity_loss, edge_index_dict = self.embedder.forward(h_dict=h_out,
                                                                            edge_index_dict=inputs["edge_index_dict"],
                                                                            global_node_idx=inputs["global_node_index"],
                                                                            sizes=inputs["sizes"],
                                                                            **kwargs)

        if hasattr(self, "classifier"):
            y_hat = self.classifier.forward(embeddings[self.head_node_type])
        else:
            y_hat = embeddings[self.head_node_type]

        return y_hat, proximity_loss, edge_index_dict

    def training_step(self, batch, batch_nb):
        X, y_true, weights = batch
        y_pred, proximity_loss, edge_pred_dict = self.forward(X)

        y_pred, y_true, weights = process_tensor_dicts(y_pred, y_true, weights)
        y_pred, y_true, weights = filter_samples_weights(Y_hat=y_pred, Y=y_true, weights=weights)
        if y_true.size(0) == 0: return torch.tensor(0.0, requires_grad=False)

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

        self.log_dict(logs, prog_bar=True, logger=True, on_step=True)

        return loss

    def validation_step(self, batch, batch_nb):
        X, y_true, weights = batch
        y_pred, proximity_loss, edge_pred_dict = self.forward(X)

        y_pred, y_true, weights = select_batch(X['batch_size'], y_pred, y_true, weights)
        y_pred, y_true, weights = filter_samples_weights(Y_hat=y_pred, Y=y_true, weights=weights)
        if y_true.size(0) == 0: return torch.tensor(0.0, requires_grad=False)

        val_loss = self.criterion.forward(y_pred, y_true, weights=weights)
        self.valid_metrics.update_metrics(y_pred, y_true)

        if proximity_loss is not None:
            val_loss = val_loss + proximity_loss

        self.log("val_loss", val_loss)

        return val_loss

    def test_step(self, batch, batch_nb):
        X, y_true, weights = batch
        y_pred, proximity_loss, edge_pred_dict = self.forward(X, save_betas=False)

        y_pred, y_true, weights = select_batch(X['batch_size'], y_pred, y_true, weights)
        y_pred, y_true, weights = filter_samples_weights(Y_hat=y_pred, Y=y_true, weights=weights)
        if y_true.size(0) == 0: return torch.tensor(0.0, requires_grad=False)

        test_loss = self.criterion(y_pred, y_true, weights=weights)

        if batch_nb == 0:
            print_pred_class_counts(y_pred, y_true, multilabel=self.dataset.multilabel)

        self.test_metrics.update_metrics(y_pred, y_true, weights=weights)

        if proximity_loss is not None:
            test_loss = test_loss + proximity_loss

        self.log("test_loss", test_loss)

        return test_loss

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


class LATTE(nn.Module):
    def __init__(self, n_layers: int, t_order: int, embedding_dim: int, num_nodes_dict: Dict[str, int],
                 metapaths: List[Tuple[str, str, str]], layer_pooling: str = None,
                 activation: str = "relu", attn_heads: int = 1, attn_activation="sharpening", attn_dropout: float = 0.5,
                 use_proximity=True, neg_sampling_ratio=2.0, edge_sampling=True,
                 hparams=None):
        super().__init__()
        self.metapaths = metapaths
        self.node_types = list(num_nodes_dict.keys())
        self.head_node_type = hparams.head_node_type

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

        layer_t_orders = {
            l: list(range(1, t_order - (n_layers - (l + 1)) + 1)) \
                if (t_order - (n_layers - (l + 1))) > 0 \
                else [1] \
            for l in reversed(range(n_layers))}

        higher_order_metapaths = copy.deepcopy(metapaths)  # Initialize another set of

        layers = []
        for l in range(n_layers):
            is_last_layer = l + 1 == n_layers
            is_output_layer = False if not hasattr(hparams,
                                                   'nb_cls_dense_size') or hparams.nb_cls_dense_size < 0 else True

            l_layer_metapaths = filter_metapaths(metapaths + higher_order_metapaths,
                                                 order=layer_t_orders[l],  # Select only up to t-order
                                                 # Skip higher-order relations that doesn't have the head node type, since it's the last output layer.
                                                 tail_type=[self.head_node_type, "GO_term"] if is_last_layer else None)

            layer = LATTEConv(input_dim=embedding_dim,
                              output_dim=hparams.n_classes if is_last_layer and is_output_layer else embedding_dim,
                              num_nodes_dict=num_nodes_dict,
                              metapaths=l_layer_metapaths,
                              layer=l, t_order=self.t_order,
                              activation=None if is_last_layer and is_output_layer else activation,
                              layernorm=False if not hasattr(hparams, "layernorm") or (
                                      is_last_layer and is_output_layer) else hparams.layernorm,
                              batchnorm=False if not hasattr(hparams, "layernorm") or (
                                      is_last_layer and is_output_layer) else hparams.batchnorm,
                              dropout=self.dropout,
                              attn_heads=attn_heads,
                              attn_activation=attn_activation,
                              attn_dropout=attn_dropout,
                              edge_threshold=hparams.edge_threshold if "edge_threshold" in hparams else 0.0,
                              use_proximity=use_proximity,
                              neg_sampling_ratio=neg_sampling_ratio, )
            if l + 1 < n_layers and layer_t_orders[l + 1] > layer_t_orders[l]:
                higher_order_metapaths = join_metapaths(l_layer_metapaths, metapaths)

            layers.append(layer)

        self.layers: List[LATTEConv] = nn.ModuleList(layers)

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')

    def configure_sharded_model(self):
        # modules are sharded across processes
        # as soon as they are wrapped with ``wrap`` or ``auto_wrap``.
        # During the forward/backward passes, weights get synced across processes
        # and de-allocated once computation is complete, saving memory.

        # Wraps the layer in a Fully Sharded Wrapper automatically
        self.layers = auto_wrap(self.layers)

    def forward(self, h_dict: Dict[str, Tensor],
                edge_index_dict: Dict[Tuple[str], Tensor],
                global_node_idx: Dict[str, Tensor],
                sizes: Dict[str, int],
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
        for l in range(self.n_layers):
            if l == 0:
                h_dict, t_loss, edge_pred_dict = self.layers[l].forward(feats=h_dict,
                                                                        edge_index_dict=edge_index_dict,
                                                                        global_node_idx=global_node_idx,
                                                                        sizes=sizes,
                                                                        save_betas=save_betas)
            else:
                h_dict, t_loss, edge_pred_dict = self.layers[l].forward(feats=h_dict,
                                                                        edge_index_dict=edge_index_dict,
                                                                        global_node_idx=global_node_idx,
                                                                        sizes=sizes,
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

    def get_attn_activation_weights(self, t):
        return dict(zip(self.layers[t].metapaths, self.layers[t].alpha_activation.detach().numpy().tolist()))

    def get_relation_weights(self, t):
        return self.layers[t].get_relation_weights()

    def get_top_relations(self, t, node_type, min_order=None):
        df = self.layers[t].get_top_relations(ntype=node_type)
        if min_order:
            df = df[df.notnull().sum(1) >= min_order]
        return df

    def get_sankey_flow(self, layer, node_type, self_loop=False, agg="median"):
        rel_attn: pd.DataFrame = self.layers[layer]._betas[node_type]
        if agg == "sum":
            rel_attn = rel_attn.sum(axis=0)
        elif agg == "median":
            rel_attn = rel_attn.median(axis=0)
        elif agg == "max":
            rel_attn = rel_attn.max(axis=0)
        elif agg == "min":
            rel_attn = rel_attn.min(axis=0)
        else:
            rel_attn = rel_attn.mean(axis=0)

        new_index = rel_attn.index.str.split(".").map(lambda tup: [str(len(tup) - i) + n for i, n in enumerate(tup)])
        all_nodes = {node for nodes in new_index for node in nodes}
        all_nodes = {node: i for i, node in enumerate(all_nodes)}

        # Links
        links = pd.DataFrame(columns=["source", "target", "value", "label", "color"])
        for i, (metapath, value) in enumerate(rel_attn.to_dict().items()):
            if len(metapath.split(".")) > 1:
                sources = [all_nodes[new_index[i][j]] for j, _ in enumerate(new_index[i][:-1])]
                targets = [all_nodes[new_index[i][j + 1]] for j, _ in enumerate(new_index[i][:-1])]

                path_links = pd.DataFrame({"source": sources,
                                           "target": targets,
                                           "value": [value, ] * len(targets),
                                           "label": [metapath, ] * len(targets)})
                links = links.append(path_links, ignore_index=True)


            elif self_loop:
                source = all_nodes[new_index[i][0]]
                links = links.append({"source": source,
                                      "target": source,
                                      "value": value,
                                      "label": metapath}, ignore_index=True)

        links["color"] = links["label"].apply(lambda label: ColorHash(label).hex)
        links = links.iloc[::-1]

        # Nodes
        node_group = [int(node[0]) for node, nid in all_nodes.items()]
        groups = [[nid for nid, node in enumerate(node_group) if node == group] for group in np.unique(node_group)]

        nodes = pd.DataFrame(columns=["label", "level", "color"])
        nodes["label"] = [node[1:] for node in all_nodes.keys()]
        nodes["level"] = [int(node[0]) for node in all_nodes.keys()]

        nodes["color"] = nodes[["label", "level"]].apply(
            lambda x: ColorHash(x["label"] + str(x["level"])).hex \
                if x["level"] % 2 == 0 \
                else ColorHash(x["label"]).hex, axis=1)

        return nodes, links


class LATTEConv(MessagePassing, pl.LightningModule):
    def __init__(self, input_dim: int, output_dim: int, num_nodes_dict: Dict[str, int], metapaths: List,
                 layer: int = 0, t_order: int = 1,
                 activation: str = "relu", attn_heads=4, attn_activation="LeakyReLU", attn_dropout=0.2,
                 layernorm=False, batchnorm=False, dropout=0.2,
                 edge_threshold=0.0, use_proximity=False, neg_sampling_ratio=1.0, verbose=False) -> None:
        super().__init__(aggr="add", flow="source_to_target", node_dim=0)
        self.layer = layer
        self.t_order = t_order
        self.node_types = list(num_nodes_dict.keys())
        self.metapaths = list(metapaths)
        print(f"LATTE {self.layer + 1} layer") if verbose else None
        pprint({ntype: [m for m in self.metapaths if m[-1] == ntype] \
                for ntype in {m[-1] for m in self.metapaths}}, width=200) if verbose else None

        self.num_nodes_dict = num_nodes_dict
        self.embedding_dim = output_dim
        self.use_proximity = use_proximity
        self.neg_sampling_ratio = neg_sampling_ratio
        self.attn_heads = attn_heads
        self.attn_dropout = attn_dropout
        self.edge_threshold = edge_threshold

        if activation == "sigmoid":
            self.activation = F.sigmoid
        elif activation == "tanh":
            self.activation = F.tanh
        elif activation == "relu":
            self.activation = F.relu
        else:
            print(f"Embedding activation arg `{activation}` did not match, so uses linear activation.")

        if batchnorm:
            self.batchnorm_l = torch.nn.ModuleDict({
                node_type: nn.BatchNorm1d(output_dim) \
                for node_type in self.node_types})
            self.batchnorm_r = torch.nn.ModuleDict({
                node_type: nn.BatchNorm1d(output_dim) \
                for node_type in self.node_types})

        if dropout:
            self.dropout = nn.Dropout(p=dropout)

        self.linear_l = nn.ModuleDict(
            {node_type: nn.Linear(input_dim, output_dim, bias=True) \
             for node_type in self.node_types})  # W.shape (F x F)
        self.linear_r = nn.ModuleDict(
            {node_type: nn.Linear(input_dim, output_dim, bias=True) \
             for node_type in self.node_types})  # W.shape (F x F}

        self.out_channels = self.embedding_dim // attn_heads
        self.attn = nn.Parameter(torch.rand((len(self.metapaths), attn_heads, self.out_channels * 2)))

        self.rel_attn_l = nn.ParameterDict({
            ntype: nn.Parameter(Tensor(attn_heads, self.out_channels)) \
            for ntype in self.node_types})
        self.rel_attn_r = nn.ParameterDict({
            ntype: nn.Parameter(Tensor(attn_heads, self.out_channels)) \
            for ntype in self.node_types})

        if attn_activation == "sharpening":
            self.alpha_activation = nn.Parameter(Tensor(len(self.metapaths)).fill_(1.0))
        elif attn_activation == "PReLU":
            self.alpha_activation = nn.PReLU(init=0.2)
        elif attn_activation == "LeakyReLU":
            self.alpha_activation = nn.LeakyReLU(negative_slope=0.2)
        else:
            print(f"WARNING: alpha_activation `{attn_activation}` did not match, so used linear activation")
            self.alpha_activation = None

        if layernorm:
            self.layernorm = torch.nn.ModuleDict({
                node_type: nn.LayerNorm(output_dim) \
                for node_type in self.node_types})

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu', 0.2)
        for i, metapath in enumerate(self.metapaths):
            nn.init.xavier_normal_(self.attn[i], gain=gain)

        gain = nn.init.calculate_gain('relu')
        for node_type in self.linear_l:
            nn.init.xavier_normal_(self.linear_l[node_type].weight, gain=gain)
            nn.init.xavier_normal_(self.linear_r[node_type].weight, gain=gain)

        gain = nn.init.calculate_gain('leaky_relu', 0.2)
        for ntype, rel_attn in self.rel_attn_l.items():
            nn.init.xavier_normal_(rel_attn, gain=gain)
        for ntype, rel_attn in self.rel_attn_r.items():
            nn.init.xavier_normal_(rel_attn, gain=gain)

    def get_beta_weights(self, query: Tensor, key: Tensor, ntype: str) -> Tensor:
        alpha_l = (query * self.rel_attn_l[ntype]).sum(dim=-1)
        alpha_r = (key * self.rel_attn_r[ntype][None, :, :]).sum(dim=-1)

        beta = alpha_l[:, None, :] + alpha_r
        beta = F.leaky_relu(beta, negative_slope=0.2)
        beta = F.softmax(beta, dim=1)
        beta = F.dropout(beta, p=self.attn_dropout, training=self.training)
        return beta

    def projection(self, feats, linear_projs: Dict[str, nn.Linear], batch_norms: Dict[str, nn.BatchNorm1d]):
        h_dict = {ntype: linear_projs[ntype].forward(x) for ntype, x in feats.items()}

        if hasattr(self, "dropout"):
            h_dict = {ntype: self.dropout(h_dict[ntype]) for ntype in h_dict}

        if batch_norms:
            h_dict = {ntype: batch_norms[ntype](h_dict[ntype]) for ntype in h_dict}

        h_dict = {ntype: h_dict[ntype].view(feats[ntype].size(0), self.attn_heads, self.out_channels) for ntype in
                  h_dict}

        return h_dict

    def forward(self, feats: Dict[str, Tensor],
                edge_index_dict: Dict[Tuple[str, str, str], Union[Tensor, Tuple[Tensor, Tensor]]],
                global_node_idx: Dict[str, Tensor],
                sizes: Dict[str, int],
                save_betas=False) -> \
            Tuple[Dict[str, Tensor], Optional[Any], Dict[Tuple[str, str, str], Tensor]]:
        """
        Args:
            feats: a dict of node attributes indexed ntype
            global_node_idx: A dict of index values indexed by ntype in this mini-batch sampling
            edge_index_dict: Sparse adjacency matrices for each metapath relation. A dict of edge_index indexed by metapath
        Returns:
             output_emb, loss:
        """
        l_dict = self.projection(feats, linear_projs=self.linear_l,
                                 batch_norms=self.batchnorm_l if hasattr(self, "batchnorm_l") else None)
        r_dict = self.projection(feats, linear_projs=self.linear_r,
                                 batch_norms=self.batchnorm_r if hasattr(self, "batchnorm_r") else None)

        # For each metapath in a node_type, use GAT message passing to aggregate h_j neighbors
        h_out = {}
        beta = {}
        # print("> Layer", self.layer + 1)

        for ntype in global_node_idx:
            if global_node_idx[ntype].size(0) == 0: continue
            h_out[ntype], edge_pred_dict = self.agg_relation_neighbors(ntype=ntype, l_dict=l_dict,
                                                                       r_dict=r_dict,
                                                                       edge_index_dict=edge_index_dict,
                                                                       sizes=sizes)

            h_out[ntype][:, -1] = r_dict[ntype]

            # Soft-select the relation-specific embeddings by a weighted average with beta[node_type]
            beta[ntype] = self.get_beta_weights(query=r_dict[ntype], key=h_out[ntype], ntype=ntype)
            h_out[ntype] = h_out[ntype] * beta[ntype].unsqueeze(-1)
            h_out[ntype] = h_out[ntype].sum(1).view(h_out[ntype].size(0), self.embedding_dim)

            if hasattr(self, "activation"):
                h_out[ntype] = self.activation(h_out[ntype])

            if hasattr(self, "layernorm"):
                h_out[ntype] = self.layernorm[ntype](h_out[ntype])

            # if hasattr(self, "dropout"):
            #     h_out[ntype] = self.dropout(h_out[ntype])

        if not self.training and save_betas:
            self.save_relation_weights({ntype: beta[ntype].mean(1) for ntype in beta}, global_node_idx)

        proximity_loss = None

        return h_out, proximity_loss, edge_pred_dict

    def agg_relation_neighbors(self, ntype: str,
                               l_dict: Dict[str, Tensor],
                               r_dict: Dict[str, Tensor],
                               edge_index_dict: Dict[Tuple[str], Tensor],
                               sizes: Dict[str, int]):
        # Initialize embeddings, size: (num_nodes, num_relations, embedding_dim)
        emb_relations = torch.zeros(
            size=(sizes[ntype],
                  self.num_tail_relations(ntype),
                  self.attn_heads,
                  self.out_channels)).type_as(self.attn)
        relations = self.get_tail_relations(ntype)

        # First order
        edge_pred_dict = {}
        for metapath in self.get_tail_relations(ntype, order=1):
            if metapath not in edge_index_dict or edge_index_dict[metapath] is None \
                    or edge_index_dict[metapath].size(1) == 0: continue
            head, tail = metapath[0], metapath[-1]
            num_node_head, num_node_tail = sizes[head], sizes[tail]

            edge_index, values = get_edge_index_values(edge_index_dict[metapath])
            if edge_index is None or edge_index.size(1) == 0: continue

            # print("\n", metapath)
            # print("x_dict", tensor_sizes({tail: h_dict[tail], head: h_dict[head]}))
            # print("global_node_idx", tensor_sizes(global_node_idx))
            # print("edge_index", tensor_sizes(edge_index), edge_index.max(1).values)
            # print("alpha", tensor_sizes({tail: alpha_r[metapath], head: alpha_l[metapath]}))
            # print({"num_node_tail": num_node_tail, "num_node_head": num_node_head})

            # Propapate flows from target nodes to source nodes
            out = self.propagate(
                edge_index=edge_index,
                x=(l_dict[head], r_dict[tail]),
                size=(num_node_head, num_node_tail),
                metapath=str(metapath),
                metapath_idx=self.metapaths.index(metapath),
                values=None)

            emb_relations[:, relations.index(metapath)] = out
            # print(ntype, out.shape, emb_relations.shape)
            edge_pred_dict[metapath] = (edge_index, self._alpha)
            self._alpha = None
        # print("\n>", ntype, self.get_tail_relations(ntype))

        remaining_orders = list(range(2, min(self.layer + 1, self.t_order) + 1))
        higher_relations = self.get_tail_relations(ntype, order=remaining_orders)
        # print("\t t-order", self.t_order, "remaining_orders", remaining_orders, "higher_relations", higher_relations)

        # Create high-order edge index for next layer (but may not be used for aggregation)
        higher_order_edge_index = join_edge_indexes(edge_index_dict_A=edge_index_dict,
                                                    edge_index_dict_B=edge_pred_dict,
                                                    sizes=sizes,
                                                    metapaths=higher_relations,
                                                    edge_sampling=False)
        # print("higher_order_edge_index")
        # pprint(tensor_sizes(higher_order_edge_index), width=250)

        # Aggregate higher order relations
        for metapath in higher_relations:
            if metapath not in higher_order_edge_index or higher_order_edge_index[metapath] == None: continue

            edge_index, values = get_edge_index_values(higher_order_edge_index[metapath], filter_edge=False)
            if edge_index is None or edge_index.size(1) == 0: continue

            head, tail = metapath[0], metapath[-1]
            head_size_in, tail_size_out = sizes[head], sizes[tail]

            # Select the right t-order context node presentations based on the order of the metapath
            # Propapate flows from higher order source nodes to target nodes
            out = self.propagate(
                edge_index=edge_index,
                x=(l_dict[head], r_dict[tail]),
                size=(head_size_in, tail_size_out),
                metapath_idx=self.metapaths.index(metapath),
                metapath=str(metapath),
                values=None)
            emb_relations[:, relations.index(metapath)] = out

            edge_pred_dict[metapath] = (edge_index, self._alpha)
            self._alpha = None

        # print(f'edge_pred_dict')
        # pprint(tensor_sizes(edge_pred_dict),width=300)

        return emb_relations, edge_pred_dict

    def message(self, x_j, x_i, index, ptr, size_i, metapath_idx, metapath, values=None):
        if values is None:
            x = torch.cat([x_i, x_j], dim=2)
            if isinstance(self.alpha_activation, Tensor):
                x = self.alpha_activation[metapath_idx] * F.leaky_relu(x, negative_slope=0.2)
            elif isinstance(self.alpha_activation, nn.Module):
                x = self.alpha_activation(x)

            alpha = (x * self.attn[metapath_idx]).sum(dim=-1)
            alpha = softmax(alpha, index=index, ptr=ptr, num_nodes=size_i)
        else:
            if values.dim() == 1:
                values = values.unsqueeze(-1)
            alpha = values
            # alpha = softmax(alpha, index=index, ptr=ptr, num_nodes=size_i)

        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.attn_dropout, training=self.training)

        return x_j * alpha.unsqueeze(-1)

    def attn_activation(self, alpha, metapath_id):
        if isinstance(self.alpha_activation, Tensor):
            return self.alpha_activation[metapath_id] * alpha
        elif isinstance(self.alpha_activation, nn.Module):
            return self.alpha_activation(alpha)
        else:
            return alpha

    def get_head_relations(self, head_node_type, order=None, str_form=False) -> list:
        relations = filter_metapaths(self.metapaths, order=order, head_type=head_node_type)

        if str_form:
            relations = [".".join(metapath) if isinstance(metapath, tuple) else metapath \
                         for metapath in relations]

        return relations

    def get_tail_relations(self, tail_node_type, order=None, str_form=False) -> list:
        relations = filter_metapaths(self.metapaths, order=order, tail_type=tail_node_type)

        if str_form:
            relations = [".".join(metapath) if isinstance(metapath, tuple) else metapath \
                         for metapath in relations]
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

    def save_relation_weights(self, beta: Dict[str, Tensor], global_node_idx: Dict[str, Tensor]):
        # Only save relation weights if beta has weights for all node_types in the global_node_idx batch
        if len(beta) < len(global_node_idx): return

        self._betas = {}
        self._beta_avg = {}
        self._beta_std = {}

        for ntype in beta:
            relations = self.get_tail_relations(ntype, str_form=True) + [ntype, ]
            if len(relations) <= 1: continue

            with torch.no_grad():
                self._betas[ntype] = pd.DataFrame(beta[ntype].squeeze(-1).cpu().numpy(),
                                                  columns=relations,
                                                  index=global_node_idx[ntype].cpu().numpy())

                _beta_avg = np.around(beta[ntype].mean(dim=0).squeeze(-1).cpu().numpy(), decimals=3)
                _beta_std = np.around(beta[ntype].std(dim=0).squeeze(-1).cpu().numpy(), decimals=2)

                self._beta_avg[ntype] = {metapath: _beta_avg[i] for i, metapath in enumerate(relations)}
                self._beta_std[ntype] = {metapath: _beta_std[i] for i, metapath in enumerate(relations)}

    def save_attn_weights(self, node_type: str, attn_weights: Tensor, node_idx: Tensor):
        if not hasattr(self, "_betas"):
            self._betas = {}
        if not hasattr(self, "_beta_avg"):
            self._beta_avg = {}
        if not hasattr(self, "_beta_std"):
            self._beta_std = {}

        betas = attn_weights.sum(1)

        relations = self.get_tail_relations(node_type, str_form=True) + [node_type, ]

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

