from typing import Dict, Tuple, List
import copy
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch_geometric.data.sampler import EdgeIndex
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

from moge.module.sampling import negative_sample
from .utils import *
from ..utils import tensor_sizes

class LATTE(nn.Module):
    def __init__(self, n_layers: int, t_order: int, embedding_dim: int, in_channels_dict: dict, num_nodes_dict: dict,
                 metapaths: list,
                 activation: str = "relu", attn_heads=1, attn_activation="sharpening", attn_dropout=0.5,
                 use_proximity=True, neg_sampling_ratio=2.0, edge_sampling=True, cpu_embeddings=False,
                 layer_pooling=False, hparams=None):
        super(LATTE, self).__init__()
        self.metapaths = metapaths
        self.node_types = list(num_nodes_dict.keys())
        self.embedding_dim = embedding_dim * n_layers
        self.use_proximity = use_proximity
        self.t_order = t_order
        self.n_layers = n_layers
        self.neg_sampling_ratio = neg_sampling_ratio
        self.edge_sampling = edge_sampling
        self.edge_threshold = hparams.edge_threshold
        self.head_node_type = hparams.head_node_type

        self.layer_pooling = layer_pooling

        # align the dimension of different types of nodes
        self.feature_projection = nn.ModuleDict({
            ntype: nn.Linear(in_channels_dict[ntype], embedding_dim) for ntype in in_channels_dict
        })
        if hparams.batchnorm:
            self.batchnorm = nn.ModuleDict({
                ntype: nn.BatchNorm1d(embedding_dim) for ntype in in_channels_dict
            })
        self.dropout = hparams.dropout if hasattr(hparams, "dropout") else 0.0

        layers = []
        higher_order_metapaths = copy.deepcopy(metapaths)  # Initialize a nother set of
        for l in range(n_layers):
            is_last_layer = (l + 1 == n_layers)
            is_output_layer = is_last_layer and (hparams.nb_cls_dense_size < 0)

            l_layer_metapaths = filter_metapaths(metapaths + higher_order_metapaths,
                                                 order=range(1, t_order + 1),  # Select only up to t-order
                                                 # Skip higher-order relations that doesn't have the head node type, since it's the last output layer.
                                                 tail_type=self.head_node_type if is_last_layer else None)

            layers.append(
                LATTEConv(input_dim=embedding_dim,
                          output_dim=hparams.n_classes if is_output_layer else embedding_dim,
                          num_nodes_dict=num_nodes_dict,
                          metapaths=l_layer_metapaths,
                          activation=None if is_output_layer else activation,
                          batchnorm=False if not hasattr(hparams,
                                                         "batchnorm") or is_output_layer else hparams.batchnorm,
                          layernorm=False if not hasattr(hparams,
                                                         "layernorm") or is_output_layer else hparams.layernorm,
                          dropout=False if not hasattr(hparams,
                                                       "dropout") or is_output_layer else hparams.dropout,
                          attn_heads=attn_heads,
                          attn_activation=attn_activation,
                          attn_dropout=attn_dropout,
                          use_proximity=use_proximity,
                          neg_sampling_ratio=neg_sampling_ratio))

            higher_order_metapaths = join_metapaths(l_layer_metapaths, metapaths)

        self.layers = nn.ModuleList(layers)

        # If some node type are not attributed, instantiate nn.Embedding for them
        if isinstance(in_channels_dict, dict):
            non_attr_node_types = (num_nodes_dict.keys() - in_channels_dict.keys())
        else:
            non_attr_node_types = []
        if len(non_attr_node_types) > 0:
            print("num_nodes_dict", num_nodes_dict)

            if cpu_embeddings:
                print("Embedding.device = 'cpu'")
                self.embeddings = {ntype: nn.Embedding(num_embeddings=num_nodes_dict[ntype],
                                                       embedding_dim=embedding_dim,
                                                       sparse=True,
                                                       _weight=hparams.embeddings[
                                                           ntype] if "embeddings" in hparams else None).cpu() \
                                   for ntype in non_attr_node_types}
            else:
                print("Embedding.device = 'gpu'")
                self.embeddings = nn.ModuleDict(
                    {ntype: nn.Embedding(num_embeddings=num_nodes_dict[ntype],
                                         embedding_dim=embedding_dim,
                                         sparse=True,
                                         _weight=hparams.embeddings[ntype] if "embeddings" in hparams else None) \
                     for ntype in non_attr_node_types})
        else:
            self.embeddings = None

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        for ntype in self.feature_projection:
            nn.init.xavier_normal_(self.feature_projection[ntype].weight, gain=gain)

        if self.embeddings is not None and len(self.embeddings.keys()) > 0:
            for ntype in self.embeddings:
                self.embeddings[ntype].reset_parameters()

    def transform_inp_feats(self, node_feats, global_node_idx):
        h_dict = {}
        for ntype in self.node_types:
            if ntype in node_feats:
                h_dict[ntype] = self.feature_projection[ntype](node_feats[ntype])
                if hasattr(self, "batchnorm"):
                    h_dict[ntype] = self.batchnorm[ntype](h_dict[ntype])

                h_dict[ntype] = F.relu(h_dict[ntype])
                if self.dropout:
                    h_dict[ntype] = F.dropout(h_dict[ntype], p=self.dropout, training=self.training)

            else:
                h_dict[ntype] = self.embeddings[ntype].weight[global_node_idx[ntype]].to(
                    global_node_idx[self.node_types[0]].device)
        return h_dict

    def forward(self, node_feats: dict, adjs: List[Dict[Tuple, EdgeIndex]], sizes: List[Dict[str, Tuple[int]]],
                global_node_idx: dict, save_betas=False):
        """
        This
        :param node_feats: Dict of <node_type>:<tensor size (batch_size, in_channels)>. If nodes are not attributed, then pass an empty dict.
        :param global_node_idx: Dict of <node_type>:<int tensor size (batch_size,)>
        :param adjs: Dict of <metapath>:<tensor size (2, num_edge_index)>
        :param save_betas: whether to save _beta values for batch
        :return embedding_output, proximity_loss, edge_pred_dict:
        """
        proximity_loss = torch.tensor(0.0,
                                      device=global_node_idx[self.node_types[0]].device) if self.use_proximity else None

        h_out = self.transform_inp_feats(node_feats, global_node_idx)

        h_in_layers = {ntype: [] for ntype in global_node_idx}
        h_out_layers = {ntype: [] for ntype in global_node_idx}
        for l in range(self.n_layers):
            if l == 0:
                edge_index_dict = adjs[l]
            else:
                edge_index_dict = join_edge_indexes(edge_index_dict_A=edge_pred_dict,
                                                    edge_index_dict_B=adjs[l],
                                                    sizes=sizes,
                                                    metapaths=self.layers[l].metapaths,
                                                    edge_threshold=self.edge_threshold,
                                                    edge_sampling=self.edge_sampling)

            # print("\n", l, "\t METAPATHS", [".".join([d[0] for d in k]) for k in self.layers[l].metapaths],
            #       "\n\t LOCAL NODES",
            #       {ntype: list(nids.shape) for ntype, nids in global_node_idx.items()})
            # print("\t EDGE_INDEX_DICT",
            #       {".".join([k[0] for k in m]): (eid[0].max(1).values, eid[1].shape) if isinstance(eid,
            #                                                                                        tuple) else eid.max(
            #           1).values
            #        for m, eid in edge_index_dict.items()})

            global_node_idx = {
                ntype: global_node_idx[ntype][: sizes[l][ntype][1]] \
                for ntype in global_node_idx if sizes[l][ntype][1] is not None}
            (h_in, h_out), t_loss, edge_pred_dict = self.layers[l].forward(x=h_out,
                                                                           prev_h_in=h_in_layers,
                                                                           edge_index_dict=edge_index_dict,
                                                                           size=sizes[l],
                                                                           global_node_idx=global_node_idx,
                                                                           save_betas=save_betas)

            # print("\t EDGE_PRED_DICT",
            #       {".".join([k[0] for k in m]): e_attr.shape for m, (eid, e_attr) in edge_pred_dict.items()})

            # Add the h_in embeddings to
            if l < self.n_layers and self.t_order > 1:
                for ntype in h_in:
                    h_in_layers[ntype].append(h_in[ntype])
                    if len(h_in_layers[ntype]) > self.t_order:
                        h_in_layers[ntype].pop(0)

            if self.use_proximity and t_loss is not None:
                proximity_loss += t_loss

            if self.layer_pooling != "last":
                h_out_layers[self.head_node_type].append(h_out[self.head_node_type][:sizes[-1][self.head_node_type][1]])

        if self.layer_pooling == "last" or self.n_layers == 1:
            out = h_out

        elif self.layer_pooling == "max":
            out = {ntype: torch.stack(h_list, dim=1) for ntype, h_list in h_out_layers.items() \
                   if len(h_list) > 0}
            out = {ntype: h_s.max(1).values for ntype, h_s in out.items()}

        elif self.layer_pooling == "mean":
            out = {ntype: torch.stack(h_list, dim=1) for ntype, h_list in h_out_layers.items() \
                   if len(h_list) > 0}
            out = {ntype: torch.mean(h_s, dim=1) for ntype, h_s in out.items()}

        elif self.layer_pooling == "concat":
            out = {ntype: torch.cat(h_list, dim=1) \
                   for ntype, h_list in h_out_layers.items() \
                   if len(h_list) > 0}

        return out, proximity_loss, edge_pred_dict

    def get_attn_activation_weights(self, t):
        return dict(zip(self.layers[t].metapaths, self.layers[t].alpha_activation.detach().numpy().tolist()))

    def get_relation_weights(self, t):
        return self.layers[t].get_relation_weights()


class LATTEConv(MessagePassing, pl.LightningModule):
    def __init__(self, input_dim: Dict[str, int], output_dim: int,
                 num_nodes_dict: Dict[str, int], metapaths: list,
                 activation: str = "relu", batchnorm=False, layernorm=False, dropout=0.0,
                 attn_heads=4, attn_activation="sharpening", attn_dropout=0.2,
                 use_proximity=False, neg_sampling_ratio=1.0) -> None:
        super(LATTEConv, self).__init__(aggr="add", flow="source_to_target", node_dim=0)
        self.node_types = list(num_nodes_dict.keys())
        self.metapaths = list(metapaths)
        self.num_nodes_dict = num_nodes_dict
        self.embedding_dim = output_dim
        self.use_proximity = use_proximity
        self.neg_sampling_ratio = neg_sampling_ratio
        self.attn_heads = attn_heads
        self.attn_dropout = attn_dropout
        print("\n LATTE", [".".join([k[0].upper() if i % 2 == 0 else k[0].lower() for i, k in enumerate(m)]) for m in
                           sorted(metapaths)])

        if activation == "sigmoid":
            self.activation = F.sigmoid
        elif activation == "tanh":
            self.activation = F.tanh
        elif activation == "relu":
            self.activation = F.relu
        else:
            print(f"Embedding activation arg `{activation}` did not match, so uses linear activation.")

        self.dropout = dropout
        if batchnorm:
            self.batchnorm = torch.nn.ModuleDict({
                node_type: nn.BatchNorm1d(output_dim) \
                for node_type in self.node_types})
        if layernorm:
            self.layernorm = torch.nn.ModuleDict({
                node_type: nn.LayerNorm(output_dim) \
                for node_type in self.node_types})

        self.conv = torch.nn.ModuleDict(
            {node_type: torch.nn.Conv1d(
                in_channels=input_dim,
                out_channels=self.num_head_relations(node_type),
                kernel_size=1) \
                for node_type in self.node_types})  # W_phi.shape (D x F)

        self.linear_l = nn.ModuleDict(
            {node_type: nn.Linear(input_dim, output_dim, bias=True) \
             for node_type in self.node_types})  # W.shape (F x F)

        self.linear_r = nn.ModuleDict(
            {node_type: nn.Linear(input_dim, output_dim, bias=True) \
             for node_type in self.node_types})  # W.shape (F x F}

        self.linear_prev = nn.ModuleDict(
            {node_type: nn.Linear(input_dim, output_dim, bias=True) \
             for node_type in self.node_types})  # W.shape (F x F)

        self.out_channels = self.embedding_dim // attn_heads
        self.attn = nn.Parameter(torch.Tensor(len(self.metapaths), attn_heads, self.out_channels * 2))

        if attn_activation == "sharpening":
            self.alpha_activation = nn.Parameter(torch.Tensor(len(self.metapaths)).fill_(1.0))
        elif attn_activation == "PReLU":
            self.alpha_activation = nn.PReLU(num_parameters=attn_heads, init=0.2)
        elif attn_activation == "LeakyReLU":
            self.alpha_activation = nn.LeakyReLU(negative_slope=0.2)
        else:
            print(f"WARNING: alpha_activation `{attn_activation}` did not match, so used linear activation")
            self.alpha_activation = None

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu', 0.2)
        for i, metapath in enumerate(self.metapaths):
            nn.init.xavier_normal_(self.attn[i], gain=gain)

        gain = nn.init.calculate_gain('relu')
        for node_type in self.linear_l:
            nn.init.xavier_normal_(self.linear_l[node_type].weight, gain=gain)
        for node_type in self.linear_r:
            nn.init.xavier_normal_(self.linear_r[node_type].weight, gain=gain)

        for node_type in self.conv:
            nn.init.xavier_normal_(self.conv[node_type].weight, gain=1)

    def forward(self, x: Dict[str, torch.Tensor],
                prev_h_in: Dict[str, List[torch.Tensor]],
                edge_index_dict: Dict[Tuple, torch.Tensor],
                size: Dict[str, Tuple[int]],
                global_node_idx: Dict[str, torch.Tensor],
                save_betas=False):
        """

        :param x: a dict of "source" node representations
        :param prev_h_in: Context embedding of the previous order, required for t >= 2. Default: None (if first order). A dict of (node_type: tensor)
        :param global_node_idx: A dict of index values indexed by node_type in this mini-batch sampling
        :param edge_index_dict: Sparse adjacency matrices for each metapath relation. A dict of edge_index indexed by metapath

        :return: output_emb, loss
        """
        x_r = {ntype: x[ntype][: size[ntype][1]] \
               for ntype in x if size[ntype][1] is not None}

        l_dict = self.get_h_dict(x, left_right="left")
        r_dict = self.get_h_dict(x_r, left_right="right")

        prev_dict = prev_h_in
        for head in {m[0] for m in edge_index_dict}:
            for metapath in filter_metapaths(edge_index_dict, head_type=head):
                order = len(metapath[1::2])
                if order == 1: continue
                h_source = prev_dict[head][-(order - 1)]

                orig_shape = h_source.shape
                h_source = self.linear_prev[head].forward(h_source.view(orig_shape[0], self.embedding_dim))

                prev_dict[head][-(order - 1)] = h_source.view(orig_shape)

        # Predict relations attention coefficients
        beta = self.get_beta_weights(r_dict)
        # Save beta weights from testing samples
        if not self.training: self.save_relation_weights(beta, global_node_idx)

        # For each metapath in a node_type, use GAT message passing to aggregate h_j neighbors
        out = {}
        alpha_dict = {}
        for ntype in global_node_idx:
            out[ntype], alphas = self.agg_relation_neighbors(node_type=ntype,
                                                             l_dict=l_dict,
                                                             r_dict=r_dict,
                                                             h_layers=prev_dict,
                                                             edge_index_dict=edge_index_dict,
                                                             size=size)
            out[ntype][:, -1] = r_dict[ntype].view(-1, self.embedding_dim)

            # Soft-select the relation-specific embeddings by a weighted average with beta[node_type]
            out[ntype] = torch.bmm(out[ntype].permute(0, 2, 1), beta[ntype]).squeeze(-1)

            if hasattr(self, "layernorm"):
                out[ntype] = self.layernorm[ntype](out[ntype])

            if hasattr(self, "activation"):
                out[ntype] = self.activation(out[ntype])

            if self.dropout:
                out[ntype] = F.dropout(out[ntype], p=self.dropout, training=self.training)

            if alphas:
                alpha_dict.update(alphas)

        proximity_loss, edge_pred_dict = None, {}
        if self.use_proximity:
            proximity_loss, _ = self.proximity_loss(edge_index_dict,
                                                    l_dict=l_dict, r_dict=r_dict,
                                                    global_node_idx=global_node_idx)
        # print("\t\t ALPHA_DICT", [".".join([d[0] for d in k]) for k in alpha_dict.keys()])
        # print("\t\t EDGE_INDEX_DICT", [".".join([d[0] for d in k]) for k in edge_index_dict.keys()])
        for metapath, edge_index in edge_index_dict.items():
            if metapath in alpha_dict:
                edge_pred_dict[metapath] = (edge_index[0] \
                                                if isinstance(edge_index, tuple) else edge_index,
                                            alpha_dict[metapath])
            else:
                edge_pred_dict[metapath] = edge_index

        return (l_dict, out), proximity_loss, edge_pred_dict

    def agg_relation_neighbors(self, node_type: str,
                               l_dict: Dict[str, torch.Tensor],
                               r_dict: Dict[str, torch.Tensor],
                               h_layers: Dict[str, List[torch.Tensor]],
                               edge_index_dict: Dict[Tuple, torch.Tensor], size: Dict[str, Tuple[int]]):
        # Initialize embeddings, size: (num_nodes, num_relations, embedding_dim)
        emb_relations = torch.zeros(
            size=(r_dict[node_type].size(0),
                  self.num_head_relations(node_type),
                  self.embedding_dim)).type_as(self.conv[node_type].weight)

        alpha = {}
        for i, metapath in enumerate(self.get_head_relations(node_type)):
            if metapath not in edge_index_dict or edge_index_dict[metapath] == None: continue
            head, tail = metapath[0], metapath[-1]

            edge_index, values = get_edge_index_values(edge_index_dict[metapath], filter_edge=False)
            if edge_index is None:
                continue

            # Select the right t-order context node presentations based on the order of the metapath
            order = len(metapath[1::2])
            if order == 1:
                h_source = l_dict[head]
                head_size_in, tail_size_out = size[head][0], size[tail][1]
            else:
                # print(f"h_layers[{head}]", tensor_sizes(h_layers[head]), "order", order)
                h_source = h_layers[head][-(order - 1)]
                head_size_in, tail_size_out = h_source.size(0), size[tail][1]

            h_target = r_dict[tail]

            # Propapate flows from target nodes to source nodes
            try:
                out = self.propagate(
                    edge_index=edge_index,
                    x=(h_source, h_target),
                    size=(head_size_in, tail_size_out),
                    metapath_idx=self.metapaths.index(metapath))
                emb_relations[:, i] = out.view(-1, self.embedding_dim)
            except Exception as e:
                print(metapath, edge_index.max(1).values, head_size_in, tail_size_out)
                raise e

            alpha[metapath] = self._alpha.max(1).values  # Select max attn value across multi-head attn.
            # if relation_weights is not None:
            #     alpha[metapath] = alpha[metapath] * relation_weights[:, i].squeeze(-1)[edge_index[0]]
            self._alpha = None

        return emb_relations, alpha

    def message(self, x_j, x_i, index, ptr, size_i, metapath_idx):
        x = torch.cat([x_i, x_j], dim=2)
        if isinstance(self.alpha_activation, nn.Module):
            x = self.alpha_activation(x)
        else:
            x = self.alpha_activation[metapath_idx] * F.leaky_relu(x, negative_slope=0.2)

        alpha = (x * self.attn[metapath_idx]).sum(dim=-1)
        alpha = softmax(alpha, index=index, ptr=ptr, num_nodes=size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.attn_dropout, training=self.training)

        return x_j * alpha.unsqueeze(-1)

    def get_h_dict(self, input, left_right="left"):
        h_dict = {}
        for ntype in input:
            if left_right == "left":
                h_dict[ntype] = self.linear_l[ntype].forward(input[ntype])
            elif left_right == "right":
                h_dict[ntype] = self.linear_r[ntype].forward(input[ntype])

            h_dict[ntype] = h_dict[ntype].view(-1, self.attn_heads, self.out_channels)

        return h_dict

    def get_beta_weights(self, h_dict):
        beta = {}
        for node_type in h_dict:
            beta[node_type] = self.conv[node_type].forward(h_dict[node_type].view(-1, self.embedding_dim).unsqueeze(-1))
            beta[node_type] = torch.softmax(beta[node_type], dim=1)

        return beta

    def predict_scores(self, edge_index, l_dict, r_dict, metapath, logits=False):
        assert metapath in self.metapaths, f"If metapath `{metapath}` is tag_negative()'ed, then pass it with untag_negative()"
        metapath_idx = self.metapaths.index(metapath)
        head, tail = metapath[0], metapath[-1]

        x = torch.cat([l_dict[head][edge_index[0]], r_dict[tail][edge_index[1]]], dim=2)
        if isinstance(self.alpha_activation, nn.Module):
            x = self.alpha_activation(x)
        else:
            x = self.alpha_activation[metapath_idx] * F.leaky_relu(x, negative_slope=0.2)

        e_pred = (x * self.attn[metapath_idx]).sum(dim=-1)

        if e_pred.size(1) > 1:
            e_pred = e_pred.max(1).values

        if logits:
            return e_pred
        else:
            return F.sigmoid(e_pred)

    def proximity_loss(self, edge_index_dict, l_dict, r_dict, global_node_idx):
        """
        For each relation/metapath type given in `edge_index_dict`, this function both predict link scores and computes
        the NCE loss for both positive and negative (sampled) links. For each relation type in `edge_index_dict`, if the
        negative metapath is not included, then the function automatically samples for random negative edges. And, if it
        is included, then computes the NCE loss over the given negative edges. This function returns the scores of the
        predicted positive and negative edges.

        :param edge_index_dict (dict): Dict of <relation/metapath>: <Tensor(2, num_edges)>
        :param alpha_l (dict): Dict of <node_type>:<alpha_l tensor>
        :param alpha_r (dict): Dict of <node_type>:<alpha_r tensor>
        :param global_node_idx (dict): Dict of <node_type>:<Tensor(node_idx,)>
        :return loss, edge_pred_dict: NCE loss. edge_pred_dict will contain both positive relations of shape (num_edges,) and negative relations of shape (num_edges*num_neg_edges, )
        """
        loss = torch.tensor(0.0, dtype=torch.float, device=self.conv[self.node_types[0]].weight.device)
        for metapath, edge_index in edge_index_dict.items():
            # KL Divergence over observed positive edges or negative edges (if included)
            if isinstance(edge_index, tuple):  # Weighted edges
                edge_index, values = edge_index
            else:
                values = 1.0
            if edge_index is None: continue

            if not is_negative(metapath):
                e_pred_logits = self.predict_scores(edge_index, l_dict, r_dict, metapath, logits=True)
                loss += -torch.mean(values * F.logsigmoid(e_pred_logits), dim=-1)
            elif is_negative(metapath):
                e_pred_logits = self.predict_scores(edge_index, l_dict, r_dict, untag_negative(metapath), logits=True)
                loss += -torch.mean(F.logsigmoid(-e_pred_logits), dim=-1)


            # Only need to sample for negative edges if negative metapath is not included
            if not is_negative(metapath) and tag_negative(metapath) not in edge_index_dict:
                neg_edge_index = negative_sample(edge_index,
                                                 M=global_node_idx[metapath[0]].size(0),
                                                 N=global_node_idx[metapath[-1]].size(0),
                                                 n_sample_per_edge=self.neg_sampling_ratio)
                if neg_edge_index is None or neg_edge_index.size(1) <= 1: continue

                e_neg_logits = self.predict_scores(neg_edge_index, l_dict, r_dict, metapath, logits=True)
                loss += -torch.mean(F.logsigmoid(-e_neg_logits), dim=-1)

        loss = torch.true_divide(loss, max(len(edge_index_dict) * 2, 1))
        return loss


    def get_head_relations(self, head_node_type, to_str=False) -> list:
        relations = [".".join(metapath) if to_str and isinstance(metapath, tuple) else metapath for metapath in
                     self.metapaths if
                     metapath[-1] == head_node_type]
        return relations

    def num_head_relations(self, node_type) -> int:
        """
        Return the number of metapaths with head node type equals to :param node_type: and plus one for none-selection.
        :param node_type (str):
        :return:
        """
        relations = self.get_head_relations(node_type)
        return len(relations) + 1

    def save_relation_weights(self, betas: Dict[str, torch.Tensor], global_node_idx):
        # Only save relation weights if beta has weights for all node_types in the global_node_idx batch
        self._betas = {}
        self._beta_avg = {}
        self._beta_std = {}

        with torch.no_grad():
            for ntype in global_node_idx:
                relations = self.get_head_relations(ntype, to_str=True) + [ntype, ]

                self._betas[ntype] = pd.DataFrame(betas[ntype].squeeze(-1).cpu().numpy(),
                                                  columns=relations,
                                                  index=global_node_idx[ntype].cpu().numpy())

                _beta_avg = np.around(betas[ntype].mean(dim=0).squeeze(-1).cpu().numpy(), decimals=3)
                _beta_std = np.around(betas[ntype].std(dim=0).squeeze(-1).cpu().numpy(), decimals=2)
                self._beta_avg[ntype] = {metapath: _beta_avg[i] for i, metapath in
                                         enumerate(relations)}
                self._beta_std[ntype] = {metapath: _beta_std[i] for i, metapath in
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


