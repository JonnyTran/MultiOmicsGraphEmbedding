import copy
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch import nn as nn, Tensor
from torch.nn import functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

from moge.model.sampling import negative_sample
from .utils import get_edge_index_values, filter_metapaths, join_metapaths, join_edge_indexes, max_num_hops
from ..relations import RelationAttention, MetapathGATConv
from ...dataset.utils import is_negative, tag_negative_metapath, untag_negative_metapath


class LATTEConv(MessagePassing, pl.LightningModule, RelationAttention):
    def __init__(self, input_dim: int, output_dim: int, node_types: list, metapaths: list, layer: int, t_order: int,
                 activation: str = "relu", batchnorm=False, layernorm=False, dropout=0.0,
                 attn_heads=4, attn_activation="LeakyReLU", attn_dropout=0.2, edge_threshold=0.0, use_proximity=False,
                 neg_sampling_ratio=1.0, layer_pooling=None) -> None:
        super(LATTEConv, self).__init__(aggr="add", flow="source_to_target", node_dim=0)
        self.layer = layer
        self.t_order = t_order
        self.node_types = node_types
        self.metapaths = list(metapaths)
        self.embedding_dim = output_dim
        self.use_proximity = use_proximity
        self.neg_sampling_ratio = neg_sampling_ratio
        self.attn_heads = attn_heads
        self.attn_dropout = attn_dropout
        self.edge_threshold = edge_threshold
        self.layer_pooling = layer_pooling

        print(f"LATTE {layer + 1}, metapaths {len(metapaths)}, max_order {max_num_hops(metapaths)}")

        if activation == "sigmoid":
            self.activation = F.sigmoid
        elif activation == "tanh":
            self.activation = F.tanh
        elif activation == "relu":
            self.activation = F.relu
        elif activation == "prelu":
            self.activation = nn.PReLU()
        else:
            print(f"Embedding activation arg `{activation}` did not match, so uses linear activation.")

        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        if batchnorm:
            self.batchnorm = torch.nn.ModuleDict({
                node_type: nn.BatchNorm1d(
                    output_dim * self.t_order if self.layer_pooling == "order_concat" else output_dim) \
                for node_type in self.node_types})
        if layernorm:
            self.layernorm = torch.nn.ModuleDict({
                node_type: nn.LayerNorm(
                    output_dim * self.t_order if self.layer_pooling == "order_concat" else output_dim) \
                for node_type in self.node_types})

        # self.conv = torch.nn.ModuleDict(
        #     {node_type: torch.nn.Conv1d(
        #         in_channels=input_dim,
        #         out_channels=self.attn_heads * self.num_tail_relations(node_type),
        #         kernel_size=1) \
        #         for node_type in self.node_types})  # W_phi.shape (D x F)

        self.linear_l = nn.ModuleDict(
            {node_type: nn.Linear(input_dim, output_dim, bias=True) \
             for node_type in self.node_types})  # W.shape (F x F)
        self.linear_r = nn.ModuleDict(
            {node_type: nn.Linear(input_dim, output_dim, bias=True) \
             for node_type in self.node_types})  # W.shape (F x F}

        self.out_channels = self.embedding_dim // attn_heads
        self.attn = nn.ParameterDict(
            {str(metapath): nn.Parameter(torch.rand((attn_heads, self.out_channels * 2))) \
             for metapath in filter_metapaths(self.metapaths, order=None)})

        # self.rel_attn_l = nn.ParameterDict({
        #     ntype: nn.Parameter(Tensor(attn_heads, self.out_channels)) \
        #     for ntype in self.node_types})
        # self.rel_attn_r = nn.ParameterDict({
        #     ntype: nn.Parameter(Tensor(attn_heads, self.out_channels)) \
        #     for ntype in self.node_types})

        self.relation_conv: Dict[str, MetapathGATConv] = nn.ParameterDict({
            ntype: MetapathGATConv(output_dim, metapaths=self.get_tail_relations(ntype), n_layers=1,
                                   attn_heads=attn_heads, attn_dropout=attn_dropout) \
            for ntype in self.node_types})

        if attn_activation == "sharpening":
            self.alpha_activation = nn.Parameter(Tensor(len(self.metapaths)).fill_(1.0))
        elif attn_activation == "PReLU":
            self.alpha_activation = nn.PReLU(num_parameters=attn_heads, init=0.2)
        elif attn_activation == "LeakyReLU":
            self.alpha_activation = nn.LeakyReLU(negative_slope=0.2)
        else:
            print(f"WARNING: attn_activation `{attn_activation}` did not match, so used linear activation")
            self.alpha_activation = nn.LeakyReLU(negative_slope=0.2)

        self.reset_parameters()

    def reset_parameters(self):
        # gain = nn.init.calculate_gain('leaky_relu', 0.2)
        for metapath in self.attn:
            nn.init.kaiming_uniform_(self.attn[metapath], mode='fan_in', nonlinearity='leaky_relu')

        # gain = nn.init.calculate_gain('relu')
        for ntype in self.linear_l:
            nn.init.kaiming_uniform_(self.linear_l[ntype].weight, mode='fan_in', nonlinearity='relu')
            nn.init.kaiming_uniform_(self.linear_r[ntype].weight, mode='fan_in', nonlinearity='relu') if hasattr(self,
                                                                                                                 "linear_r") else None

        if hasattr(self, "rel_attn_l"):
            for ntype in self.rel_attn_l:
                nn.init.kaiming_uniform_(self.rel_attn_l[ntype], mode='fan_in', nonlinearity='leaky_relu')
                nn.init.kaiming_uniform_(self.rel_attn_r[ntype], mode='fan_in', nonlinearity='leaky_relu') if hasattr(
                    self, "rel_attn_r") else None

        if hasattr(self, "conv"):
            for node_type in self.conv:
                nn.init.kaiming_uniform_(self.conv[node_type].weight, mode='fan_in', nonlinearity='leaky_relu')

    # def get_beta_weights(self, h_dict):
    #     beta = {}
    #
    #     for ntype in h_dict:
    #         num_nodes = h_dict[ntype].size(0)
    #
    #         beta[ntype] = self.conv[ntype].forward(h_dict[ntype].unsqueeze(-1))
    #         beta[ntype] = beta[ntype].view(num_nodes,
    #                                        self.num_tail_relations(ntype),
    #                                        self.attn_heads)
    #         beta[ntype] = torch.softmax(beta[ntype], dim=1)
    #
    #     return beta

    def get_beta_weights(self, query, key, ntype: str):
        alpha_l = (query * self.rel_attn_l[ntype]).sum(dim=-1)
        alpha_r = (key * self.rel_attn_r[ntype][None, :, :]).sum(dim=-1)

        # print("alpha_l", alpha_l.shape, "alpha_r", alpha_r.shape)
        beta = alpha_l[:, None, :] + alpha_r
        beta = F.leaky_relu(beta, negative_slope=0.2)
        beta = F.softmax(beta, dim=1)
        # beta = F.dropout(beta, p=self.attn_dropout, training=self.training)
        return beta

    def forward(self, x: Dict[str, Tensor],
                prev_h_in: Dict[str, List[Tensor]],
                edge_index_dict: Dict[Tuple, Tensor],
                prev_edge_index_dict: Dict[Tuple, Tensor],
                sizes: List[Dict[str, Tuple[int]]],
                global_node_index: List[Dict[str, Tensor]],
                save_betas=False,
                verbose=False, **kwargs) -> Tuple[Tuple[Dict[str, Tensor], Dict[str, Tensor]], Dict[Tuple, Tensor]]:
        """
        Args:
            x: a dict of "source" node representations
            prev_h_in: Context embedding of the previous order, required for t >= 2.
                Default: None (if first order). A dict of (node_type: tensor)
            global_node_index: A dict of index values indexed by node_type in this mini-batch sampling
            edge_index_dict: Sparse adjacency matrices for each metapath relation. A dict of edge_index indexed by metapath

        Returns:

        """
        x_r = {ntype: x[ntype][: sizes[self.layer][ntype][1]] \
               for ntype in x if sizes[self.layer][ntype][1]}

        l_dict = {ntype: self.linear_l[ntype].forward(feat).view(feat.size(0), self.attn_heads, self.out_channels) \
                  for ntype, feat in x.items()}
        r_dict = {ntype: self.linear_r[ntype].forward(feat).view(feat.size(0), self.attn_heads, self.out_channels) \
                  for ntype, feat in x_r.items()}

        print("\nLayer", self.layer + 1, ) if verbose else None

        betas = {}
        h_out = {}
        edge_pred_dict = {}

        # For each metapath in a node_type, use GAT message passing to aggregate l_dict neighbors
        for ntype in x_r:
            h_out[ntype], edge_attn_dict = self.agg_relation_neighbors(ntype=ntype,
                                                                       l_dict=l_dict,
                                                                       r_dict=r_dict,
                                                                       edge_index_dict=edge_index_dict,
                                                                       prev_l_dict=prev_h_in,
                                                                       prev_edge_index_dict=prev_edge_index_dict,
                                                                       sizes=sizes)
            h_out[ntype][:, -1] = l_dict[ntype][:sizes[self.layer][ntype][1]]

            if verbose:
                rel_embedding = h_out[ntype].detach().clone()

            # Soft-select the relation-specific embeddings by a weighted average with beta[node_type]
            h_out[ntype], betas[ntype] = self.relation_conv[ntype].forward(
                h_out[ntype].view(h_out[ntype].size(0), self.num_tail_relations(ntype), self.embedding_dim))

            # betas[ntype] = self.get_beta_weights(query=r_dict[ntype], key=h_out[ntype], ntype=ntype)
            if verbose:
                print("  >", ntype, global_node_index[self.layer][ntype].shape, )
                for i, (etype, beta_mean, beta_std) in enumerate(zip(self.get_tail_relations(ntype) + [ntype],
                                                                     betas[ntype].mean(-1).mean(0),
                                                                     betas[ntype].mean(-1).std(0))):
                    print(f"   - {'.'.join(etype[1::2]) if isinstance(etype, tuple) else etype}, "
                          f"\tedge_index: {edge_index_dict[etype].size(1) if etype in edge_index_dict else 0}, "
                          f"\tbeta: {beta_mean.item():.2f} ± {beta_std.item():.2f}, "
                          f"\tnorm: {torch.norm(rel_embeddings[:, i]).item() if rel_embeddings.dim() >= 3 else -1:.2f}")
            #
            # h_out[ntype] = (h_out[ntype] * betas[ntype].unsqueeze(-1)).sum(1)
            # h_out[ntype] = h_out[ntype].view(h_out[ntype].size(0), self.embedding_dim)

            if hasattr(self, "activation"):
                h_out[ntype] = self.activation(h_out[ntype])

            if hasattr(self, "dropout"):
                h_out[ntype] = self.dropout(h_out[ntype])

            if hasattr(self, "layernorm"):
                h_out[ntype] = self.layernorm[ntype](h_out[ntype])


        # Save beta weights from testing samples
        if save_betas:
            beta_mean = {ntype: betas[ntype].mean(2) for ntype in betas}
            global_node_idx_out = {ntype: nid[:sizes[self.layer][ntype][1]] \
                                   for ntype, nid in global_node_index[self.layer].items() \
                                   if ntype in beta_mean and sizes[self.layer][ntype][1]}
            self.save_relation_weights(beta_mean, global_node_idx_out)

        return (l_dict, h_out), edge_pred_dict

    def message(self, x_j, x_i, index, ptr, size_i, metapath_idx, metapath, values=None):
        if values is None:
            x = torch.cat([x_i, x_j], dim=2)
            if isinstance(self.alpha_activation, Tensor):
                x = self.alpha_activation[metapath_idx] * F.leaky_relu(x, negative_slope=0.2)
            elif isinstance(self.alpha_activation, nn.Module):
                x = self.alpha_activation(x)

            alpha = (x * self.attn[metapath]).sum(dim=-1)
            alpha = softmax(alpha, index=index, ptr=ptr, num_nodes=size_i)
        else:
            if values.dim() == 1:
                values = values.unsqueeze(-1)
            alpha = values
            # alpha = softmax(alpha, index=index, ptr=ptr, num_nodes=size_i)

        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.attn_dropout, training=self.training)

        return x_j * alpha.unsqueeze(-1)

    def agg_relation_neighbors(self, ntype: str,
                               l_dict: Dict[str, Tensor],
                               r_dict: Dict[str, Tensor],
                               edge_index_dict: Dict[Tuple, Tuple[Tensor]],
                               prev_l_dict: Dict[str, List[Tensor]],
                               prev_edge_index_dict: Dict[Tuple, Tuple[Tensor]],
                               sizes: List[Dict[str, Tuple[int]]]):
        # Initialize embeddings, size: (num_nodes, num_relations, embedding_dim)
        # print(node_type, tensor_sizes(r_dict))
        emb_relations = torch.zeros(
            size=(r_dict[ntype].size(0),
                  self.num_tail_relations(ntype),
                  self.attn_heads,
                  self.out_channels)).type_as(r_dict[ntype])

        relations = self.get_tail_relations(ntype)

        edge_pred_dict = {}
        for metapath in self.get_tail_relations(ntype, order=1):
            if metapath not in edge_index_dict or edge_index_dict[metapath] is None: continue
            head_type, tail_type = metapath[0], metapath[-1]

            edge_index, values = get_edge_index_values(edge_index_dict[metapath], filter_edge=False)
            if edge_index is None: continue
            head_size_in, tail_size_out = sizes[self.layer][head_type][0], sizes[self.layer][tail_type][1]

            # pprint(tensor_sizes(metapath=metapath,
            #                     edge_index=list(
            #                         zip(["src_" + head_type, "dst_" + tail_type], edge_index.max(1).values)),
            #                     edge_values=values,
            #                     x=[l_dict[head_type], r_dict[tail_type]],
            #                     head_size_in=head_size_in, tail_size_out=tail_size_out))

            # Propapate flows from target nodes to source nodes
            out = self.propagate(
                edge_index=edge_index,
                x=(l_dict[head_type], r_dict[tail_type]),
                size=(head_size_in, tail_size_out),
                metapath_idx=self.metapaths.index(metapath),
                metapath=str(metapath),
                values=None)
            emb_relations[:, relations.index(metapath)] = out
            edge_pred_dict[metapath] = (edge_index, self._alpha)
            self._alpha = None

        remaining_orders = list(range(2, min(self.layer + 1, self.t_order) + 1))
        higher_relations = self.get_tail_relations(ntype, order=remaining_orders)

        higher_order_edge_index = join_edge_indexes(edge_index_dict_A=prev_edge_index_dict,
                                                    edge_index_dict_B=edge_pred_dict,
                                                    sizes=sizes, layer=self.layer,
                                                    filter_metapaths=higher_relations,
                                                    edge_threshold=self.edge_threshold)
        # print("remaining_orders", list(remaining_orders), node_type,
        #       self.get_tail_relations(node_type, order=remaining_orders), "edge_index", higher_order_edge_index.keys())

        for metapath in higher_relations:
            if metapath not in higher_order_edge_index or higher_order_edge_index[metapath] == None: continue
            head_type, tail_type = metapath[0], metapath[-1]

            edge_index, values = get_edge_index_values(higher_order_edge_index[metapath], filter_edge=False)
            if edge_index is None: continue

            # Select the right t-order context node presentations based on the order of the metapath
            order = len(metapath[1::2])
            h_source = prev_l_dict[head_type][-(order - 1)]
            head_size_in, tail_size_out = h_source.size(0), sizes[self.layer][tail_type][1]


            # Propapate flows from higher order source nodes to target nodes
            out = self.propagate(
                edge_index=edge_index,
                x=(h_source, r_dict[tail_type]),
                size=(head_size_in, tail_size_out),
                metapath_idx=self.metapaths.index(metapath),
                metapath=str(metapath),
                values=None)
            emb_relations[:, relations.index(metapath)] = out

            edge_pred_dict[metapath] = (edge_index, self._alpha)
            self._alpha = None

        return emb_relations, edge_pred_dict

    def get_top_relations(self, ntype="paper"):
        # columns: [..., 2-neighbors, 2-order relations, 1-neighbors, 1-order relations, targets]
        top_rels: pd.DataFrame = self._betas[ntype].idxmax(axis=1).str.split(".", expand=True)

        # Shift top meta relations to the right if its right value is None
        for col_idx in top_rels.columns[1::2]:
            rows2shift = top_rels[top_rels[top_rels.columns[col_idx:]].isnull().all(1)]
            rows2shift = rows2shift.shift(top_rels.shape[1] - col_idx, axis=1)
            top_rels.loc[rows2shift.index] = rows2shift

        # Rename columns
        ordinal = lambda n: str(n) + {1: 'st', 2: 'nd', 3: 'rd'}.get(4 if 10 <= n % 100 < 20 else n % 10, "th")
        columns = [f"{ordinal(int(np.ceil(i / 2)))} neighbors" \
                       if i % 2 == 0 \
                       else f"{int(np.ceil(i / 2))}-order relations"
                   for i, _ in enumerate(top_rels.columns)]
        columns[0] = "target"
        top_rels.columns = reversed(columns)

        return top_rels

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
                e_pred_logits = self.predict_scores(edge_index, l_dict, r_dict, untag_negative_metapath(metapath),
                                                    logits=True)
                loss += -torch.mean(F.logsigmoid(-e_pred_logits), dim=-1)

            # Only need to sample for negative edges if negative metapath is not included
            if not is_negative(metapath) and tag_negative_metapath(metapath) not in edge_index_dict:
                neg_edge_index = negative_sample(edge_index,
                                                 M=global_node_idx[metapath[0]].size(0),
                                                 N=global_node_idx[metapath[-1]].size(0),
                                                 n_sample_per_edge=self.neg_sampling_ratio)
                if neg_edge_index is None or neg_edge_index.size(1) <= 1: continue

                e_neg_logits = self.predict_scores(neg_edge_index, l_dict, r_dict, metapath, logits=True)
                loss += -torch.mean(F.logsigmoid(-e_neg_logits), dim=-1)

        loss = torch.true_divide(loss, max(len(edge_index_dict) * 2, 1))
        return loss

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


class LATTE(nn.Module):
    def __init__(self, n_layers: int, t_order: int, embedding_dim: int, num_nodes_dict: dict, metapaths: list,
                 activation: str = "relu", attn_heads=1, attn_activation="sharpening", attn_dropout=0.5,
                 layer_pooling=False, use_proximity=True, neg_sampling_ratio=2.0, edge_sampling=True,
                 hparams=None):
        super().__init__()
        self.metapaths = metapaths
        self.node_types = list(num_nodes_dict.keys())
        self.head_node_type = hparams.head_node_type

        self.embedding_dim = embedding_dim
        self.t_order = t_order
        self.n_layers = n_layers

        self.edge_sampling = edge_sampling
        self.edge_threshold = hparams.edge_threshold if "edge_threshold" in hparams else 0.0
        self.use_proximity = use_proximity
        self.neg_sampling_ratio = neg_sampling_ratio
        self.layer_pooling = layer_pooling

        layers = []
        higher_order_metapaths = copy.deepcopy(metapaths)  # Initialize another set of

        layer_t_orders = {
            l: list(range(1, t_order - (n_layers - (l + 1)) + 1)) \
                if (t_order - (n_layers - (l + 1))) > 0 \
                else [1] \
            for l in reversed(range(n_layers))}

        for l in range(n_layers):
            is_last_layer = (l + 1 == n_layers)

            l_layer_metapaths = filter_metapaths(metapaths + higher_order_metapaths,
                                                 order=layer_t_orders[l],  # Select only up to t-order
                                                 # Skip higher-order relations that doesn't have the head node type, since it's the last output layer.
                                                 tail_type=self.head_node_type if is_last_layer else None)

            layers.append(
                LATTEConv(input_dim=embedding_dim,
                          output_dim=embedding_dim,
                          node_types=list(num_nodes_dict.keys()),
                          metapaths=l_layer_metapaths,
                          layer=l,
                          t_order=self.t_order,
                          activation=activation,
                          batchnorm=False if "batchnorm" not in hparams else hparams.batchnorm,
                          layernorm=False if "layernorm" not in hparams else hparams.layernorm,
                          dropout=False if "dropout" not in hparams else hparams.dropout,
                          attn_heads=attn_heads, attn_activation=attn_activation, attn_dropout=attn_dropout,
                          edge_threshold=hparams.edge_threshold if "edge_threshold" in hparams else 0.0,
                          use_proximity=use_proximity, neg_sampling_ratio=neg_sampling_ratio,
                          layer_pooling=layer_pooling if is_last_layer else None))

            if l + 1 < n_layers and layer_t_orders[l + 1] > layer_t_orders[l]:
                higher_order_metapaths = join_metapaths(l_layer_metapaths, metapaths)

        self.layers: List[LATTEConv] = nn.ModuleList(layers)

    def forward(self, feats: Dict[str, Tensor], edge_index_dict: List[Dict[Tuple, Tensor]],
                sizes: List[Dict[str, Tuple[int]]], global_node_index: List[Dict], **kwargs) \
            -> Tuple[Dict[str, Tensor], List[Dict[Tuple, Tensor]]]:
        """
        Args:
            feats: Dict of <node_type>:<tensor size (batch_size, in_channels)>. If nodes are not attributed, then pass an empty dict.
            global_node_index: Dict of <node_type>:<int tensor size (batch_size,)>
            edge_index_dict: Dict of <metapath>:<tensor size (2, num_edge_index)>

        Returns:
            embedding_output, edge_pred_dict:
        """
        h_out = feats
        prev_h_in = {ntype: [] for ntype in feats}

        prev_edge_index_dict = [None, ] * self.n_layers
        edge_pred_dict = None
        for l in range(self.n_layers):
            (h_in, h_out), edge_pred_dict = self.layers[l].forward(x=h_out,
                                                                   prev_h_in=prev_h_in,
                                                                   edge_index_dict=edge_index_dict[l],
                                                                   prev_edge_index_dict=edge_pred_dict,
                                                                   sizes=sizes,
                                                                   global_node_index=global_node_index,
                                                                   **kwargs)

            prev_edge_index_dict[l] = edge_pred_dict

            # Add the h_in embeddings to
            if l < self.n_layers and self.t_order > 1:
                for ntype in h_in:
                    prev_h_in[ntype].append(h_in[ntype])
                    if len(prev_h_in[ntype]) > self.t_order:
                        prev_h_in[ntype].pop(0)

        return h_out, edge_pred_dict

    def get_attn_activation_weights(self, t):
        return dict(zip(self.layers[t].metapaths, self.layers[t].alpha_activation.detach().numpy().tolist()))

    def get_relation_weights(self, t, **kwargs):
        return self.layers[t].get_relation_weights(**kwargs)

    def get_top_relations(self, t, node_type, min_order=None):
        df = self.layers[t].get_top_relations(ntype=node_type)
        if min_order:
            df = df[df.notnull().sum(1) >= min_order]
        return df

    def __getitem__(self, item) -> LATTEConv:
        return self.layers[item]
