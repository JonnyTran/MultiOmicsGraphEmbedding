import random
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
from ..utils import tensor_sizes, preprocess_input
from ...visualization.utils import main_colors as colors
from colorhash import ColorHash

class LATTE(nn.Module):
    def __init__(self, n_layers: int, t_order: int, embedding_dim: int, num_nodes_dict: dict,
                 metapaths: list,
                 activation: str = "relu", attn_heads=1, attn_activation="sharpening", attn_dropout=0.5,
                 use_proximity=True, neg_sampling_ratio=2.0, edge_sampling=True,
                 layer_pooling=False, hparams=None):
        super(LATTE, self).__init__()
        self.metapaths = metapaths
        self.node_types = list(num_nodes_dict.keys())
        self.head_node_type = hparams.head_node_type

        self.embedding_dim = embedding_dim * n_layers
        self.layer_pooling = layer_pooling

        self.t_order = t_order
        self.n_layers = n_layers

        self.edge_sampling = edge_sampling
        self.edge_threshold = hparams.edge_threshold
        self.use_proximity = use_proximity
        self.neg_sampling_ratio = neg_sampling_ratio

        layers = []
        higher_order_metapaths = copy.deepcopy(metapaths)  # Initialize another set of

        layer_t_orders = {
            l: list(range(1, t_order - (n_layers - (l + 1)) + 1)) if (t_order - (n_layers - (l + 1))) > 0 else [1] \
            for l in reversed(range(n_layers))}

        for l in range(n_layers):
            is_last_layer = (l + 1 == n_layers)
            is_output_layer = is_last_layer and (hparams.nb_cls_dense_size < 0)

            l_layer_metapaths = filter_metapaths(metapaths + higher_order_metapaths,
                                                 order=layer_t_orders[l],  # Select only up to t-order
                                                 # Skip higher-order relations that doesn't have the head node type, since it's the last output layer.
                                                 tail_type=self.head_node_type if is_last_layer else None)

            layers.append(
                LATTEConv(input_dim=embedding_dim,
                          output_dim=hparams.n_classes if is_output_layer else embedding_dim,
                          node_types=list(num_nodes_dict.keys()),
                          layer=l,
                          t_order=self.t_order,
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
                          edge_threshold=hparams.edge_threshold if "edge_threshold" in hparams else 0.0,
                          use_proximity=use_proximity,
                          neg_sampling_ratio=neg_sampling_ratio))

            if l + 1 < n_layers and layer_t_orders[l + 1] > layer_t_orders[l]:
                higher_order_metapaths = join_metapaths(l_layer_metapaths, metapaths)

        self.layers: List[LATTEConv] = nn.ModuleList(layers)



    def forward(self, node_feats: dict, adjs: List[Dict[Tuple, torch.Tensor]], sizes: List[Dict[str, Tuple[int]]],
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


        edge_pred_dicts = [None for l in range(self.n_layers)]
        edge_pred_dict = None

        h_in_layers = {ntype: [] for ntype in global_node_idx}
        h_out_layers = {ntype: [] for ntype in global_node_idx}
        h_out = node_feats
        for l in range(self.n_layers):
            global_node_idx = {
                ntype: global_node_idx[ntype][: sizes[l][ntype][1]] \
                for ntype in global_node_idx \
                if sizes[l][ntype][1] is not None}

            (h_in, h_out), t_loss, edge_pred_dict = self.layers[l].forward(x=h_out,
                                                                           prev_h_in=h_in_layers,
                                                                           edge_index_dict=adjs[l],
                                                                           prev_edge_index_dict=edge_pred_dict,
                                                                           sizes=sizes,
                                                                           global_node_idx=global_node_idx,
                                                                           save_betas=save_betas)

            edge_pred_dicts[l] = edge_pred_dict

            # Add the h_in embeddings to
            if l < self.n_layers and self.t_order > 1:
                for ntype in h_in:
                    h_in_layers[ntype].append(h_in[ntype])
                    if len(h_in_layers[ntype]) > self.t_order:
                        h_in_layers[ntype].pop(0)

            if self.use_proximity and t_loss is not None:
                proximity_loss += t_loss

            if self.layer_pooling != "last":
                h = h_out[self.head_node_type][:sizes[-1][self.head_node_type][1]]
                h_out_layers[self.head_node_type].append(h)

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

        return out, proximity_loss, edge_pred_dicts

    def get_attn_activation_weights(self, t):
        return dict(zip(self.layers[t].metapaths, self.layers[t].alpha_activation.detach().numpy().tolist()))

    def get_relation_weights(self, t, **kwargs):
        return self.layers[t].get_relation_weights(**kwargs)

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
    def __init__(self, input_dim: int, output_dim: int,
                 node_types: list, metapaths: list, layer: int, t_order: int,
                 activation: str = "relu", batchnorm=False, layernorm=False, dropout=0.0,
                 attn_heads=4, attn_activation="sharpening", attn_dropout=0.2,
                 edge_threshold=0.0,
                 use_proximity=False, neg_sampling_ratio=1.0) -> None:
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
                out_channels=self.attn_heads * self.num_head_relations(node_type),
                kernel_size=1) \
                for node_type in self.node_types})  # W_phi.shape (D x F)

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
        #     ntype: nn.Parameter(torch.Tensor(attn_heads, self.out_channels)) \
        #     for ntype in self.node_types})
        # self.rel_attn_r = nn.ParameterDict({
        #     ntype: nn.Parameter(torch.Tensor(attn_heads, self.out_channels)) \
        #     for ntype in self.node_types})

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
        for metapath in self.attn:
            nn.init.xavier_normal_(self.attn[metapath], gain=gain)

        gain = nn.init.calculate_gain('relu')
        for ntype in self.linear_l:
            nn.init.xavier_normal_(self.linear_l[ntype].weight, gain=gain)
            nn.init.xavier_normal_(self.linear_r[ntype].weight, gain=gain)

        # gain = nn.init.calculate_gain('leaky_relu', 0.2)
        # for ntype in self.rel_attn_l:
        #     nn.init.xavier_normal_(self.rel_attn_l[ntype], gain=gain)
        #     nn.init.xavier_normal_(self.rel_attn_r[ntype], gain=gain)

        for node_type in self.conv:
            nn.init.xavier_normal_(self.conv[node_type].weight, gain=1)

    # def get_beta_weights(self, query, key, ntype: str):
    #     alpha_l = (query * self.rel_attn_l[ntype]).sum(dim=-1)
    #     alpha_r = (key * self.rel_attn_r[ntype][:, None, :]).sum(dim=-1)
    #
    #     beta = alpha_l[:, :, None] + alpha_r
    #     beta = F.leaky_relu(beta, negative_slope=0.2)
    #     beta = F.softmax(beta, dim=2)
    #     # beta = F.dropout(beta, p=self.attn_dropout, training=self.training)
    #     return beta

    def get_beta_weights(self, h_dict, global_node_idx):
        beta = {}
        for ntype in global_node_idx:
            num_nodes = h_dict[ntype].size(0)

            beta[ntype] = self.conv[ntype].forward(h_dict[ntype].unsqueeze(-1))
            beta[ntype] = beta[ntype].view(num_nodes,
                                           self.attn_heads,
                                           self.num_head_relations(ntype))
            beta[ntype] = torch.softmax(beta[ntype], dim=2)

        return beta

    def forward(self, x: Dict[str, torch.Tensor],
                prev_h_in: Dict[str, List[torch.Tensor]],
                edge_index_dict: Dict[Tuple, torch.Tensor],
                prev_edge_index_dict: Dict[Tuple, torch.Tensor],
                sizes: List[Dict[str, Tuple[int]]],
                global_node_idx: Dict[str, torch.Tensor],
                save_betas=False):
        """

        :param x: a dict of "source" node representations
        :param prev_h_in: Context embedding of the previous order, required for t >= 2. Default: None (if first order). A dict of (node_type: tensor)
        :param global_node_idx: A dict of index values indexed by node_type in this mini-batch sampling
        :param edge_index_dict: Sparse adjacency matrices for each metapath relation. A dict of edge_index indexed by metapath

        :return: output_emb, loss
        """
        x_r = {ntype: x[ntype][: sizes[self.layer][ntype][1]] \
               for ntype in x if sizes[self.layer][ntype][1] is not None}

        l_dict = self.get_h_dict(x, source_target="source")
        r_dict = self.get_h_dict(x_r, source_target="target")

        # Predict relations attention coefficients
        beta = self.get_beta_weights(x_r, global_node_idx)

        h_out = {}
        edge_pred_dict = {}
        # For each metapath in a node_type, use GAT message passing to aggregate l_dict neighbors
        for ntype in global_node_idx:
            h_out[ntype], edge_attn_dict = self.agg_relation_neighbors(node_type=ntype,
                                                                       l_dict=l_dict,
                                                                       r_dict=r_dict,
                                                                       edge_index_dict=edge_index_dict,
                                                                       prev_l_dict=prev_h_in,
                                                                       prev_edge_index_dict=prev_edge_index_dict,
                                                                       sizes=sizes)
            h_out[ntype][:, :, -1, :] = r_dict[ntype]  # [:sizes[self.layer][ntype][1]]

            # beta[ntype] = self.get_beta_weights(query=r_dict[ntype], key=h_out[ntype], ntype=ntype)
            # Soft-select the relation-specific embeddings by a weighted average with beta[node_type]
            h_out[ntype] = h_out[ntype] * beta[ntype].unsqueeze(-1)
            h_out[ntype] = h_out[ntype].sum(2).view(h_out[ntype].size(0), self.embedding_dim)

            if hasattr(self, "layernorm"):
                h_out[ntype] = self.layernorm[ntype](h_out[ntype])

            if hasattr(self, "activation"):
                h_out[ntype] = self.activation(h_out[ntype])

            if self.dropout:
                h_out[ntype] = F.dropout(h_out[ntype], p=self.dropout, training=self.training)

            if edge_attn_dict:
                edge_pred_dict.update(edge_attn_dict)

        # Save beta weights from testing samples
        if save_betas and not self.training:
            self.save_relation_weights({ntype: beta[ntype].mean(1) for ntype in beta}, global_node_idx)

        proximity_loss = None
        if self.use_proximity:
            proximity_loss, _ = self.proximity_loss(edge_index_dict,
                                                    l_dict=l_dict, r_dict=r_dict,
                                                    global_node_idx=global_node_idx)

        return (l_dict, h_out), proximity_loss, edge_pred_dict

    def agg_relation_neighbors(self, node_type: str,
                               l_dict: Dict[str, torch.Tensor],
                               r_dict: Dict[str, torch.Tensor],
                               edge_index_dict: Dict[Tuple, Tuple[torch.Tensor]],
                               prev_l_dict: Dict[str, List[torch.Tensor]],
                               prev_edge_index_dict: Dict[Tuple, Tuple[torch.Tensor]],
                               sizes: List[Dict[str, Tuple[int]]]):
        # Initialize embeddings, size: (num_nodes, num_relations, embedding_dim)
        emb_relations = torch.zeros(
            size=(r_dict[node_type].size(0),
                  self.attn_heads,
                  self.num_head_relations(node_type),
                  self.out_channels)).type_as(r_dict[node_type])

        relations = self.get_head_relations(node_type)

        edge_pred_dict = {}
        for metapath in self.get_head_relations(node_type, order=1):
            if metapath not in edge_index_dict or edge_index_dict[metapath] is None: continue
            head, tail = metapath[0], metapath[-1]

            edge_index, values = get_edge_index_values(edge_index_dict[metapath], filter_edge=False)
            if edge_index is None: continue
            head_size_in, tail_size_out = sizes[self.layer][head][0], sizes[self.layer][tail][1]

            # Propapate flows from target nodes to source nodes
            out = self.propagate(
                edge_index=edge_index,
                x=(l_dict[head], r_dict[tail]),
                size=(head_size_in, tail_size_out),
                metapath_idx=self.metapaths.index(metapath),
                metapath=str(metapath),
                values=None)
            emb_relations[:, :, relations.index(metapath), :] = out

            edge_pred_dict[metapath] = (edge_index, self._alpha)
            self._alpha = None

        remaining_orders = range(2, min(self.layer + 1, self.t_order) + 1)
        higher_relations = self.get_head_relations(node_type, order=remaining_orders)

        higher_order_edge_index = join_edge_indexes(edge_index_dict_A=prev_edge_index_dict,
                                                    edge_index_dict_B=edge_pred_dict,
                                                    sizes=sizes, layer=self.layer,
                                                    metapaths=higher_relations,
                                                    edge_threshold=self.edge_threshold,
                                                    edge_sampling=False)
        # print("remaining_orders", list(remaining_orders), node_type,
        #       self.get_head_relations(node_type, order=remaining_orders), "edge_index", higher_order_edge_index.keys())

        for metapath in higher_relations:
            if metapath not in higher_order_edge_index or higher_order_edge_index[metapath] == None: continue
            head, tail = metapath[0], metapath[-1]

            edge_index, values = get_edge_index_values(higher_order_edge_index[metapath], filter_edge=False)
            if edge_index is None: continue

            # Select the right t-order context node presentations based on the order of the metapath
            order = len(metapath[1::2])
            h_source = prev_l_dict[head][-(order - 1)]
            head_size_in, tail_size_out = h_source.size(0), sizes[self.layer][tail][1]

            # Propapate flows from higher order source nodes to target nodes
            try:
                out = self.propagate(
                    edge_index=edge_index,
                    x=(h_source, r_dict[tail]),
                    size=(head_size_in, tail_size_out),
                    metapath_idx=self.metapaths.index(metapath),
                    metapath=str(metapath),
                    values=None)
                emb_relations[:, :, relations.index(metapath), :] = out
            except Exception as e:
                print(e)
                print(metapath, edge_index.max(1).values,
                      {"values": values.shape if values is not None else None,
                       "self._alpha": self._alpha.shape if self._alpha is not None else None},
                      {"head_size_in": head_size_in, "tail_size_out": tail_size_out},
                      {"l_dict[head]": l_dict[head].shape, "r_dict[tail]": r_dict[tail].shape})
                raise e

            edge_pred_dict[metapath] = (edge_index, self._alpha)
            self._alpha = None

        return emb_relations, edge_pred_dict

    def message(self, x_j, x_i, index, ptr, size_i, metapath_idx, metapath, values=None):
        if values is None:
            x = torch.cat([x_i, x_j], dim=2)
            if isinstance(self.alpha_activation, nn.Module):
                x = self.alpha_activation(x)
            else:
                x = self.alpha_activation[metapath_idx] * F.leaky_relu(x, negative_slope=0.2)

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

    def get_h_dict(self, input, source_target="source"):
        h_dict = {}
        for ntype in input:
            if source_target == "source":
                h_dict[ntype] = self.linear_l[ntype].forward(input[ntype])
            elif source_target == "target":
                h_dict[ntype] = self.linear_r[ntype].forward(input[ntype])

            h_dict[ntype] = F.tanh(h_dict[ntype].view(-1, self.attn_heads, self.out_channels))
            # if self.dropout:
            #     h_dict[ntype] = F.dropout(h_dict[ntype], p=self.dropout, training=self.training)

        return h_dict

    def get_head_relations(self, head_node_type, order=None, str_form=False) -> list:
        relations = filter_metapaths(self.metapaths, order=order, tail_type=head_node_type)

        if str_form:
            relations = [".".join(metapath) if isinstance(metapath, tuple) else metapath \
                         for metapath in relations]

        return relations

    def num_head_relations(self, node_type) -> int:
        """
        Return the number of metapaths with head node type equals to :param node_type: and plus one for none-selection.
        :param node_type (str):
        :return:
        """
        relations = self.get_head_relations(node_type)
        return len(relations) + 1

    def save_relation_weights(self, betas: Dict[str, torch.Tensor],
                              global_node_idx: Dict[str, torch.Tensor]):
        # Only save relation weights if beta has weights for all node_types in the global_node_idx batch
        if not hasattr(self, "_betas"): return

        with torch.no_grad():
            for ntype in global_node_idx:
                relations = self.get_head_relations(ntype, str_form=True) + [ntype, ]
                df = pd.DataFrame(betas[ntype].cpu().numpy(),
                                  columns=relations,
                                  index=global_node_idx[ntype].cpu().numpy(),
                                  dtype=np.float16)

                if len(self._betas) == 0 or ntype not in self._betas:
                    self._betas[ntype] = df
                else:
                    self._betas[ntype].update(df, overwrite=True)

    def get_relation_weights(self, std=True, std_minus=True):
        """
        Get the mean and std of relation attention weights for all nodes
        :return:
        """
        _beta_avg = {}
        _beta_std = {}
        for ntype in self._betas:
            relations = self.get_head_relations(ntype, str_form=True) + [ntype, ]
            _beta_avg = np.around(self._betas[ntype].mean(), decimals=3)
            _beta_std = np.around(self._betas[ntype].std(), decimals=2)
            self._beta_avg[ntype] = {metapath: _beta_avg[i] for i, metapath in
                                     enumerate(relations)}
            self._beta_std[ntype] = {metapath: _beta_std[i] for i, metapath in
                                     enumerate(relations)}

        print_output = {}
        for node_type in self._beta_avg:
            for metapath, avg in self._beta_avg[node_type].items():
                if std:
                    print_output[metapath] = (avg, self._beta_std[node_type][metapath])
                else:
                    print_output[metapath] = avg
        return print_output

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

    def save_attn_weights(self, node_type: str,
                          attn_weights: torch.Tensor,
                          node_idx: torch.Tensor):
        self._betas = {}
        self._beta_avg = {}
        self._beta_std = {}

        betas = attn_weights.sum(1).cpu().numpy()
        metapaths = self.get_head_relations(node_type, str_form=True) + [node_type, ]

        self._betas[node_type] = pd.DataFrame(betas,
                                              columns=metapaths,
                                              index=node_idx.cpu().numpy())

        _beta_avg = np.around(betas.mean(dim=0), decimals=3)
        _beta_std = np.around(betas.std(dim=0), decimals=2)
        self._beta_avg[node_type] = {metapath: _beta_avg[i] for i, metapath in
                                     enumerate(metapaths)}
        self._beta_std[node_type] = {metapath: _beta_std[i] for i, metapath in
                                     enumerate(metapaths)}

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
