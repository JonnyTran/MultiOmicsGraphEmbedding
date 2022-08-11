import copy
from typing import Union, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from dgl.heterograph import DGLHeteroGraph, DGLBlock
from dgl.udf import EdgeBatch, NodeBatch
from dgl.utils import expand_as_pair
from torch import nn as nn, Tensor

from moge.model.PyG.utils import filter_metapaths, max_num_hops, join_metapaths
from moge.model.dgl.utils import ChainMetaPaths


class LATTE(nn.Module):
    def __init__(self, t_order: int, embedding_dim: int, num_nodes_dict: Dict[str, int],
                 head_node_type: Union[str, List[str]], metapaths: List[Tuple[str, str, str]], batchnorm=False,
                 layernorm=True, edge_dir="in",
                 activation: str = "relu", dropout=0.2,
                 attn_heads=1, attn_activation="sharpening", attn_dropout=0.5, ):
        super().__init__()
        self.t_order = t_order
        self.edge_dir = edge_dir
        self.node_types = list(num_nodes_dict.keys())
        self.metapaths = metapaths
        self.head_node_type = head_node_type

        layers = []
        higher_order_metapaths = copy.deepcopy(metapaths)
        for l in range(t_order):
            is_last_layer = (l + 1 == t_order)
            l_layer_metapaths = filter_metapaths(metapaths + higher_order_metapaths,
                                                 # tail_type=head_node_type if is_last_layer and head_node_type else None
                                                 )

            layers.append(LATTEConv(in_dim=embedding_dim, embedding_dim=embedding_dim, num_nodes_dict=num_nodes_dict,
                                    metapaths=l_layer_metapaths, layer=l, t_order=t_order, batchnorm=batchnorm,
                                    layernorm=layernorm, edge_dir=edge_dir, activation=activation, dropout=dropout,
                                    attn_heads=attn_heads, attn_activation=attn_activation, attn_dropout=attn_dropout))

            higher_order_metapaths = join_metapaths(l_layer_metapaths, metapaths)

        self.layers: List[LATTEConv] = nn.ModuleList(layers)

    def forward(self, blocks: Union[DGLBlock, List[str]], h_dict: Dict[str, Tensor], **kwargs):
        block = blocks[0] if isinstance(blocks, (tuple, list)) else blocks
        for t in range(self.t_order):
            last_layer = (t + 1 == self.t_order)

            if t + 1 > 1:
                transform = ChainMetaPaths(join_metapaths(block.canonical_etypes, blocks[t].canonical_etypes,
                                                          return_dict=True,
                                                          tail_types=self.head_node_type if last_layer and self.head_node_type else None),
                                           keep_orig_edges=True)
                block = transform(block, blocks[t])

            h_new = self.layers[t].forward(block, h_dict, **kwargs)
            for ntype in h_new:
                h_dict[ntype][:h_new[ntype].size(0)] = h_new[ntype]

        dst_block = blocks[-1] if isinstance(blocks, (tuple, list)) else blocks
        h_dict = {ntype: emb[:dst_block.num_dst_nodes(ntype=ntype)] \
                  for ntype, emb in h_dict.items() if dst_block.num_dst_nodes(ntype=ntype)}

        return h_dict


class LATTEConv(nn.Module):
    def __init__(self, in_dim, embedding_dim, num_nodes_dict: Dict, metapaths: List[Tuple[str, str, str]], layer: int,
                 t_order: int, batchnorm=False, layernorm=True, edge_dir="in", activation: str = "relu", dropout=0.2,
                 attn_heads=4, attn_activation="LeakyReLU", attn_dropout=0.2) -> None:
        super().__init__()
        self.node_types = list(num_nodes_dict.keys())
        self.layer = layer
        self.t_order = t_order
        print(f"LATTE {self.layer + 1}, metapaths {len(metapaths)}, max_order {max_num_hops(metapaths)}")

        self.edge_dir = edge_dir

        self.metapaths = [(metapath[0], ".".join(metapath[1::2]), metapath[-1]) for metapath in metapaths]
        self.metapath_id = {".".join(metapath[1::2]): i for i, metapath in enumerate(self.metapaths)}
        self.num_nodes_dict = num_nodes_dict
        self.embedding_dim = embedding_dim
        self.attn_heads = attn_heads
        self.attn_dropout = attn_dropout
        self.dropout = dropout

        if activation == "sigmoid":
            self.activation = F.sigmoid
        elif activation == "tanh":
            self.activation = F.tanh
        elif activation == "relu":
            self.activation = F.relu
        else:
            print(f"Embedding activation arg `{activation}` did not match, so uses linear activation.")

        self.linear_l = nn.ModuleDict(
            {node_type: nn.Linear(in_dim, embedding_dim, bias=True) \
             for node_type in self.node_types})  # W.shape (F x D_m)
        self.linear_r = nn.ModuleDict(
            {node_type: nn.Linear(in_dim, embedding_dim, bias=True) \
             for node_type in self.node_types})  # W.shape (F x D_m)

        self.out_channels = self.embedding_dim // attn_heads
        self.attn_l = nn.Parameter(torch.ones(len(self.metapaths), attn_heads, self.out_channels))
        self.attn_r = nn.Parameter(torch.ones(len(self.metapaths), attn_heads, self.out_channels))

        self.rel_attn_l = nn.ParameterDict({
            ntype: nn.Parameter(torch.ones(attn_heads, self.out_channels)) \
            for ntype in self.node_types})
        self.rel_attn_r = nn.ParameterDict({
            ntype: nn.Parameter(torch.ones(attn_heads, self.out_channels)) \
            for ntype in self.node_types})

        if layernorm:
            self.layernorm = nn.ModuleDict({
                ntype: nn.LayerNorm(embedding_dim) for ntype in self.node_types})

        if batchnorm:
            self.batchnorm = torch.nn.ModuleDict({
                ntype: torch.nn.BatchNorm1d(embedding_dim) for ntype in self.node_types})

        if attn_activation == "sharpening":
            self.alpha_activation = nn.Parameter(torch.ones(len(self.metapaths)))
        elif attn_activation == "PReLU":
            self.alpha_activation = nn.PReLU(init=0.2)
        elif attn_activation == "LeakyReLU":
            self.alpha_activation = nn.LeakyReLU(negative_slope=0.2)
        else:
            print(f"WARNING: alpha_activation `{attn_activation}` did not match, so used linear activation")
            self.alpha_activation = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.attn_l)
        nn.init.xavier_uniform_(self.attn_r)

        for ntype in self.node_types:
            nn.init.xavier_uniform_(self.rel_attn_l[ntype])
            nn.init.xavier_uniform_(self.rel_attn_r[ntype])

            nn.init.xavier_uniform_(self.linear_l[ntype].weight)
            nn.init.xavier_uniform_(self.linear_r[ntype].weight)

    def edge_attention(self, edges: EdgeBatch):
        srctype, etype, dsttype = edges.canonical_etype

        att_l = (edges.src["k"] * self.attn_l[self.metapath_id[etype]]).sum(dim=-1)
        att_r = (edges.dst["v"] * self.attn_r[self.metapath_id[etype]]).sum(dim=-1)

        att = att_l + att_r
        att = F.dropout(att, p=self.attn_dropout, training=self.training)
        att = self.alpha_activation(att) if isinstance(self.alpha_activation, nn.Module) else \
            att * self.alpha_activation[self.metapath_id[etype]]

        return {etype: att, "h": edges.dst["v"]}

    def message_func(self, edges: EdgeBatch):
        srctype, etype, dsttype = edges.canonical_etype

        return {etype: edges.data[etype], "h": edges.data["h"]}

    def reduce_func(self, nodes: NodeBatch):
        '''
            Softmax based on target node's id (edge_index_i).
        '''
        output = {}
        for srctype, etype, dsttype in self.metapaths:
            if etype not in nodes.mailbox: continue

            att = F.softmax(nodes.mailbox[etype], dim=1)
            h = torch.sum(att.unsqueeze(dim=-1) * nodes.mailbox['h'], dim=1)
            output[etype] = h

        return output

    def get_beta_weights(self, key, query, ntype):
        alpha_l = (key * self.rel_attn_l[ntype]).sum(dim=-1)
        alpha_r = (query * self.rel_attn_r[ntype][None, :, :]).sum(dim=-1)

        beta = alpha_l[:, None, :] + alpha_r
        beta = F.leaky_relu(beta, negative_slope=0.2)
        beta = F.softmax(beta, dim=2)
        beta = F.dropout(beta, p=self.attn_dropout, training=self.training)
        return beta

    def forward(self, g: Union[DGLBlock, DGLHeteroGraph], feat: Dict[str, Tensor], save_betas=False):
        feat_src, feat_dst = expand_as_pair(input_=feat, g=g)

        funcs = {}
        for srctype in set(srctype for srctype, etype, dsttype in g.canonical_etypes):
            g.srcnodes[srctype].data['k'] = self.linear_l[srctype](feat_src[srctype]) \
                .view(-1, self.attn_heads, self.out_channels)

        for dsttype in set(dsttype for srctype, etype, dsttype in g.canonical_etypes):
            g.dstnodes[dsttype].data['v'] = self.linear_r[dsttype](feat_dst[dsttype]) \
                .view(-1, self.attn_heads, self.out_channels)

        for srctype, etype, dsttype in g.canonical_etypes:
            # Compute node-level attention coefficients
            g.apply_edges(func=self.edge_attention, etype=etype)

            if g.batch_num_edges(etype=etype).nelement() > 1 or g.batch_num_edges(etype=etype).item() > 0:
                funcs[etype] = (self.message_func, self.reduce_func)

        g.multi_update_all(funcs, cross_reducer='mean')

        # For each metapath in a node_type, use GAT message passing to aggregate h_j neighbors
        beta = {}
        out = {}
        for ntype in set(g.ntypes):
            etypes = [etype for etype in self.get_head_relations(ntype, etype_only=True) \
                      if etype in g.dstnodes[ntype].data]

            # If node type doesn't have any messages
            if len(etypes) == 0:
                out[ntype] = feat_dst[ntype]
                continue
            # If homogeneous graph
            # if len(g.etypes) == 1:
            #     out[ntype] = out[ntype] = torch.stack(
            #         [g.dstnodes[ntype].data[etype] for etype in etypes] + [
            #             g.dstnodes[ntype].data["feat"].view(-1, self.embedding_dim), ], dim=1)
            #     if hasattr(self, "activation"):
            #         out[ntype] = self.activation(out[ntype])
            #     continue

            # Soft-select the relation-specific embeddings by a weighted average with beta[node_type]
            out[ntype] = torch.stack([g.dstnodes[ntype].data[etype] for etype in etypes] +
                                     [g.dstnodes[ntype].data["v"].view(-1, self.attn_heads, self.out_channels)],
                                     dim=1)
            # out[ntype] = torch.mean(out[ntype], dim=1)

            beta[ntype] = self.get_beta_weights(key=out[ntype][:, -1, :], query=out[ntype], ntype=ntype)
            out[ntype] = out[ntype] * beta[ntype].unsqueeze(-1)
            out[ntype] = out[ntype].sum(1).view(out[ntype].size(0), self.embedding_dim)

            if hasattr(self, "layernorm"):
                out[ntype] = self.layernorm[ntype](out[ntype])

            if hasattr(self, "batchnorm"):
                out[ntype] = self.batchnorm[ntype](out[ntype])

            if hasattr(self, "activation"):
                out[ntype] = self.activation(out[ntype])

            if hasattr(self, "dropout"):
                out[ntype] = F.dropout(out[ntype], p=self.dropout, training=self.training)

        if save_betas and not self.training:
            beta_mean = {ntype: beta[ntype].mean(2) for ntype in beta}
            global_node_idx_out = {ntype: nid for ntype, nid in g.ndata["_ID"].items()}
            self.save_relation_weights(beta_mean, global_node_idx_out)

        return out

    def get_head_relations(self, head_node_type, to_str=False, etype_only=False) -> list:
        if self.edge_dir == "out":
            relations = [metapath for metapath in self.metapaths \
                         if metapath[0] == head_node_type]
        elif self.edge_dir == "in":
            relations = [metapath for metapath in self.metapaths \
                         if metapath[-1] == head_node_type]

        if to_str:
            relations = [".".join(metapath) if isinstance(metapath, tuple) else metapath \
                         for metapath in relations]
        if etype_only:
            relations = [metapath[1] if isinstance(metapath, tuple) else metapath \
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

    def save_relation_weights(self, betas: Dict[str, Tensor],
                              global_node_idx: Dict[str, Tensor]):
        # Only save relation weights if beta has weights for all node_types in the global_node_idx batch
        if not hasattr(self, "_betas"): return

        with torch.no_grad():
            for ntype in betas:
                if ntype not in global_node_idx or global_node_idx[ntype].numel() == 0: continue
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
