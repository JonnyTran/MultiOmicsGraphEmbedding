import copy
from typing import Union, Dict, List, Tuple

import torch
import torch.nn.functional as F
from dgl.heterograph import DGLBlock
from dgl.udf import EdgeBatch, NodeBatch
from dgl.utils import expand_as_pair
from torch import nn as nn, Tensor

from moge.model.PyG.utils import filter_metapaths, max_num_hops, join_metapaths
from moge.model.dgl.utils import ChainMetaPaths
from moge.model.relations import RelationAttention, MetapathGATConv


class LATTEConv(nn.Module, RelationAttention):
    def __init__(self, in_dim, out_dim, num_nodes_dict: Dict, metapaths: List[Tuple[str, str, str]], layer: int,
                 t_order: int, batchnorm=False, layernorm=True, edge_dir="in", activation: str = "relu", dropout=0.2,
                 attn_heads=4, attn_activation="LeakyReLU", attn_dropout=0.2) -> None:
        super().__init__()
        self.node_types = list(num_nodes_dict.keys())
        self.layer = layer
        self.t_order = t_order
        self.edge_dir = edge_dir
        print(f"LATTE {self.layer + 1}, metapaths {len(metapaths)}, max_order {max_num_hops(metapaths)}")

        self.metapaths = metapaths
        self.etypes = [(metapath[0], ".".join(metapath[1::2]), metapath[-1]) for metapath in metapaths]
        self.etype_id = {".".join(metapath[1::2]): i for i, metapath in enumerate(self.etypes)}
        self.num_nodes_dict = num_nodes_dict
        self.embedding_dim = out_dim
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
            {node_type: nn.Linear(in_dim, out_dim, bias=True) \
             for node_type in self.node_types})  # W.shape (F x D_m)
        self.linear_r = nn.ModuleDict(
            {node_type: nn.Linear(in_dim, out_dim, bias=True) \
             for node_type in self.node_types})  # W.shape (F x D_m)

        self.out_channels = self.embedding_dim // attn_heads
        self.attn_l = nn.Parameter(torch.ones(len(self.etypes), attn_heads, self.out_channels))
        self.attn_r = nn.Parameter(torch.ones(len(self.etypes), attn_heads, self.out_channels))

        # self.rel_attn_l = nn.ParameterDict({
        #     ntype: nn.Parameter(torch.ones(attn_heads, self.out_channels)) \
        #     for ntype in self.node_types})
        # self.rel_attn_r = nn.ParameterDict({
        #     ntype: nn.Parameter(torch.ones(attn_heads, self.out_channels)) \
        #     for ntype in self.node_types})

        self.relation_conv: Dict[str, MetapathGATConv] = nn.ParameterDict({
            ntype: MetapathGATConv(out_dim, n_layers=1,
                                   metapaths=self.get_tail_relations(ntype),
                                   attn_heads=attn_heads, attn_dropout=attn_dropout) \
            for ntype in self.node_types})

        if layernorm:
            self.layernorm = nn.ModuleDict({
                ntype: nn.LayerNorm(out_dim) for ntype in self.node_types})

        if batchnorm:
            self.batchnorm = torch.nn.ModuleDict({
                ntype: torch.nn.BatchNorm1d(out_dim) for ntype in self.node_types})

        if attn_activation == "sharpening":
            self.alpha_activation = nn.Parameter(torch.ones(len(self.etypes)))
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
            nn.init.xavier_uniform_(self.rel_attn_l[ntype]) if hasattr(self, "rel_attn_l") else None
            nn.init.xavier_uniform_(self.rel_attn_r[ntype]) if hasattr(self, "rel_attn_r") else None

            nn.init.xavier_uniform_(self.linear_l[ntype].weight) if hasattr(self, "linear_l") else None
            nn.init.xavier_uniform_(self.linear_r[ntype].weight) if hasattr(self, "linear_r") else None

    def edge_attention(self, edges: EdgeBatch):
        srctype, etype, dsttype = edges.canonical_etype

        att_l = (edges.src["k"] * self.attn_l[self.etype_id[etype]]).sum(dim=-1)
        att_r = (edges.dst["v"] * self.attn_r[self.etype_id[etype]]).sum(dim=-1)

        att = att_l + att_r
        att = F.dropout(att, p=self.attn_dropout, training=self.training)
        att = self.alpha_activation(att) if isinstance(self.alpha_activation, nn.Module) else \
            att * self.alpha_activation[self.etype_id[etype]]

        return {etype: att, "h": edges.dst["v"]}

    def message_func(self, edges: EdgeBatch):
        srctype, etype, dsttype = edges.canonical_etype

        return {etype: edges.data[etype], "h": edges.data["h"]}

    def reduce_func(self, nodes: NodeBatch):
        '''
            Softmax based on target node's id (edge_index_i).
        '''
        output = {}
        for srctype, etype, dsttype in self.etypes:
            if etype in nodes.mailbox:
                att = F.softmax(nodes.mailbox[etype], dim=1)
                h = torch.sum(att.unsqueeze(dim=-1) * nodes.mailbox['h'], dim=1)
                output[etype] = h
            else:
                output[etype] = torch.zeros(nodes.batch_size(), self.attn_heads, self.out_channels,
                                            device=self.linear_l[srctype].weight.device,
                                            dtype=self.linear_l[srctype].weight.dtype)

        return output

    def get_beta_weights(self, query: Tensor, key: Tensor, ntype: str):
        beta_l = F.relu(query * self.rel_attn_l[ntype], )
        beta_r = F.relu(key * self.rel_attn_r[ntype], )

        beta = (beta_l[:, None, :, :] * beta_r).sum(-1)
        beta = F.softmax(beta, dim=1)
        # beta = F.dropout(beta, p=self.attn_dropout, training=self.training)
        return beta

    def forward(self, g: DGLBlock, feat: Dict[str, Tensor], save_betas=False, verbose=False):
        feat_src, feat_dst = expand_as_pair(input_=feat, g=g)

        funcs = {}
        for srctype in set(srctype for srctype, etype, dsttype in g.canonical_etypes):
            g.srcnodes[srctype].data['k'] = self.linear_l[srctype](feat_src[srctype]) \
                .view(-1, self.attn_heads, self.out_channels)

        for dsttype in set(dsttype for srctype, etype, dsttype in g.canonical_etypes):
            g.dstnodes[dsttype].data['v'] = self.linear_r[dsttype](feat_dst[dsttype]) \
                .view(-1, self.attn_heads, self.out_channels)

        for srctype, etype, dsttype in g.canonical_etypes:
            if (srctype, etype, dsttype) not in self.etypes: continue
            # Compute node-level attention coefficients
            g.apply_edges(func=self.edge_attention, etype=etype)

            # if g.batch_num_edges(etype=etype).nelement() > 1 or g.batch_num_edges(etype=etype).item() > 0:
            funcs[etype] = (self.message_func, self.reduce_func)

        g.multi_update_all(funcs, cross_reducer='mean')

        # For each metapath in a node_type, use GAT message passing to aggregate h_j neighbors
        betas = {}
        out = {}
        for ntype in set(g.ntypes):
            etypes = self.get_tail_etypes(ntype, etype_only=True)

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
            out[ntype], betas[ntype] = self.relation_conv[ntype].forward(
                out[ntype].view(out[ntype].size(0), self.num_tail_relations(ntype), self.embedding_dim))
            # beta[ntype] = self.get_beta_weights(query=out[ntype][:, -1, :], key=out[ntype], ntype=ntype)
            # out[ntype] = (out[ntype] * beta[ntype].unsqueeze(-1)).sum(1)
            # out[ntype] = out[ntype].view(out[ntype].size(0), self.embedding_dim)

            if verbose:
                global_node_index = {ntype: nid[:betas[ntype].size(0)] \
                                     for ntype, nid in g.ndata["_ID"].items() if ntype in betas}
                num_edges = {metapath: g.num_edges(etype=metapath) for metapath in g.canonical_etypes}

                print("  >", ntype, global_node_index[ntype].shape, )
                for i, (etype, beta_mean, beta_std) in enumerate(zip(self.get_tail_relations(ntype) + [ntype],
                                                                     betas[ntype].mean(-1).mean(0),
                                                                     betas[ntype].mean(-1).std(0))):
                    print(f"   - {'.'.join(etype[1::2]) if isinstance(etype, tuple) else etype}, "
                          f"\tedge_index: {num_edges[etype] if etype in num_edges else None}, "
                          f"\tbeta: {beta_mean.item():.2f} ± {beta_std.item():.2f}, "
                          f"\tnorm: {torch.norm(out[ntype][:, i]).item():.2f}")

            if hasattr(self, "activation"):
                out[ntype] = self.activation(out[ntype])

            if hasattr(self, "dropout"):
                out[ntype] = F.dropout(out[ntype], p=self.dropout, training=self.training)

            if hasattr(self, "layernorm"):
                out[ntype] = self.layernorm[ntype](out[ntype])

        if save_betas:
            beta_mean = {ntype: betas[ntype].mean(2) for ntype in betas}
            global_node_index = {ntype: nid[:beta_mean[ntype].size(0)] \
                                 for ntype, nid in g.ndata["_ID"].items() if ntype in beta_mean}
            self.save_relation_weights(beta_mean, global_node_index)

        return out

    def get_tail_etypes(self, head_node_type, str_form=False, etype_only=False) -> List[Tuple[str, str, str]]:
        relations = [metapath for metapath in self.etypes if metapath[-1] == head_node_type]

        if str_form:
            relations = [".".join(metapath) if isinstance(metapath, tuple) else metapath \
                         for metapath in relations]
        if etype_only:
            relations = [metapath[1] if isinstance(metapath, tuple) else metapath \
                         for metapath in relations]

        return relations

    def num_tail_relations(self, node_type) -> int:
        """
        Return the number of metapaths with head node type equals to :param node_type: and plus one for none-selection.
        :param node_type (str):
        :return:
        """
        relations = self.get_tail_etypes(node_type)
        return len(relations) + 1


class LATTE(nn.Module):
    def __init__(self, n_layers, t_order: int, embedding_dim: int, num_nodes_dict: Dict[str, int],
                 head_node_type: Union[str, List[str]], metapaths: List[Tuple[str, str, str]], batchnorm=False,
                 layernorm=True, edge_dir="in", activation: str = "relu", dropout=0.2, attn_heads=2,
                 attn_activation="sharpening", attn_dropout=0.2):
        super().__init__()
        self.n_layers = n_layers
        self.t_order = t_order
        self.edge_dir = edge_dir
        self.node_types = list(num_nodes_dict.keys())
        self.metapaths = metapaths
        self.head_node_type = head_node_type

        layer_t_orders = {
            l: list(range(1, t_order - (n_layers - (l + 1)) + 1)) \
                if (t_order - (n_layers - (l + 1))) > 0 \
                else [1] \
            for l in reversed(range(n_layers))}

        layers = []
        higher_order_metapaths = copy.deepcopy(metapaths)
        for l in range(n_layers):
            is_last_layer = (l + 1 == n_layers)
            l_layer_metapaths = filter_metapaths(metapaths + higher_order_metapaths,
                                                 order=layer_t_orders[l],  # Select only up to t-order
                                                 tail_type=head_node_type if is_last_layer and head_node_type else None
                                                 )

            layers.append(LATTEConv(in_dim=embedding_dim, out_dim=embedding_dim, num_nodes_dict=num_nodes_dict,
                                    metapaths=l_layer_metapaths, layer=l, t_order=t_order, batchnorm=batchnorm,
                                    layernorm=layernorm, edge_dir=edge_dir, activation=activation, dropout=dropout,
                                    attn_heads=attn_heads, attn_activation=attn_activation, attn_dropout=attn_dropout))

            higher_order_metapaths = join_metapaths(l_layer_metapaths, metapaths)

        self.layers: List[LATTEConv] = nn.ModuleList(layers)

    def forward(self, blocks: Union[DGLBlock, List[str]], h_dict: Dict[str, Tensor], **kwargs):
        block = blocks[0] if isinstance(blocks, (tuple, list)) else blocks
        for t in range(self.n_layers):
            last_layer = (t + 1 == self.n_layers)

            if t + 1 > 1 and t + 1 <= self.t_order:
                transform = ChainMetaPaths(
                    join_metapaths(block.canonical_etypes, blocks[t].canonical_etypes, return_dict=True,
                                   tail_types=self.head_node_type if last_layer and self.head_node_type else None),
                    keep_orig_edges=True)
                block = transform(block, blocks[t])

            h_new = self.layers[t].forward(block, h_dict, **kwargs)
            for ntype in h_new:
                h_dict[ntype] = torch.cat([h_new[ntype], h_dict[ntype][h_new[ntype].size(0):]], dim=0)

        dst_block = blocks[-1] if isinstance(blocks, (tuple, list)) else blocks
        h_dict = {ntype: emb[:dst_block.num_dst_nodes(ntype=ntype)] \
                  for ntype, emb in h_dict.items() if dst_block.num_dst_nodes(ntype=ntype)}

        return h_dict

    def __getitem__(self, item) -> LATTEConv:
        return self.layers[item]
