import copy
from typing import Union, Dict, List, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from dgl.heterograph import DGLBlock
from dgl.udf import EdgeBatch, NodeBatch
from dgl.utils import expand_as_pair
from moge.model.PyG.relations import RelationAttention
from moge.model.PyG.utils import filter_metapaths, max_num_hops, join_metapaths
from moge.model.dgl.utils import ChainMetaPaths
from torch import nn as nn, Tensor


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
        dup_etypes = pd.Series(self.etype_id.keys())
        dup_etypes = dup_etypes[dup_etypes.duplicated()].tolist()
        if dup_etypes:
            print(f"Duplicated etypes: {dup_etypes}")
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

        self.rel_attn_l = nn.ParameterDict({
            ntype: nn.Parameter(torch.ones(attn_heads, self.out_channels)) \
            for ntype in self.node_types})
        self.rel_attn_r = nn.ParameterDict({
            ntype: nn.Parameter(torch.ones(attn_heads, self.out_channels)) \
            for ntype in self.node_types})

        # self.relation_conv: Dict[str, MetapathGATConv] = nn.ModuleDict({
        #     ntype: MetapathGATConv(out_dim, n_layers=1,
        #                            metapaths=self.get_tail_relations(ntype),
        #                            attn_heads=attn_heads, attn_dropout=attn_dropout) \
        #     for ntype in self.node_types})

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

        att_l = edges.src["q"] * self.attn_l[self.etype_id[etype]]
        att_r = edges.dst["k"] * self.attn_r[self.etype_id[etype]]

        att = (att_l + att_r).sum(dim=-1)
        att = F.dropout(att, p=self.attn_dropout, training=self.training)
        if isinstance(self.alpha_activation, nn.Module):
            att = self.alpha_activation(att)
        else:
            att * self.alpha_activation[self.etype_id[etype]]

        return {etype: att, "h": edges.dst["k"]}

    def message_func(self, edges: EdgeBatch):
        srctype, etype, dsttype = edges.canonical_etype

        return {etype: edges.data[etype], "h": edges.data["h"]}

    def reduce_func(self, nodes: NodeBatch):
        '''
            Softmax based on target node's id (edge_index_i).
        '''
        msg = {}
        for srctype, etype, dsttype in self.etypes:
            if etype in nodes.mailbox:
                att = F.softmax(nodes.mailbox[etype], dim=1)
                h = torch.sum(att.unsqueeze(dim=-1) * nodes.mailbox['h'], dim=1)
                msg[etype] = h
            else:
                msg[etype] = torch.zeros(nodes.batch_size(), self.attn_heads, self.out_channels,
                                         requires_grad=False,
                                         device=self.linear_l[srctype].weight.device,
                                         dtype=self.linear_l[srctype].weight.dtype)

        return msg

    def get_beta_weights(self, query: Tensor, key: Tensor, ntype: str):
        beta_l = F.relu(query * self.rel_attn_l[ntype], )
        beta_r = F.relu(key * self.rel_attn_r[ntype], )

        beta = (beta_l[:, None, :, :] + beta_r).sum(-1)
        beta = F.softmax(beta, dim=1)
        # beta = F.dropout(beta, p=self.attn_dropout, training=self.training)
        return beta

    def forward(self, g: DGLBlock, feat: Dict[str, Tensor], save_betas=False, verbose=False, **kwargs):
        feat_src, feat_dst = expand_as_pair(input_=feat, g=g)
        funcs = {}
        for srctype in set(srctype for srctype, etype, dsttype in g.canonical_etypes):
            g.srcnodes[srctype].data['q'] = self.linear_l[srctype](feat_src[srctype]) \
                .view(-1, self.attn_heads, self.out_channels)

        for dsttype in set(dsttype for srctype, etype, dsttype in g.canonical_etypes):
            g.dstnodes[dsttype].data['k'] = self.linear_r[dsttype](feat_dst[dsttype]) \
                .view(-1, self.attn_heads, self.out_channels)

        for srctype, etype, dsttype in g.canonical_etypes:
            if (srctype, etype, dsttype) not in self.etypes: continue
            # Compute node-level attention coefficients
            g.apply_edges(func=self.edge_attention, etype=etype)
            funcs[etype] = (self.message_func, self.reduce_func)

        g.multi_update_all(funcs, cross_reducer="mean")

        print("\nLayer", self.layer + 1, ) if verbose else None
        # For each metapath in a node_type, use GAT message passing to aggregate h_j neighbors
        betas = {}
        h_out = {}
        for ntype in set(g.ntypes):
            etypes = self.get_tail_etypes(ntype, etype_only=True)

            # If node type doesn't have any messages
            if len(etypes) == 0:
                # h_out[ntype] = feat_dst[ntype]
                continue
            elif g.num_dst_nodes(ntype) == 0:
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
            h_out[ntype] = torch.stack(
                [g.dstnodes[ntype].data[etype] for etype in etypes] +
                [g.dstnodes[ntype].data["k"].view(-1, self.attn_heads, self.out_channels)],
                dim=1)

            if verbose and hasattr(self, 'relation_conv'):
                rel_embedding = h_out[ntype].detach().clone()

            # h_out[ntype] = h_out[ntype].view(h_out[ntype].size(0), self.num_tail_relations(ntype), self.embedding_dim)
            # h_out[ntype], betas[ntype] = self.relation_conv[ntype].forward(h_out[ntype])

            betas[ntype] = self.get_beta_weights(query=h_out[ntype][:, -1, :], key=h_out[ntype], ntype=ntype)

            if verbose:
                num_edges = {metapath: g.num_edges(etype=metapath) for metapath in g.canonical_etypes}
                rel_embedding = h_out[ntype] if h_out[ntype].dim() >= 3 else rel_embedding

                print("  >", ntype, h_out[ntype].shape, )
                for i, (etype, beta_mean, beta_std) in enumerate(zip(self.get_tail_relations(ntype) + [ntype],
                                                                     betas[ntype].mean(-1).mean(0),
                                                                     betas[ntype].mean(-1).std(0))):
                    print(f"   - {'.'.join(etype[1::2]) if isinstance(etype, tuple) else etype}: "
                          f"\tedge_index: {num_edges[etype] if etype in num_edges else 0}, "
                          f"\tbeta: {beta_mean.item():.2f} ± {beta_std.item():.2f}, "
                          f"\tnorm: {torch.norm(rel_embedding[:, i], dim=0).mean().item():.2f}")

            h_out[ntype] = (h_out[ntype] * betas[ntype].unsqueeze(-1)).sum(1)
            h_out[ntype] = h_out[ntype].view(h_out[ntype].size(0), self.embedding_dim)

            # if hasattr(self, "activation"):
            #     h_out[ntype] = self.activation(h_out[ntype])

            if hasattr(self, "dropout"):
                h_out[ntype] = F.dropout(h_out[ntype], p=self.dropout, training=self.training)

            if hasattr(self, "layernorm"):
                h_out[ntype] = self.layernorm[ntype](h_out[ntype])
            elif hasattr(self, "batchnorm"):
                h_out[ntype] = self.batchnorm[ntype](h_out[ntype])

            if verbose:
                print(f"   -> {self.activation.__name__ if hasattr(self, 'activation') else ''} "
                      f"{'dropout:' + str(self.dropout) if hasattr(self, 'dropout') else ''} "
                      f"{'batchnorm' if hasattr(self, 'batchnorm') else ''} "
                      f"{'layernorm' if hasattr(self, 'layernorm') else ''}: "
                      f"{torch.norm(h_out[ntype], dim=1).mean().item():.2f} \n")

        if save_betas:
            beta_mean = {ntype: betas[ntype].mean(2) for ntype in betas}
            global_node_index = {ntype: nid[:beta_mean[ntype].size(0)] \
                                 for ntype, nid in g.ndata["_ID"].items() if ntype in beta_mean}
            self.update_relation_attn(beta_mean, global_node_index,
                                      batch_size={ntype: emb.size(0) for ntype, emb in feat_dst.items()})

        return h_out

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
                 attn_activation="LeakyReLU", attn_dropout=0.2):
        super().__init__()
        self.n_layers = n_layers
        self.t_order = t_order
        self.edge_dir = edge_dir
        self.node_types = list(num_nodes_dict.keys())
        self.metapaths = metapaths
        self.head_node_type = head_node_type

        layers = []
        higher_order_metapaths = copy.deepcopy(metapaths)
        for l in range(n_layers):
            is_last_layer = (l + 1 == n_layers)
            l_layer_metapaths = filter_metapaths(metapaths + higher_order_metapaths,
                                                 order=list(range(1, min(l + 1, t_order) + 1)),
                                                 # Select only up to t-order
                                                 tail_type=head_node_type if is_last_layer else None)

            layers.append(LATTEConv(in_dim=embedding_dim, out_dim=embedding_dim, num_nodes_dict=num_nodes_dict,
                                    metapaths=l_layer_metapaths, layer=l, t_order=t_order, batchnorm=batchnorm,
                                    layernorm=layernorm, edge_dir=edge_dir, activation=activation, dropout=dropout,
                                    attn_heads=attn_heads, attn_activation=attn_activation, attn_dropout=attn_dropout))

            if l + 1 < t_order:
                higher_order_metapaths = join_metapaths(l_layer_metapaths, metapaths)

        self.layers: List[LATTEConv] = nn.ModuleList(layers)

    def forward(self, blocks: Union[DGLBlock, List[str]], h_dict: Dict[str, Tensor], **kwargs):
        block = blocks[0] if isinstance(blocks, (tuple, list)) else blocks
        for l in range(self.n_layers):
            last_layer = (l + 1 == self.n_layers)

            if l + 1 > 1 and l + 1 <= self.t_order:
                metapaths = join_metapaths(block.canonical_etypes, blocks[l].canonical_etypes,
                                           tail_types=self.head_node_type if last_layer and self.head_node_type else None,
                                           return_dict=True)
                metapaths = {etype: mtp_chain for etype, mtp_chain in metapaths.items() \
                             if etype in self.layers[l].etype_id.keys()}
                transform = ChainMetaPaths(metapaths, keep_orig_edges=True)
                block = transform(block, blocks[l], device=None)

            h_new = self.layers[l].forward(block, h_dict, **kwargs)
            for ntype in h_new:
                h_dict[ntype] = torch.cat([h_new[ntype], h_dict[ntype][h_new[ntype].size(0):]], dim=0)

        dst_block = blocks[-1] if isinstance(blocks, (tuple, list)) else blocks
        h_dict = {ntype: emb[:dst_block.num_dst_nodes(ntype=ntype)] \
                  for ntype, emb in h_dict.items() if dst_block.num_dst_nodes(ntype=ntype)}

        return h_dict

    def __getitem__(self, item) -> LATTEConv:
        return self.layers[item]
