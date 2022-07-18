import math
from typing import Dict, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.heterograph import DGLBlock
from dgl.udf import EdgeBatch, NodeBatch
from dgl.utils import expand_as_pair
from torch import Tensor


class HGTLayer(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 node_dict,
                 edge_dict,
                 n_heads,
                 dropout=0.2,
                 use_norm=False):
        super(HGTLayer, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.num_types = len(node_dict)
        self.num_relations = len(edge_dict)
        self.total_rel = self.num_types * self.num_relations * self.num_types
        self.n_heads = n_heads
        self.d_k = out_dim // n_heads
        self.sqrt_dk = math.sqrt(self.d_k)
        self.att = None

        self.k_linears = nn.ModuleList()
        self.q_linears = nn.ModuleList()
        self.v_linears = nn.ModuleList()
        self.a_linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.use_norm = use_norm

        for t in range(self.num_types):
            self.k_linears.append(nn.Linear(in_dim, out_dim))
            self.q_linears.append(nn.Linear(in_dim, out_dim))
            self.v_linears.append(nn.Linear(in_dim, out_dim))
            self.a_linears.append(nn.Linear(out_dim, out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))

        self.relation_pri = nn.Parameter(torch.ones(self.num_relations, self.n_heads))
        self.relation_att = nn.Parameter(torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k))
        self.relation_msg = nn.Parameter(torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k))
        self.skip = nn.Parameter(torch.ones(self.num_types))
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)

    def edge_attention(self, edges: EdgeBatch):
        srctype, etype, dsttype = edges.canonical_etype
        etype_id = self.edge_dict[etype]

        '''
            Step 1: Heterogeneous Mutual Attention
        '''
        relation_att = self.relation_att[etype_id]
        relation_pri = self.relation_pri[etype_id]
        key = torch.bmm(edges.src['k'].transpose(1, 0), relation_att).transpose(1, 0)
        att = (edges.dst['q'] * key).sum(dim=-1) * relation_pri / self.sqrt_dk

        '''
            Step 2: Heterogeneous Message Passing
        '''
        relation_msg = self.relation_msg[etype_id]
        val = torch.bmm(edges.src['v'].transpose(1, 0), relation_msg).transpose(1, 0)
        # print("att", att.shape, "val", val.shape)
        return {'a': att, 'v': val}

    def message_func(self, edges: EdgeBatch):
        # print("edges msg", edges.canonical_etype, edges.data['v'].device, edges.data['a'].device)
        return {'v': edges.data['v'], 'a': edges.data['a']}

    def reduce_func(self, nodes: NodeBatch):
        '''
            Softmax based on target node's id (edge_index_i).
            NOTE: Using DGL's API, there is a minor difference with this softmax with the original one.
                  This implementation will do softmax only on edges belong to the same relation type, instead of for all of the edges.
        '''
        # print(nodes.ntype, nodes.data.keys(), nodes.mailbox)
        att = F.softmax(nodes.mailbox['a'], dim=1)
        h = torch.sum(att.unsqueeze(dim=-1) * nodes.mailbox['v'], dim=1)
        return {'t': h.view(-1, self.out_dim)}

    def apply_node_func(self, nodes: NodeBatch):
        nty_id = self.node_dict[nodes.ntype]
        alpha = torch.sigmoid(self.skip[nty_id])

        trans_out = self.dropout(self.a_linears[nty_id].forward(nodes.data['t']))
        trans_out = trans_out * alpha + nodes.data["t"] * (1 - alpha)

        if self.use_norm:
            trans_out = self.norms[nty_id](trans_out)

        return {"h": trans_out}

    def forward(self, G: DGLBlock, feat):
        feat_src, feat_dst = expand_as_pair(input_=feat, g=G)
        # print(G)
        with G.local_scope():
            for srctype in set(srctype for srctype, etype, dsttype in G.canonical_etypes):
                k_linear = self.k_linears[self.node_dict[srctype]]
                v_linear = self.v_linears[self.node_dict[srctype]]
                G.srcnodes[srctype].data['k'] = k_linear(feat_src[srctype]).view(-1, self.n_heads, self.d_k)
                G.srcnodes[srctype].data['v'] = v_linear(feat_src[srctype]).view(-1, self.n_heads, self.d_k)

            for dsttype in set(dsttype for srctype, etype, dsttype in G.canonical_etypes):
                q_linear = self.q_linears[self.node_dict[dsttype]]
                G.dstnodes[dsttype].data['q'] = q_linear(feat_dst[dsttype]).view(-1, self.n_heads, self.d_k)

            funcs = {}
            for srctype, etype, dsttype in G.canonical_etypes:
                G.apply_edges(func=self.edge_attention, etype=etype)

                if G.batch_num_edges(etype=etype).item() > 0:
                    funcs[etype] = (self.message_func, self.reduce_func)

            # print("funcs", funcs.keys())
            G.multi_update_all(funcs, cross_reducer='mean')

            new_h = {}
            for ntype in G.ntypes:
                '''
                    Step 3: Target-specific Aggregation
                    x = norm( W[node_type] * gelu( Agg(x) ) + x )
                '''
                nty_id = self.node_dict[ntype]
                alpha = torch.sigmoid(self.skip[nty_id])
                # print(ntype, G.srcnodes[ntype].data.keys(), G.dstnodes[ntype].data.keys())

                if "t" in G.dstnodes[ntype].data:
                    trans_out = self.dropout(self.a_linears[nty_id].forward(G.dstnodes[ntype].data['t']))
                else:
                    trans_out = self.dropout(feat_dst[ntype])
                trans_out = trans_out * alpha + feat_dst[ntype] * (1 - alpha)
                if self.use_norm:
                    new_h[ntype] = self.norms[nty_id](trans_out)
                else:
                    new_h[ntype] = trans_out
            return new_h


class Hgt(nn.Module):
    def __init__(self, node_dict: Dict[str, int], edge_dict: Dict[str, int],
                 n_inp: int, n_hid: int, n_out: int, n_layers: int, n_heads: int, use_norm: bool = True):
        super(Hgt, self).__init__()
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.n_inp = n_inp
        self.n_hid = n_hid
        self.n_out = n_out
        self.n_layers = n_layers

        self.linear_inp = nn.ModuleList()
        for t in range(len(node_dict)):
            self.linear_inp.append(nn.Linear(n_inp, n_hid))

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(HGTLayer(n_hid, n_hid, node_dict, edge_dict, n_heads, use_norm=use_norm))

    def forward(self, G: Union[DGLBlock, List[DGLBlock]], feat: Dict[str, Tensor]):
        h = {}
        for ntype in feat:
            n_id = self.node_dict[ntype]
            h[ntype] = F.gelu(self.linear_inp[n_id].forward(feat[ntype]))

        for i in range(self.n_layers):
            h = self.layers[i].forward(G=G[i] if isinstance(G, (list, tuple)) else G,
                                       feat=h)

        return h
