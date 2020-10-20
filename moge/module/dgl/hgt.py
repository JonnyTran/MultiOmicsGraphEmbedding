import dgl
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn


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

    def edge_attention(self, edges):
        etype = edges.data['id'][0]

        '''
            Step 1: Heterogeneous Mutual Attention
        '''
        relation_att = self.relation_att[etype]
        relation_pri = self.relation_pri[etype]
        key = torch.bmm(edges.src['k'].transpose(1, 0), relation_att).transpose(1, 0)
        att = (edges.dst['q'] * key).sum(dim=-1) * relation_pri / self.sqrt_dk

        '''
            Step 2: Heterogeneous Message Passing
        '''
        relation_msg = self.relation_msg[etype]
        val = torch.bmm(edges.src['v'].transpose(1, 0), relation_msg).transpose(1, 0)
        return {'a': att, 'v': val}

    def message_func(self, edges):
        return {'v': edges.data['v'], 'a': edges.data['a']}

    def reduce_func(self, nodes):
        '''
            Softmax based on target node's id (edge_index_i).
            NOTE: Using DGL's API, there is a minor difference with this softmax with the original one.
                  This implementation will do softmax only on edges belong to the same relation type, instead of for all of the edges.
        '''
        att = F.softmax(nodes.mailbox['a'], dim=1)
        h = torch.sum(att.unsqueeze(dim=-1) * nodes.mailbox['v'], dim=1)
        return {'t': h.view(-1, self.out_dim)}

    def forward(self, G, h):
        with G.local_scope():
            node_dict, edge_dict = self.node_dict, self.edge_dict
            for srctype, etype, dsttype in G.canonical_etypes:
                k_linear = self.k_linears[node_dict[srctype]]
                v_linear = self.v_linears[node_dict[srctype]]
                q_linear = self.q_linears[node_dict[dsttype]]

                G.nodes[srctype].data['k'] = k_linear(h[srctype]).view(-1, self.n_heads, self.d_k)
                G.nodes[srctype].data['v'] = v_linear(h[srctype]).view(-1, self.n_heads, self.d_k)
                G.nodes[dsttype].data['q'] = q_linear(h[dsttype]).view(-1, self.n_heads, self.d_k)

                G.apply_edges(func=self.edge_attention, etype=etype)
            G.multi_update_all({etype: (self.message_func, self.reduce_func) \
                                for etype in edge_dict}, cross_reducer='mean')
            new_h = {}
            for ntype in G.ntypes:
                '''
                    Step 3: Target-specific Aggregation
                    x = norm( W[node_type] * gelu( Agg(x) ) + x )
                '''
                n_id = node_dict[ntype]
                alpha = torch.sigmoid(self.skip[n_id])
                trans_out = self.dropout(self.a_linears[n_id](G.nodes[ntype].data['t']))
                trans_out = trans_out * alpha + h[ntype] * (1 - alpha)
                if self.use_norm:
                    new_h[ntype] = self.norms[n_id](trans_out)
                else:
                    new_h[ntype] = trans_out
            return new_h


class HGT(nn.Module):
    def __init__(self, node_dict, edge_dict, n_inp, n_hid, n_out, n_layers, n_heads, use_norm=True):
        super(HGT, self).__init__()
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.gcs = nn.ModuleList()
        self.n_inp = n_inp
        self.n_hid = n_hid
        self.n_out = n_out
        self.n_layers = n_layers
        self.adapt_ws = nn.ModuleList()
        for t in range(len(node_dict)):
            self.adapt_ws.append(nn.Linear(n_inp, n_hid))
        for _ in range(n_layers):
            self.gcs.append(HGTLayer(n_hid, n_hid, node_dict, edge_dict, n_heads, use_norm=use_norm))

    def forward(self, G):
        h = {}
        for ntype in G.ntypes:
            n_id = self.node_dict[ntype]
            h[ntype] = F.gelu(self.adapt_ws[n_id].forward(G.nodes[ntype].data['feat']))
        for i in range(self.n_layers):
            h = self.gcs[i].forward(G, h)

        return h
