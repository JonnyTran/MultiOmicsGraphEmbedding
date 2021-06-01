import torch
from dgl.heterograph import DGLBlock
from dgl.nn import pytorch as dglnn
from dgl.udf import EdgeBatch, NodeBatch
from dgl.utils import expand_as_pair
from torch import nn
from torch.nn import functional as F


class GAT(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers, ntypes, etypes):
        super().__init__()
        self.n_layers = n_layers

        self.linear_inp = nn.ModuleDict()
        for ntype in ntypes:
            self.linear_inp[ntype] = nn.Linear(in_dim, hid_dim)

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(GATLayer(hid_dim, out_dim, ntypes, etypes))

    def forward(self, blocks, feat_dict):
        h = {}
        for ntype in feat_dict:
            h[ntype] = F.gelu(self.linear_inp[ntype].forward(feat_dict[ntype]))

        for i in range(self.n_layers):
            h = self.layers[i].forward(blocks[i], h)
            # print(f"layer {i}", tensor_sizes(h))

        return h


class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, ntypes, etypes):
        super(GATLayer, self).__init__()
        self.ntypes = ntypes
        self.etypes = etypes

        self.W = nn.ModuleDict({
            ntype: nn.Linear(in_dim, out_dim, bias=False) for ntype in ntypes
        })

        self.attn = nn.ModuleDict({
            etype: nn.Linear(2 * out_dim, 1, bias=False) for etype in etypes
        })

        self.dropout = nn.Dropout(p=0.4)

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        for ntype in self.W.keys():
            nn.init.xavier_normal_(self.W[ntype].weight, gain=gain)

        for etype in self.attn.keys():
            nn.init.xavier_normal_(self.attn[etype].weight, gain=gain)

    def edge_attention(self, edges: EdgeBatch):
        srctype, etype, dsttype = edges.canonical_etype
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)

        a = self.attn[etype].forward(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges: EdgeBatch):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes: NodeBatch):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, g: DGLBlock, input_dict: dict):
        feat_dict = {ntype: self.W[ntype](ndata) for ntype, ndata in input_dict.items()}
        feat_src, feat_dst = expand_as_pair(input_=feat_dict, g=g)

        with g.local_scope():
            # print(g)
            for ntype in feat_dict:
                # print("ntype", "srcdata", g.srcnodes[ntype].data.keys(), "dstdata", g.srcnodes[ntype].data.keys())
                g.srcnodes[ntype].data['z'] = feat_src[ntype]
                g.dstnodes[ntype].data['z'] = feat_dst[ntype]

            funcs = {}
            for etype in self.etypes:
                if g.batch_num_edges(etype=etype).item() > 0:
                    g.apply_edges(self.edge_attention, etype=etype)
                    funcs[etype] = (self.message_func, self.reduce_func)

            g.multi_update_all(funcs, cross_reducer="mean")

            new_h = {}
            for ntype in g.ntypes:
                if "h" in g.dstnodes[ntype].data:
                    new_h[ntype] = self.dropout(g.dstnodes[ntype].data['h'])

            return new_h


class StochasticTwoLayerRGCN(nn.Module):
    def __init__(self, in_feat, hidden_feat, out_feat, rel_names):
        super().__init__()
        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GATConv(in_feat, hidden_feat, num_heads=4)
            for rel in rel_names
        })
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GATConv(hidden_feat, out_feat, num_heads=4)
            for rel in rel_names
        })

    def forward(self, blocks, x):
        x = self.conv1(blocks[0], x)
        x = self.conv2(blocks[1], x)
        return x


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)  # (M, 1)
        beta = torch.softmax(w, dim=0)  # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)

        return (beta * z).sum(1)  # (N, D * K)
