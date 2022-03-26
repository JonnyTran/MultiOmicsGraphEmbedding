import torch
import dgl
from dgl.heterograph import DGLBlock
from dgl.nn import pytorch as dglnn
from dgl.udf import EdgeBatch, NodeBatch
from dgl.utils import expand_as_pair
from torch import nn
from torch.nn import functional as F
from dgl.nn.pytorch import GATConv
from typing import List


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


class HANLayer(torch.nn.Module):
    """
    HAN layer.
    Arguments
    ---------
    num_metapath : number of metapath based sub-graph
    in_size : input feature dimension
    out_size : output feature dimension
    layer_num_heads : number of attention heads
    dropout : Dropout probability
    Inputs
    ------
    g : DGLHeteroGraph
        The heterogeneous graph
    h : tensor
        Input features
    Outputs
    -------
    tensor
        The output feature
    """

    def __init__(self, num_metapath, in_size, out_size, layer_num_heads, dropout):
        super(HANLayer, self).__init__()

        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        for i in range(num_metapath):
            self.gat_layers.append(GATConv(in_size, out_size, layer_num_heads,
                                           dropout, dropout, activation=F.elu,
                                           allow_zero_in_degree=True))
        self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)
        self.num_metapath = num_metapath

    def forward(self, block_list, h_list):
        semantic_embeddings = []

        for i, block in enumerate(block_list):
            semantic_embeddings.append(self.gat_layers[i](block, h_list[i]).flatten(1))
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)  # (N, M, D * K)

        return self.semantic_attention(semantic_embeddings)  # (N, D * K)

class HAN(nn.Module):
    def __init__(self, num_metapath, in_size, hidden_size, out_size, num_heads, dropout):
        super(HAN, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(HANLayer(num_metapath, in_size, hidden_size, num_heads[0], dropout))
        for l in range(1, len(num_heads)):
            self.layers.append(HANLayer(num_metapath, hidden_size * num_heads[l - 1],
                                        hidden_size, num_heads[l], dropout))
        self.predict = nn.Linear(hidden_size * num_heads[-1], out_size)

    def forward(self, g, h):
        for gnn in self.layers:
            h = gnn(g, h)

        return self.predict(h)
