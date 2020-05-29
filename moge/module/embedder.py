from argparse import ArgumentParser
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax


class GAT(nn.Module):
    def __init__(self, hparams) -> None:
        super(GAT, self).__init__()

        self.embedder = GATConv(
            in_channels=hparams.encoding_dim,
            out_channels=int(hparams.embedding_dim / hparams.nb_attn_heads),
            heads=hparams.nb_attn_heads,
            concat=True,
            dropout=hparams.nb_attn_dropout
        )
        # self.batchnorm = torch.nn.BatchNorm1d(hparams.embedding_dim)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument('--embedding_dim', type=int, default=128)
        parser.add_argument('--nb_attn_heads', type=int, default=4)
        parser.add_argument('--nb_attn_dropout', type=float, default=0.5)
        return parser

    def forward(self, encodings, subnetwork):
        # print("subnetwork", subnetwork.shape)
        embeddings = self.embedder(encodings, subnetwork)
        # embeddings = self.batchnorm(embeddings)
        return embeddings


class GCN(nn.Module):
    def __init__(self, hparams) -> None:
        super(GCN, self).__init__()

        self.embedder = GCNConv(
            in_channels=hparams.encoding_dim,
            out_channels=hparams.embedding_dim,
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument('--embedding_dim', type=int, default=128)
        return parser

    def forward(self, encodings, subnetwork):
        return self.embedder(encodings, subnetwork)


class GraphSAGE(nn.Module):
    def __init__(self, hparams) -> None:
        super(GraphSAGE, self).__init__()

        self.embedder = SAGEConv(
            in_channels=hparams.encoding_dim,
            out_channels=hparams.embedding_dim,
            concat=True,
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument('--embedding_dim', type=int, default=128)
        return parser

    def forward(self, encodings, subnetwork):
        return self.embedder(encodings, subnetwork)


class MultiplexLayerAttention(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, layers, attention_dropout=0.0, bias=True):
        super(MultiplexLayerAttention, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.layers = layers

        self.weight = nn.Parameter(torch.Tensor(embedding_dim, hidden_dim))
        self.att = nn.Parameter(torch.Tensor(1, hidden_dim))
        self.dropout = nn.Dropout(attention_dropout)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(hidden_dim))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.att)
        zeros(self.bias)

    def forward(self, embeddings):
        w = self.compute_attention(embeddings)
        z = torch.matmul(torch.stack(embeddings, 2), w)
        return z

    def compute_attention(self, embeddings):
        assert len(embeddings) == len(self.layers)
        w = torch.zeros((len(self.layers), 1)).type_as(self.att)

        for i, layer in enumerate(self.layers):
            x = torch.tanh(torch.matmul(embeddings[i], self.weight) + self.bias)
            x = self.dropout(x)
            w[i] = torch.mean(torch.matmul(x, self.att.t()), dim=0)

        w = torch.softmax(w, 0)
        return w.squeeze(-1)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument('--embedding_dim', type=int, default=128)
        return parser


class MultiplexNodeAttention(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, layers, attention_dropout=0.0, bias=True):
        super(MultiplexNodeAttention, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.layers = layers

        self.weight = nn.Parameter(torch.Tensor(embedding_dim, hidden_dim))
        # self.att_weight = nn.Parameter(torch.Tensor(embedding_dim, hidden_dim))
        self.att = nn.Parameter(torch.Tensor(1, hidden_dim))
        self.dropout = nn.Dropout(attention_dropout)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(hidden_dim))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        # glorot(self.att_weight)
        glorot(self.weight)
        glorot(self.att)
        zeros(self.bias)

    def forward(self, embeddings):
        w = self.compute_attention(embeddings)

        # for i, layer in enumerate(self.layers):
        #     embeddings[i] = torch.matmul(embeddings[i], self.weight)
        z = torch.matmul(torch.stack(embeddings, dim=2), w)
        z = z.squeeze(2)
        return z

    def compute_attention(self, embeddings):
        assert len(embeddings) == len(self.layers)
        batch_size, in_channels = embeddings[0].size()
        w = torch.zeros((batch_size, len(self.layers), 1), requires_grad=False).type_as(self.weight)

        for i, layer in enumerate(self.layers):
            x = torch.tanh(torch.matmul(embeddings[i], self.weight) + self.bias)
            x = self.dropout(x)
            w[:, i] = torch.matmul(x, self.att.t())

        w = torch.softmax(w, 1)
        return w

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument('--embedding_dim', type=int, default=128)
        return parser


class HeterogeneousMultiplexAttentionEmbedding(MessagePassing):
    def __init__(self, in_channels, out_channels, node_types: [], layers: [], heads=1, concat=False,
                 negative_slope=0.2, dropout=0, bias=True, **kwargs):
        super(HeterogeneousMultiplexAttentionEmbedding, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.node_types = node_types
        self.layers = layers
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.weight = Parameter(torch.Tensor(len(node_types), in_channels, heads * out_channels))
        self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(len(layers) * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.att)
        zeros(self.bias)

    def convert_edge_index_multiplex(self, sample_idx_by_type: dict, edge_index: dict):
        for layer in self.layers:
            if edge_index[layer].size(1) == 0: continue
            nodetype_1 = layer.split("-")[0] + "_seqs"
            nodetype_2 = layer.split("-")[1] + "_seqs"
            # Shift layer edges to the right index based on node type
            edge_index[layer][0] = edge_index[layer][0] + sample_idx_by_type[nodetype_1]
            edge_index[layer][1] = edge_index[layer][1] + sample_idx_by_type[nodetype_2]

        edge_index = torch.cat([edge_index[layer] for layer in self.layers], dim=1)

        return edge_index

    def forward(self, x: dict, sample_idx_by_type: dict, edge_index: dict, size=None):
        encodings = [torch.matmul(x[node_type], self.weight[i, :, :].squeeze(0)) for i, node_type in
                     enumerate(self.node_types)]
        # print("encodings", [encoding.shape for encoding in encodings])
        # print("sample_idx_by_type", sample_idx_by_type)
        x = torch.cat(encodings)
        # print("x concat", x.shape)

        # print("edge_index", [edges.size() for layer, edges in edge_index.items()])
        edge_index = self.convert_edge_index_multiplex(sample_idx_by_type, edge_index)

        if torch.is_tensor(x):
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index,
                                           num_nodes=x.size(self.node_dim))
        # print("edge_index", edge_index.shape, ", max:", torch.max(edge_index), "\n", edge_index)

        return self.propagate(edge_index, size=size, x=x)

    def message(self, edge_index_i, x_i, x_j, size_i):
        # Compute attention coefficients.
        x_j = x_j.view(-1, self.heads, self.out_channels)
        if x_i is None:
            alpha = (x_j * self.att[:, :, self.out_channels:]).sum(dim=-1)
        else:
            x_i = x_i.view(-1, self.heads, self.out_channels)
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, size_i)

        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.layers * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.layers)
