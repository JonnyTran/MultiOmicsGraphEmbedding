from argparse import ArgumentParser

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros


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


class MultiplexAttentionEmbedding(MessagePassing):
    def __init__(self, in_channels, out_channels, layers=1, concat=True,
                 negative_slope=0.2, dropout=0, bias=True, **kwargs):
        super(GATConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = layers
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.weight = Parameter(torch.Tensor(in_channels, layers * out_channels))
        self.att = Parameter(torch.Tensor(1, layers, 2 * out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(layers * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x: dict, edge_index: dict):
        """"""
        if size is None and torch.is_tensor(x):
            for layer, edges in edge_index.items():
                edges, _ = remove_self_loops(edges)
                edges, _ = add_self_loops(edges, num_nodes=x.size(self.node_dim))
                edge_index[layer] = edges

        if torch.is_tensor(x):
            x = torch.matmul(x, self.weight)
        else:
            x = (None if x[0] is None else torch.matmul(x[0], self.weight),
                 None if x[1] is None else torch.matmul(x[1], self.weight))

        embedding = {}
        for layer, edges in edge_index.items():
            embedding[layer] = self.propagate(edges, size=size, x=x)

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
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)
