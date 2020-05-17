from argparse import ArgumentParser

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
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

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument('--embedding_dim', type=int, default=128)
        parser.add_argument('--nb_attn_heads', type=int, default=4)
        parser.add_argument('--nb_attn_dropout', type=float, default=0.5)
        return parser

    def forward(self, encodings, subnetwork):
        return self.embedder(encodings, subnetwork)


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


class MultiplexLayerAttention(nn.MultiLabelSoftMarginLoss):
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


class MultiplexNodeAttention(nn.MultiLabelSoftMarginLoss):
    def __init__(self, embedding_dim, hidden_dim, layers, attention_dropout=0.0, bias=True):
        super(MultiplexNodeAttention, self).__init__()

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
        # print("embeddings", len(embeddings), torch.stack(embeddings, dim=2).shape)
        # print("w", w.shape)
        z = torch.matmul(torch.stack(embeddings, dim=2), w)
        # print("z", z.shape)
        z = z.squeeze(2)
        return z

    def compute_attention(self, embeddings):
        assert len(embeddings) == len(self.layers)
        batch_size, in_channels = embeddings[0].size()
        w = torch.zeros((batch_size, len(self.layers), 1)).type_as(self.att)

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
