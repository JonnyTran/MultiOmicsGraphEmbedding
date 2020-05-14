from argparse import ArgumentParser

import torch.nn as nn
from torch_geometric.nn import GATConv, GCNConv, SAGEConv


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


class MultiplexGAT(nn.Module):
    def __init__(self, hparams) -> None:
        super(MultiplexGAT, self).__init__()

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
