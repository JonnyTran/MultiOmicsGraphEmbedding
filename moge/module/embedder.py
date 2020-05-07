import torch.nn as nn
from torch_geometric.nn import GATConv


class GAT(nn.Module):
    def __init__(self, hparams) -> None:
        super(GAT, self).__init__()

        self.gat = GATConv(
            in_channels=hparams.encoding_dim,
            out_channels=hparams.embedding_dim,
            heads=hparams.nb_attn_heads,
            concat=True,
            dropout=hparams.nb_attn_dropout
        )

    def forward(self, X):
        input_seqs, subnetwork = X["input_seqs"], X["subnetwork"]

        return self.gat(subnetwork)
