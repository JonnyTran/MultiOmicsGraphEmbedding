import torch.nn as nn
from torch_geometric.nn import GATConv


class GAT(nn.Module):
    def __init__(self, encoding_dim=128, embedding_dim=128, nb_attn_heads=4, nb_attn_dropout=0.5) -> None:
        super(GAT, self).__init__()

        self.nb_attn_heads = nb_attn_heads
        self.nb_attn_dropout = nb_attn_dropout

        self.gat = GATConv(
            in_channels=encoding_dim,
            out_channels=embedding_dim,
            heads=self.nb_attn_heads,
            dropout=self.nb_attn_dropout
        )

    def forward(self, subnetwork):
        return self.gat(subnetwork)
