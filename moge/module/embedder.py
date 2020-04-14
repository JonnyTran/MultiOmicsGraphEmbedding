import torch.nn as nn


class GATEmbedder(nn.Module):
    def __init__(self, embedding_dim=128, ) -> None:
        super(GATEmbedder, self).__init__()
