import torch.nn as nn


class GATEmbedder(nn.Module):
    def __init__(self, embedding) -> None:
        super(GATEmbedder, self).__init__()
