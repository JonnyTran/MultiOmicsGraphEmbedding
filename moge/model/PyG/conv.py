from typing import Tuple, List

import torch
from torch_geometric.nn import GATConv, HGTConv, FastRGCNConv, HeteroConv


class HGT(torch.nn.Module):
    def __init__(self, embedding_dim, num_heads, num_layers, node_types: List[str],
                 metadata: Tuple[List[str], List[Tuple[str, str, str]]]):
        super().__init__()

        # self.lin_dict = torch.nn.ModuleDict()
        # for node_type in node_types:
        #     self.lin_dict[node_type] = Linear(-1, embedding_dim)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(embedding_dim, embedding_dim, metadata,
                           num_heads, group='sum')
            self.convs.append(conv)

    def forward(self, x_dict, edge_index_dict, **kwargs):
        # x_dict = {
        #     node_type: self.lin_dict[node_type](x).relu_()
        #     for node_type, x in x_dict.items()
        # }

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        return x_dict


class RGCN(torch.nn.Module):
    def __init__(self, embedding_dim, num_layers, num_relations, num_bases, num_blocks) -> None:
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = FastRGCNConv(in_channels=embedding_dim, out_channels=embedding_dim, num_relations=num_relations,
                                num_bases=num_bases, num_blocks=num_blocks)
            self.convs.append(conv)

    def forward(self, x_dict, edge_index_dict, **kwargs):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        return x_dict


class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, metapaths: List[Tuple[str, str, str]]):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                metapath: GATConv(hidden_channels, hidden_channels) for metapath in metapaths
            }, aggr='sum')
            self.convs.append(conv)

    def forward(self, x_dict, edge_index_dict, **kwargs):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}

        return x_dict
