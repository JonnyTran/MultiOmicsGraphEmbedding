from typing import Tuple, List, Dict

import torch
from torch import Tensor
from torch_geometric.nn import GATConv, HGTConv, FastRGCNConv, HeteroConv

from moge.model.PyG.metapaths import get_edge_index_values


class HGT(torch.nn.Module):
    def __init__(self, embedding_dim, num_heads, num_layers, node_types: List[str],
                 metadata: Tuple[List[str], List[Tuple[str, str, str]]]):
        super().__init__()

        self.layers: List[HGTConv] = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(embedding_dim, embedding_dim, metadata, num_heads, group='sum')
            self.layers.append(conv)

    def forward(self, x_dict: Dict[str, Tensor], edge_index_dict: Dict[Tuple[str, str, str], Tensor], **kwargs):
        edge_index_dict = {etype: get_edge_index_values(tup)[0] \
                           for etype, tup in edge_index_dict.items()}

        for conv in self.layers:
            x_dict = conv(x_dict, edge_index_dict)

        return x_dict


class RGCN(torch.nn.Module):
    def __init__(self, embedding_dim, num_layers, num_relations, num_bases, num_blocks) -> None:
        super().__init__()

        self.convs: List[FastRGCNConv] = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = FastRGCNConv(in_channels=embedding_dim, out_channels=embedding_dim, num_relations=num_relations,
                                num_bases=num_bases, num_blocks=num_blocks)
            self.convs.append(conv)

    def forward(self, x_dict: Dict[str, Tensor], edge_index_dict: Dict[Tuple[str, str, str], Tensor], **kwargs):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        return x_dict


class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, metapaths: List[Tuple[str, str, str]]):
        super().__init__()

        self.convs: List[HeteroConv] = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                metapath: GATConv(hidden_channels, hidden_channels) for metapath in metapaths
            }, aggr='sum')
            self.convs.append(conv)

    def forward(self, x_dict: Dict[str, Tensor], edge_index_dict: Dict[Tuple[str, str, str], Tensor], **kwargs):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {ntype: x.relu() for ntype, x in x_dict.items()}

        return x_dict
