import torch

from torch_geometric.data import DataLoader, Data
from torch_geometric.utils import to_undirected
from torch_geometric.data import NeighborSampler
from torch_geometric.utils.hetero import group_hetero_graph
import torch_sparse

import dgl

from ..datasets import HeteroNetDataset


class DGLHeteroSampler(HeteroNetDataset):

    def __init__(self, dataset, node_types, metapaths=None, head_node_type=None, directed=True, train_ratio=0.7,
                 add_reverse_metapaths=True, process_graphs=True):
        super().__init__(dataset, node_types, metapaths, head_node_type, directed, train_ratio, add_reverse_metapaths,
                         process_graphs)

    def process_graph_sampler(self):
        dgl_edge_index = {}
        for metapath, edge_index in self.edge_index_dict.items():
            edges = edge_index.numpy()
            dgl_edge_index[metapath] = (edges[0], edges[1])
            if self.use_reverse:
                dgl_edge_index[tuple(reversed(metapath))] = (edges[1], edges[0])

        self.G = dgl.heterograph(dgl_edge_index)
        self.G.subgraph(self.training_idx).all_edges(etype=('paper', 'cites', 'paper'))
        self.G.to
