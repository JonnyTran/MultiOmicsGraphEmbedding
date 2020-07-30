import torch
import torch_sparse
from torch_geometric.data import DataLoader, Data
from torch_geometric.utils import to_undirected
from torch_geometric.data import NeighborSampler
from torch_geometric.utils.hetero import group_hetero_graph

from moge.generator.datasets import HeteroNetDataset


class HeteroNeighborSampler(HeteroNetDataset):

    def __init__(self, dataset, node_types, metapaths=None, head_node_type=None, directed=True, train_ratio=0.7,
                 add_reverse_metapaths=True, num_neighbors=25):
        super().__init__(dataset, node_types, metapaths, head_node_type, directed, train_ratio, add_reverse_metapaths)
        self.num_neighbors = num_neighbors

    def process_graphs(self):
        out = group_hetero_graph(self.edge_index_dict, self.num_nodes_dict)
        edge_index, edge_type, node_type, local_node_idx, local2global, key2int = out

        x_dict = {}
        for key, x in self.x_dict.items():
            x_dict[key2int[key]] = x

        num_nodes_dict = {}
        for key, N in self.num_nodes_dict.items():
            num_nodes_dict[key2int[key]] = N

    def train_dataloader(self, collate_fn=None, batch_size=128, num_workers=12):
        loader = NeighborSampler(edge_index, node_idx=self.training_idx,
                                 sizes=[self.num_neighbors, ] * len(self.edge_index_dict), batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=num_workers)
        return loader

    def val_dataloader(self, collate_fn=None, batch_size=128, num_workers=4):
        loader = NeighborSampler(edge_index, node_idx=self.validation_idx,
                                 sizes=[self.num_neighbors, ] * len(self.edge_index_dict), batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=num_workers)
        return loader

    def test_dataloader(self, collate_fn=None, batch_size=128, num_workers=4):
        loader = NeighborSampler(edge_index, node_idx=self.testing_idx,
                                 sizes=[self.num_neighbors, ] * len(self.edge_index_dict), batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=num_workers)
        return loader


class NeighborSampler(HeteroNetDataset):
    def __init__(self, dataset, node_types, metapaths=None, head_node_type=None, directed=True, train_ratio=0.7,
                 add_reverse_metapaths=True):
        super().__init__(dataset, node_types, metapaths, head_node_type, directed, train_ratio, add_reverse_metapaths)

    def process_graphs(self):
        self.graphs = {}

        for metapath in self.metapaths:
            edge_index = self.edge_index_dict[metapath]
            adj = torch_sparse.SparseTensor(row=edge_index[0], col=edge_index[1],
                                            value=torch.ones(edge_index.size(1), dtype=torch.float),
                                            sparse_sizes=(self.num_nodes_dict[metapath[0]],
                                                          self.num_nodes_dict[metapath[-1]]),
                                            is_sorted=True)
            self.graphs[metapath] = adj
            if self.use_reverse:
                self.graphs[tuple(reversed(metapath))] = adj.t()

    def sample_adj(self, metapath, source_node_id, num_neighbors=20):
        adj, n_id = self.graphs[metapath].sample_adj(source_node_id, num_neighbors=num_neighbors)
        row, col, e_id = adj.coo()
        edge_index = torch.stack([n_id[row], n_id[col]], dim=0)
        return edge_index
