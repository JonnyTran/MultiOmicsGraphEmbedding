import torch
import torch_sparse
from torch_geometric.data import DataLoader, Data
from torch_geometric.utils import to_undirected
from torch_geometric.data import NeighborSampler
from torch_geometric.utils.hetero import group_hetero_graph

from moge.generator.datasets import HeteroNetDataset


class HeteroNeighborSampler(HeteroNetDataset):
    def __init__(self, dataset, node_types, metapaths=None, head_node_type=None, directed=True, train_ratio=0.7,
                 add_reverse_metapaths=True, neighbor_sizes=[25, 20]):
        super().__init__(dataset, node_types, metapaths, head_node_type, directed, train_ratio, add_reverse_metapaths)
        self.neighbor_sizes = neighbor_sizes

    def process_graph_sampler(self):
        # if self.use_reverse:
        #     self.add_reverse_edge_index(self.edge_index_dict)

        out = group_hetero_graph(self.edge_index_dict, self.num_nodes_dict)
        self.edge_index, self.edge_type, self.node_type, self.local_node_idx, self.local2global, self.key2int = out

        self.int2node_type = {v: k for k, v in self.key2int.items() if isinstance(k, str)}
        self.int2edge_type = {v: k for k, v in self.key2int.items() if isinstance(k, tuple)}

        x_dict = {}
        for key, x in self.x_dict.items():
            x_dict[self.key2int[key]] = x

        num_nodes_dict = {}
        for key, N in self.num_nodes_dict.items():
            num_nodes_dict[self.key2int[key]] = N

        loader = NeighborSampler(self.edge_index, node_idx=self.training_idx,
                                 sizes=self.neighbor_sizes, batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=num_workers)

    def sample_adj(self, metapath, source_node_id, num_neighbors=20):
        adj, n_id = self.graphs[metapath].sample_adj(source_node_id, num_neighbors=num_neighbors)
        row, col, e_id = adj.coo()
        edge_index = torch.stack([n_id[row], n_id[col]], dim=0)
        return edge_index

    def get_collate_fn(self, collate_fn: str, batch_size=None):
        if batch_size is not None:
            self.batch_size = batch_size * len(self.node_types)

        if "adj_sample" in collate_fn:
            return self.collate_adj_sample
        else:
            raise Exception(f"Correct collate function {collate_fn} not found.")

    def collate_adj_sample(self, iloc):
        pass


class NeighborSampler(HeteroNetDataset):
    def __init__(self, dataset, node_types, metapaths=None, head_node_type=None, directed=True, train_ratio=0.7,
                 add_reverse_metapaths=True):
        super().__init__(dataset, node_types, metapaths, head_node_type, directed, train_ratio, add_reverse_metapaths)

    def process_graph_sampler(self):
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

    def get_collate_fn(self, collate_fn: str, batch_size=None):
        if batch_size is not None:
            self.batch_size = batch_size * len(self.node_types)

        if "adj_sample" in collate_fn:
            return self.collate_adj_sample
        else:
            raise Exception(f"Correct collate function {collate_fn} not found.")

    def collate_adj_sample(self, iloc):
        pass
