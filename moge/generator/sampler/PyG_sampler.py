import multiprocessing
from collections import OrderedDict
import numpy as np
import torch

import torch_sparse
from torch_geometric.data import DataLoader, Data
from torch_geometric.utils import to_undirected
from torch_geometric.data import NeighborSampler
from torch_geometric.utils.hetero import group_hetero_graph

from moge.generator.datasets import HeteroNetDataset
from moge.generator.sampler.networkx_sampler import NetworkXSampler


class HeteroNeighborSampler(NetworkXSampler):
    def __init__(self, dataset, node_types, metapaths=None, head_node_type=None, directed=True, train_ratio=0.7,
                 add_reverse_metapaths=True, neighbor_sizes=[25, 20], process_graphs=True, multiworker=True):
        self.neighbor_sizes = neighbor_sizes
        super(HeteroNeighborSampler, self).__init__(dataset, node_types, metapaths, head_node_type, directed,
                                                    train_ratio, add_reverse_metapaths, process_graphs)
        try:
            cpus = multiprocessing.cpu_count()
        except NotImplementedError:
            cpus = 1

        if multiworker:
            self.graphs = {}
            pool = multiprocessing.Pool(processes=cpus)
            output = pool.map(self.create_nx_graph, self.metapaths)
            for (metapath, graph) in output:
                self.graphs[metapath] = graph
            pool.close()
        else:
            self.graphs = {metapath: graph for metapath, graph in
                           [self.create_nx_graph(metapath) for metapath in self.metapaths]}
        # We don't need the join_graph since will be using NeighborSampler with the group_hetero_graph()

    def process_graph_sampler(self):
        if self.use_reverse:
            self.add_reverse_edge_index(self.edge_index_dict)

        # Ensure head_node_type is first item in num_nodes_dict, since NeighborSampler.sample() function takes in index only the first
        assert self.node_types[0] == self.head_node_type
        num_nodes_dict = OrderedDict([(node_type, self.num_nodes_dict[node_type]) for node_type in self.node_types])

        out = group_hetero_graph(self.edge_index_dict, num_nodes_dict)
        self.edge_index, self.edge_type, self.node_type, self.local_node_idx, self.local2global, self.key2int = out

        self.int2node_type = {v: k for k, v in self.key2int.items() if isinstance(k, str)}
        self.int2edge_type = {v: k for k, v in self.key2int.items() if isinstance(k, tuple)}

        self.neighbor_sampler = NeighborSampler(self.edge_index, node_idx=self.training_idx,
                                                sizes=self.neighbor_sizes, batch_size=128, shuffle=True)

    def get_collate_fn(self, collate_fn: str, batch_size=None):
        if "neighbor_sampler" in collate_fn:
            return self.collate_neighbor_sampler
        else:
            raise Exception(f"Collate function {collate_fn} not found.")

    def neighbors_traversal(self, iloc):
        """

        param iloc: A tensor of indices for nodes of `head_node_type`
        :return:
        """
        batch_size, n_id, adjs = self.neighbor_sampler.sample(self.local_node_idx[iloc])
        sampled_nodes = {}
        for adj_idx in range(len(adjs)):
            for row_col in [0, 1]:
                node_ids = n_id[adjs[adj_idx].edge_index[row_col]]
                node_type_ids = self.node_type[node_ids]

                for node_type_id in node_type_ids.unique():
                    mask = node_type_ids == node_type_id
                    local_node_ids = self.local_node_idx[node_ids[mask]]
                    sampled_nodes.setdefault(self.int2node_type[node_type_id.item()], []).append(local_node_ids)

        sampled_nodes = {k: torch.cat(v, dim=0).unique() for k, v in
                         sampled_nodes.items()}  # concatenate & remove duplicates
        return sampled_nodes, n_id, adjs

    def collate_neighbor_sampler(self, iloc):
        """

        :param iloc: A tensor of a batch of indices in training_idx, validation_idx, or testing_idx
        :return:
        """
        if not isinstance(iloc, torch.Tensor):
            iloc = torch.tensor(iloc)

        sampled_nodes = self.neighbors_traversal(iloc)
        indices = np.isin(sampled_nodes[self.head_node_type], self.training_idx)
        sampled_nodes[self.head_node_type] = sampled_nodes[self.head_node_type][indices]

        X = {"edge_index_dict": {}, "global_node_index": sampled_nodes, "x_dict": {}}

        for metapath in self.metapaths:
            head_type, tail_type = metapath[0], metapath[-1]
            if head_type not in sampled_nodes or len(sampled_nodes[head_type]) == 0: continue
            if tail_type not in sampled_nodes or len(sampled_nodes[tail_type]) == 0: continue
            try:
                X["edge_index_dict"][metapath] = self.get_adj_edge_index(
                    self.graphs[metapath],
                    nodes_A=self.convert_index2name(sampled_nodes[head_type], head_type),
                    nodes_B=self.convert_index2name(sampled_nodes[tail_type], tail_type))

            except Exception as e:
                print("sampled_nodes[head_type]", sampled_nodes[head_type])
                print("sampled_nodes[tail_type]", sampled_nodes[tail_type])
                raise e

        if self.use_reverse:
            self.add_reverse_edge_index(X["edge_index_dict"])

        if hasattr(self, "x_dict"):
            X["x_dict"] = {node_type: self.x_dict[node_type][X["global_node_index"][node_type]] for node_type in
                           self.x_dict}

        if len(self.y_dict) > 1:
            y = {node_type: y_true[X["global_node_index"][node_type]] for node_type, y_true in self.y_dict.items()}
        else:
            y = self.y_dict[self.head_node_type][X["global_node_index"][self.head_node_type]].squeeze(-1)
        return X, y, None

