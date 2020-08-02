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


class HeteroNeighborSampler(HeteroNetDataset):
    def __init__(self, dataset, node_types, metapaths=None, head_node_type=None, directed=True, train_ratio=0.7,
                 add_reverse_metapaths=True, neighbor_sizes=[25, 20], process_graphs=True):
        self.neighbor_sizes = neighbor_sizes
        super(HeteroNeighborSampler, self).__init__(dataset, node_types, metapaths, head_node_type, directed,
                                                    train_ratio, add_reverse_metapaths, process_graphs)

    def process_graph_sampler(self):
        if self.use_reverse:
            self.add_reverse_edge_index(self.edge_index_dict)

        # Ensure head_node_type is first item in num_nodes_dict, since NeighborSampler.sample() function takes in index only the first
        assert self.node_types[0] == self.head_node_type
        num_nodes_dict = OrderedDict([(node_type, self.num_nodes_dict[node_type]) for node_type in self.node_types])

        out = group_hetero_graph(self.edge_index_dict, num_nodes_dict)
        self.edge_index, self.edge_type, self.node_type, self.local_node_idx, self.local2global, self.key2int = out

        self.int2node_type = {type_int: node_type for node_type, type_int in self.key2int.items() if
                              isinstance(node_type, str)}
        self.int2edge_type = {type_int: edge_type for edge_type, type_int in self.key2int.items() if
                              isinstance(edge_type, tuple)}

        self.neighbor_sampler = NeighborSampler(self.edge_index, node_idx=self.training_idx,
                                                sizes=self.neighbor_sizes, batch_size=128, shuffle=True)

    def get_collate_fn(self, collate_fn: str, batch_size=None, mode=None):
        assert mode is not None, "Must pass arg mode at get_collate_fn()"

        def collate_wrapper(iloc):
            return self.collate_neighbor_sampler(iloc, mode)

        if "neighbor_sampler" in collate_fn:
            return collate_wrapper
        else:
            raise Exception(f"Collate function {collate_fn} not found.")

    def get_all_sampled_nodes_dict(self, adjs, n_id):
        """

        :param iloc: A tensor of indices for nodes of `head_node_type`
        :return sampled_nodes, n_id, adjs:
        """
        sampled_nodes = {}
        for adj in adjs:
            for row_col in [0, 1]:
                node_ids = n_id[adj.edge_index[row_col]]
                node_types = self.node_type[node_ids]

                for node_type_id in node_types.unique():
                    mask = node_types == node_type_id
                    local_node_ids = self.local_node_idx[node_ids[mask]]
                    sampled_nodes.setdefault(self.int2node_type[node_type_id.item()], []).append(local_node_ids)

        # Concatenate & remove duplicate nodes
        sampled_nodes = {k: torch.cat(v, dim=0).unique() for k, v in sampled_nodes.items()}
        return sampled_nodes

    def collate_neighbor_sampler(self, iloc, mode):
        """

        :param iloc: A tensor of a batch of indices in training_idx, validation_idx, or testing_idx
        :return:
        """
        if not isinstance(iloc, torch.Tensor):
            iloc = torch.tensor(iloc)

        batch_size, n_id, adjs = self.neighbor_sampler.sample(self.local2global[self.head_node_type][iloc])

        # Sample neighbors and return `sampled_nodes` as the set of all nodes traversed
        sampled_nodes = self.get_all_sampled_nodes_dict(adjs, n_id)

        # Ensure the sampled nodes only either belongs to training, validation, or testing set
        # if "train" in mode:
        #     allowed_nodes = self.training_idx
        # elif "valid" in mode:
        #     allowed_nodes = self.validation_idx
        # elif "test" in mode:
        #     allowed_nodes = self.testing_idx
        # else:
        #     raise Exception(f"Must set `mode` to either 'training', 'validation', or 'testing'. mode={mode}")
        #
        # indices = np.isin(sampled_nodes[self.head_node_type], allowed_nodes)
        # sampled_nodes[self.head_node_type] = sampled_nodes[self.head_node_type][indices]

        X = {"edge_index_dict": {}, "global_node_index": sampled_nodes, "x_dict": {}}

        global2batch_idx_dict = {
            node_type: dict(zip(sampled_nodes[node_type].numpy(), range(len(sampled_nodes[node_type])))) \
            for node_type in sampled_nodes}

        # Conbine all edge_index's and convert local node id to batch node ID
        X["edge_index_dict"] = {}
        for adj in adjs:
            for edge_type_id in self.edge_type[adj.e_id].unique():
                metapath = self.int2edge_type[edge_type_id.item()]
                head_type, tail_type = metapath[0], metapath[-1]

                edge_mask = self.edge_type[adj.e_id] == edge_type_id
                edge_index = adj.edge_index[:, edge_mask]

                edge_index[0] = self.local_node_idx[n_id[edge_index[0]]]
                edge_index[1] = self.local_node_idx[n_id[edge_index[1]]]

                # allowed_nodes_idx = self.local2global[self.head_node_type][sampled_nodes[self.head_node_type]]
                #
                # if head_type == self.head_node_type and tail_type == self.head_node_type:
                #     edge_set_mask = np.isin(edge_index[0], allowed_nodes_idx) \
                #                     & np.isin(edge_index[1], allowed_nodes_idx)
                #     edge_index = edge_index[:, edge_set_mask]
                # elif head_type == self.head_node_type:
                #     edge_set_mask = np.isin(edge_index[0], allowed_nodes_idx)
                #     edge_index = edge_index[:, edge_set_mask]
                # elif tail_type == self.head_node_type:
                #     edge_set_mask = np.isin(edge_index[1], allowed_nodes_idx)
                #     edge_index = edge_index[:, edge_set_mask]

                edge_index[0] = edge_index[0].apply_(
                    lambda x: global2batch_idx_dict[head_type][x])
                edge_index[1] = edge_index[1].apply_(
                    lambda x: global2batch_idx_dict[tail_type][x])

                X["edge_index_dict"].setdefault(metapath, []).append(edge_index)

        # TODO ensure no duplicate edge from adjs[0] to adjs[1]...
        X["edge_index_dict"] = {metapath: torch.cat(X["edge_index_dict"][metapath], dim=1) \
                                for metapath in X["edge_index_dict"]}

        if hasattr(self, "x_dict"):
            X["x_dict"] = {node_type: self.x_dict[node_type][X["global_node_index"][node_type]] for node_type in
                           self.x_dict}

        if len(self.y_dict) > 1:
            y = {node_type: y_true[X["global_node_index"][node_type]] for node_type, y_true in self.y_dict.items()}
        else:
            y = self.y_dict[self.head_node_type][X["global_node_index"][self.head_node_type]].squeeze(-1)
        return X, y, None
