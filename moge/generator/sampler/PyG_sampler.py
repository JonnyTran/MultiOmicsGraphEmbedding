import multiprocessing
from collections import OrderedDict
import pandas as pd
import numpy as np
import torch

from cogdl.datasets.han_data import HANDataset

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
        if "HAN" in collate_fn:
            return self.collate_HAN
        else:
            raise Exception(f"Collate function {collate_fn} not found.")

    def get_local_nodes_dict(self, adjs, n_id):
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

    def collate_neighbor_sampler(self, iloc, mode, filter_nodes=False):
        """

        :param iloc: A tensor of a batch of indices in training_idx, validation_idx, or testing_idx
        :return:
        """
        if not isinstance(iloc, torch.Tensor):
            iloc = torch.tensor(iloc)

        batch_size, n_id, adjs = self.neighbor_sampler.sample(self.local2global[self.head_node_type][iloc])
        if not isinstance(adjs, list):
            adjs = [adjs]
        # Sample neighbors and return `sampled_local_nodes` as the set of all nodes traversed (in local index)
        sampled_local_nodes = self.get_local_nodes_dict(adjs, n_id)

        # Ensure the sampled nodes only either belongs to training, validation, or testing set
        if "train" in mode:
            allowed_nodes = self.training_idx
        elif "valid" in mode:
            allowed_nodes = self.validation_idx
        elif "test" in mode:
            allowed_nodes = self.testing_idx
        else:
            raise Exception(f"Must set `mode` to either 'training', 'validation', or 'testing'. mode={mode}")

        assert np.isin(iloc, allowed_nodes).all()

        if filter_nodes:
            node_mask = np.isin(sampled_local_nodes[self.head_node_type], allowed_nodes)
            sampled_local_nodes[self.head_node_type] = sampled_local_nodes[self.head_node_type][node_mask]

        # `global_node_index` here actually refers to the 'local' type-specific index of the original graph
        X = {"edge_index_dict": {},
             "global_node_index": sampled_local_nodes,
             "x_dict": {}}

        local2batch_idx_dict = {
            node_type: dict(zip(sampled_local_nodes[node_type].numpy(),
                                range(len(sampled_local_nodes[node_type])))
                            ) for node_type in sampled_local_nodes}

        # Conbine all edge_index's and convert local node id to "batch node index" that aligns with `x_dict` and `global_node_index`
        X["edge_index_dict"] = {}
        for adj in adjs:
            for edge_type_id in self.edge_type[adj.e_id].unique():
                metapath = self.int2edge_type[edge_type_id.item()]
                head_type, tail_type = metapath[0], metapath[-1]

                # Filter edges to correct edge_type_id
                edge_mask = self.edge_type[adj.e_id] == edge_type_id
                edge_index = adj.edge_index[:, edge_mask]

                # convert from "sampled_edge_index" to global index
                edge_index[0] = n_id[edge_index[0]]
                edge_index[1] = n_id[edge_index[1]]

                if filter_nodes:
                    # If node_type==self.head_node_type, then remove edge_index with nodes not in allowed_nodes_idx
                    allowed_nodes_idx = self.local2global[self.head_node_type][sampled_local_nodes[self.head_node_type]]
                    if head_type == self.head_node_type and tail_type == self.head_node_type:
                        mask = np.isin(edge_index[0], allowed_nodes_idx) & np.isin(edge_index[1], allowed_nodes_idx)
                        edge_index = edge_index[:, mask]
                    elif head_type == self.head_node_type:
                        mask = np.isin(edge_index[0], allowed_nodes_idx)
                        edge_index = edge_index[:, mask]
                    elif tail_type == self.head_node_type:
                        mask = np.isin(edge_index[1], allowed_nodes_idx)
                        edge_index = edge_index[:, mask]

                # Convert node global index -> local index -> batch index
                edge_index[0] = self.local_node_idx[edge_index[0]].apply_(
                    lambda x: local2batch_idx_dict[head_type][x])
                edge_index[1] = self.local_node_idx[edge_index[1]].apply_(
                    lambda x: local2batch_idx_dict[tail_type][x])

                X["edge_index_dict"].setdefault(metapath, []).append(edge_index)

        # Join edges from the adjs
        X["edge_index_dict"] = {metapath: torch.cat(edge_index, dim=1) \
                                for metapath, edge_index in X["edge_index_dict"].items()}
        # Ensure no duplicate edge from adjs[0] to adjs[1]...
        X["edge_index_dict"] = {metapath: edge_index[:, self.nonduplicate_indices(edge_index)] \
                                for metapath, edge_index in X["edge_index_dict"].items()}

        if hasattr(self, "x_dict"):
            X["x_dict"] = {node_type: self.x_dict[node_type][X["global_node_index"][node_type]] \
                           for node_type in self.x_dict}

        if len(self.y_dict) > 1:
            y = {node_type: y_true[X["global_node_index"][node_type]] for node_type, y_true in self.y_dict.items()}
        else:
            y = self.y_dict[self.head_node_type][X["global_node_index"][self.head_node_type]].squeeze(-1)

        weights = torch.tensor(np.isin(X["global_node_index"][self.head_node_type], allowed_nodes), dtype=torch.float)
        return X, y, weights

    def nonduplicate_indices(self, edge_index):
        edge_df = pd.DataFrame(edge_index.t().numpy())  # shape: (n_edges, 2)
        return ~edge_df.duplicated(subset=[0, 1])

