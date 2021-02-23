from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
from ogb.nodeproppred import PygNodePropPredDataset

import torch_geometric
from torch_geometric.data.sampler import Adj, EdgeIndex
from torch_geometric.utils.hetero import group_hetero_graph

from moge.generator.network import HeteroNetDataset
from moge.generator.PyG.khop_sampler import KHopSampler
from moge.module.utils import tensor_sizes


class NeighborSampler(torch_geometric.data.NeighborSampler):
    def sample(self, batch):
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch)

        batch_size: int = len(batch)

        adjs = []
        n_id = batch
        for size in self.sizes:
            adj_t, n_id = self.adj_t.sample_adj(n_id, size, replace=False)
            e_id = adj_t.storage.value()
            size = adj_t.sparse_sizes()[::-1]
            if self.__val__ is not None:
                adj_t.set_value_(self.__val__[e_id], layout='coo')

            if self.is_sparse_tensor:
                adjs.append(Adj(adj_t, e_id, size))
            else:
                row, col, _ = adj_t.coo()
                edge_index = torch.stack([col, row], dim=0)  # flow source_to_target
                adjs.append(EdgeIndex(edge_index, e_id, size))

        if len(adjs) > 1:
            return batch_size, n_id, adjs[::-1]
        else:
            return batch_size, n_id, adjs[0]


class HeteroNeighborSampler(HeteroNetDataset):
    def __init__(self, dataset, neighbor_sizes, node_types=None, metapaths=None, head_node_type=None, directed=True,
                 resample_train=None, add_reverse_metapaths=True, inductive=False):
        self.neighbor_sizes = neighbor_sizes
        super(HeteroNeighborSampler, self).__init__(dataset, node_types, metapaths, head_node_type, directed,
                                                    resample_train, add_reverse_metapaths, inductive)

        if self.use_reverse:
            self.add_reverse_edge_index(self.edge_index_dict)

        # Ensure head_node_type is first item in num_nodes_dict, since NeighborSampler.sample() function takes in index only the first
        num_nodes_dict = OrderedDict([(node_type, self.num_nodes_dict[node_type]) for node_type in self.node_types])

        self.edge_index, self.edge_type, self.node_type, self.local_node_idx, self.local2global, self.key2int = \
            group_hetero_graph(self.edge_index_dict, num_nodes_dict)

        self.int2node_type = {type_int: node_type for node_type, type_int in self.key2int.items() if
                              node_type in self.node_types}
        self.int2edge_type = {type_int: edge_type for edge_type, type_int in self.key2int.items() if
                              edge_type in self.edge_index_dict}

        self.neighbor_sampler = NeighborSampler(self.edge_index, node_idx=self.training_idx,
                                                sizes=self.neighbor_sizes, batch_size=128, shuffle=True)

    def process_PygNodeDataset_hetero(self, dataset: PygNodePropPredDataset, ):
        data = dataset[0]
        self._name = dataset.name
        self.edge_index_dict = data.edge_index_dict
        self.num_nodes_dict = data.num_nodes_dict if hasattr(data, "num_nodes_dict") else self.get_num_nodes_dict(
            self.edge_index_dict)

        if self.node_types is None:
            self.node_types = list(self.num_nodes_dict.keys())

        if hasattr(data, "x_dict"):
            self.x_dict = data.x_dict
        elif hasattr(data, "x"):
            self.x_dict = {self.head_node_type: data.x}
        else:
            self.x_dict = {}

        if hasattr(data, "y_dict"):
            self.y_dict = data.y_dict
        elif hasattr(data, "y"):
            self.y_dict = {self.head_node_type: data.y}
        else:
            self.y_dict = {}

        self.y_index_dict = {node_type: torch.arange(self.num_nodes_dict[node_type]) for node_type in
                             self.y_dict.keys()}

        if self.head_node_type is None:
            if hasattr(self, "y_dict"):
                self.head_node_type = list(self.y_dict.keys())[0]
            else:
                self.head_node_type = self.node_types[0]

        self.metapaths = list(self.edge_index_dict.keys())

        split_idx = dataset.get_idx_split()
        self.training_idx, self.validation_idx, self.testing_idx = split_idx["train"][self.head_node_type], \
                                                                   split_idx["valid"][self.head_node_type], \
                                                                   split_idx["test"][self.head_node_type]

    def process_PygNodeDataset_homo(self, dataset: PygNodePropPredDataset, ):
        data = dataset[0]
        self._name = dataset.name
        self.head_node_type = "entity"

        if not hasattr(data, "edge_reltype") and not hasattr(data, "edge_attr"):
            self.metapaths = [(self.head_node_type, "default", self.head_node_type)]
            self.edge_index_dict = {self.metapaths[0]: data.edge_index}
            self.num_nodes_dict = self.get_num_nodes_dict(self.edge_index_dict)


        elif False and hasattr(data, "edge_attr") and hasattr(data, "node_species"):  # for ogbn-proteins
            self.edge_index_dict = {}
            edge_reltype = data.edge_attr.argmax(1)

            for node_type in data.node_species.unique():
                for edge_type in range(data.edge_attr.size(1)):
                    edge_mask = edge_reltype == edge_type
                    node_mask = (data.node_species[data.edge_index[0]].squeeze(-1) == node_type).logical_and(
                        data.node_species[data.edge_index[1]].squeeze(-1) == node_type)

                    edge_index = data.edge_index[:, node_mask.logical_and(edge_mask)]

                    if edge_index.size(1) == 0: continue
                    self.edge_index_dict[(str(node_type.item()), str(edge_type), str(node_type.item()))] = edge_index

            self.num_nodes_dict = {str(node_type.item()): data.node_species.size(0) for node_type in
                                   data.node_species.unique()}
            self.metapaths = list(self.edge_index_dict.keys())
            self.head_node_type = self.metapaths[0][0]
            self.y_dict = {node_type: data.y for node_type in self.num_nodes_dict}
            # TODO need to convert global node_index to local index

        elif hasattr(data, "edge_attr"):  # for ogbn-proteins
            self.edge_index_dict = {}
            edge_reltype = data.edge_attr.argmax(1)

            for edge_type in range(data.edge_attr.size(1)):
                mask = edge_reltype == edge_type
                edge_index = data.edge_index[:, mask]

                if edge_index.size(1) == 0: continue
                self.edge_index_dict[(self.head_node_type, str(edge_type), self.head_node_type)] = edge_index

            self.metapaths = list(self.edge_index_dict.keys())
            self.num_nodes_dict = self.get_num_nodes_dict(self.edge_index_dict)

        else:
            raise Exception("Something wrong here")

        if self.node_types is None:
            self.node_types = list(self.num_nodes_dict.keys())

        self.x_dict = {self.head_node_type: data.x} if hasattr(data, "x") and data.x is not None else {}
        if not hasattr(self, "y_dict"):
            self.y_dict = {self.head_node_type: data.y} if hasattr(data, "y") else {}

        self.metapaths = list(self.edge_index_dict.keys())

        split_idx = dataset.get_idx_split()
        self.training_idx, self.validation_idx, self.testing_idx = split_idx["train"], split_idx["valid"], split_idx[
            "test"]

    def get_collate_fn(self, collate_fn: str, mode=None):
        assert mode is not None, "Must pass arg `mode` at get_collate_fn(). {'train', 'valid', 'test'}"

        def collate_wrapper(iloc):
            return self.sample(iloc, mode=mode)

        if "neighbor_sampler" in collate_fn or collate_fn is None:
            return collate_wrapper
        else:
            return super().get_collate_fn(collate_fn, mode=mode)

    def get_local_node_index(self, adjs, n_id):
        sampled_nodes = {}
        for adj in adjs:
            for i in [0, 1]:
                node_ids = n_id[adj.edge_index[i]]
                node_types = self.node_type[node_ids]

                for node_type_id in node_types.unique():
                    mask = node_types == node_type_id
                    local_node_ids = self.local_node_idx[node_ids[mask]]
                    sampled_nodes.setdefault(self.int2node_type[node_type_id.item()], []).append(local_node_ids)

        # Concatenate & remove duplicate nodes
        sampled_nodes = {k: torch.cat(v, dim=0).unique() for k, v in sampled_nodes.items()}
        return sampled_nodes

    def sample(self, n_idx, mode):
        """

        Args:
            n_idx: A tensor of a batch of node indices in training_idx, validation_idx, or testing_idx
            mode:

        Returns:

        """
        if not isinstance(n_idx, torch.Tensor) and not isinstance(n_idx, dict):
            n_idx = torch.tensor(n_idx)

        if isinstance(n_idx, dict):
            n_idx_to_sample = torch.cat([self.local2global[ntype][nid] for ntype, nid in n_idx.items()], 0)
        else:
            n_idx_to_sample = self.local2global[self.head_node_type][n_idx]

        batch_size, n_id, adjs = self.neighbor_sampler.sample(n_idx_to_sample)
        if not isinstance(adjs, list):
            adjs = [adjs]

        # Sample neighbors and return `sampled_local_nodes` as the set of all nodes traversed (in local index)
        sampled_local_nodes = self.get_local_node_index(adjs, n_id)

        # Ensure the sampled nodes only either belongs to training, validation, or testing set
        if "train" in mode:
            filter = True if self.inductive else False
            if self.inductive and hasattr(self, "training_subgraph_idx"):
                allowed_nodes = self.training_subgraph_idx
            else:
                allowed_nodes = self.training_idx
        elif "valid" in mode:
            filter = True if self.inductive else False
            if self.inductive and hasattr(self, "training_subgraph_idx"):
                allowed_nodes = torch.cat([self.validation_idx, self.training_subgraph_idx])
            else:
                allowed_nodes = self.validation_idx
        elif "test" in mode:
            filter = False
            allowed_nodes = self.testing_idx
        else:
            raise Exception(f"Must set `mode` to either 'training', 'validation', or 'testing'. mode={mode}")

        if filter:
            node_mask = np.isin(sampled_local_nodes[self.head_node_type], allowed_nodes)
            sampled_local_nodes[self.head_node_type] = sampled_local_nodes[self.head_node_type][node_mask]

        # `global_node_index` here actually refers to the 'local' type-specific index of the original graph
        X = {"edge_index_dict": {},
             "global_node_index": sampled_local_nodes,
             "x_dict": {}}



        X["edge_index_dict"] = self.get_local_edge_index_dict(adjs=adjs, n_id=n_id,
                                                              sampled_local_nodes=sampled_local_nodes,
                                                              filter_nodes=filter)

        # x_dict attributes
        if hasattr(self, "x_dict") and len(self.x_dict) > 0:
            X["x_dict"] = {node_type: self.x_dict[node_type][X["global_node_index"][node_type]] \
                           for node_type in self.x_dict}

        # y_dict
        if hasattr(self, "y_dict") and len(self.y_dict) > 1:
            y = {node_type: y_true[X["global_node_index"][node_type]] for node_type, y_true in self.y_dict.items()}
        elif hasattr(self, "y_dict"):
            y = self.y_dict[self.head_node_type][X["global_node_index"][self.head_node_type]].squeeze(-1)
        else:
            y = None

        weights = (y != -1) & np.isin(X["global_node_index"][self.head_node_type], allowed_nodes)
        weights = torch.tensor(weights, dtype=torch.float)

        if hasattr(self, "x_dict") and len(self.x_dict) > 0:
            assert X["global_node_index"][self.head_node_type].size(0) == X["x_dict"][self.head_node_type].size(0)

        # assert y.size(0) == X["global_node_index"][self.head_node_type].size(0)
        # assert y.size(0) == weights.size(0)
        return X, y, weights

    def get_local_edge_index_dict(self, adjs, n_id, sampled_local_nodes: dict, filter_nodes: bool):
        """ Conbine all edge_index's and convert local node id to "batch node index" that aligns with `x_dict` and
        `global_node_index`

        :param adjs:
        :param n_id:
        :param sampled_local_nodes:
        :param filter_nodes:
        :return:
        """
        relabel_nodes = {
            node_type: dict(zip(sampled_local_nodes[node_type].numpy(),
                                range(len(sampled_local_nodes[node_type])))
                            ) for node_type in sampled_local_nodes}

        edge_index_dict = {}
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

                # Filter nodes for only head node type
                if filter_nodes < 2:
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

                # Filter nodes from all node types
                else:
                    if head_type not in relabel_nodes or tail_type not in relabel_nodes: continue

                    allowed_nodes_idx = torch.cat([self.local2global[ntype][list(local_global.keys())] \
                                                   for ntype, local_global in relabel_nodes.items()], dim=0)

                    mask = np.isin(edge_index[0], allowed_nodes_idx) & np.isin(edge_index[1], allowed_nodes_idx)
                    edge_index = edge_index[:, mask]

                if edge_index.shape[1] == 0: continue

                # Convert node global index -> local index -> batch index
                edge_index[0] = self.local_node_idx[edge_index[0]].apply_(relabel_nodes[head_type].get)
                edge_index[1] = self.local_node_idx[edge_index[1]].apply_(relabel_nodes[tail_type].get)

                edge_index_dict.setdefault(metapath, []).append(edge_index)

        # Join edges from the adjs (from iterative layer-wise sampling)
        edge_index_dict = {metapath: torch.cat(edge_index, dim=1) \
                           for metapath, edge_index in edge_index_dict.items()}

        # Ensure no duplicate edges in each metapath
        edge_index_dict = {metapath: edge_index[:, self.nonduplicate_indices(edge_index)] \
                           for metapath, edge_index in edge_index_dict.items()}
        return edge_index_dict

    def nonduplicate_indices(self, edge_index):
        edge_df = pd.DataFrame(edge_index.t().numpy())  # shape: (n_edges, 2)
        return ~edge_df.duplicated(subset=[0, 1])


def intersection(edge_index_dict_A, edge_index_dict_B):
    inters = {}
    for metapath, edge_index in edge_index_dict_A.items():
        if metapath not in edge_index_dict_B:
            inters[metapath] = 0
            continue

        A = pd.DataFrame(edge_index.T.numpy(), columns=["source", "target"])
        B = pd.DataFrame(edge_index_dict_B[metapath].T.numpy(), columns=["source", "target"])
        int_df = pd.merge(A, B, how='inner', on=["source", "target"])
        inters[metapath] = int_df.shape[0]

    return inters
