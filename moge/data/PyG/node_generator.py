import numpy as np
import torch
from ogb.nodeproppred import PygNodePropPredDataset

from moge.data.PyG.neighbor_sampler import NeighborSampler
from moge.data.network import HeteroNetDataset
from moge.module.PyG.utils import join_edge_indexes


class HeteroNeighborGenerator(HeteroNetDataset):
    def __init__(self, dataset, neighbor_sizes, node_types=None, metapaths=None, head_node_type=None, edge_dir=True,
                 resample_train=None, add_reverse_metapaths=True, inductive=False):
        self.neighbor_sizes = neighbor_sizes
        super(HeteroNeighborGenerator, self).__init__(dataset, node_types, metapaths, head_node_type, edge_dir,
                                                      resample_train, add_reverse_metapaths, inductive)

        if self.use_reverse:
            self.add_reverse_edge_index(self.edge_index_dict)

        self.graph_sampler = NeighborSampler(neighbor_sizes, self.edge_index_dict, self.num_nodes_dict,
                                             self.node_types, self.head_node_type)

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
        self.head_node_type = "_N"

        if not hasattr(data, "edge_reltype") and (not hasattr(data, "edge_attr") or data.edge_attr is None):
            self.metapaths = [(self.head_node_type, "_E", self.head_node_type)]
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
                    metapath = (str(node_type.item()), str(edge_type), str(node_type.item()))
                    print(metapath)
                    self.edge_index_dict[metapath] = edge_index

            self.num_nodes_dict = {str(node_type.item()): (data.node_species == node_type).sum() \
                                   for node_type in data.node_species.unique()}
            self.metapaths = list(self.edge_index_dict.keys())
            self.head_node_type = self.metapaths[0][0]
            self.y_dict = {node_type: data.y for node_type in self.num_nodes_dict}
            # TODO need to convert global node_index to local index

        elif hasattr(data, "edge_attr"):  # for ogbn-proteins
            self.edge_index_dict = {}
            # edge_reltype = data.edge_attr.argmax(1)

            for edge_type in range(data.edge_attr.size(1)):
                mask = data.edge_attr[:, edge_type] > 0.18
                edge_index = data.edge_index[:, mask]

                if edge_index.size(1) == 0: continue
                metapath = (self.head_node_type, str(edge_type), self.head_node_type)
                self.edge_index_dict[metapath] = edge_index

            self.metapaths = list(self.edge_index_dict.keys())
            self.num_nodes_dict = self.get_num_nodes_dict(self.edge_index_dict)

        else:
            raise Exception("Something wrong here")

        self.metapaths = list(self.edge_index_dict.keys())

        if self.node_types is None:
            self.node_types = list(self.num_nodes_dict.keys())

        self.x_dict = {self.head_node_type: data.x} if hasattr(data, "x") and data.x is not None else {}
        if not hasattr(self, "y_dict"):
            self.y_dict = {self.head_node_type: data.y} if hasattr(data, "y") else {}

        split_idx = dataset.get_idx_split()
        self.training_idx, self.validation_idx, self.testing_idx = split_idx["train"], split_idx["valid"], split_idx[
            "test"]

    def get_collate_fn(self, collate_fn: str, mode=None):
        assert mode is not None, "Must pass arg `mode` at get_collate_fn(). {'train', 'valid', 'test'}"

        def default_sampler(iloc):
            return self.sample(iloc, mode=mode)

        def khop_sampler(iloc):
            return self.khop_sampler(iloc, mode=mode)

        if "neighbor_sampler" in collate_fn or collate_fn is None:
            return default_sampler
        elif "khop_sampler" == collate_fn:
            return khop_sampler
        else:
            return super().get_collate_fn(collate_fn, mode=mode)

    def get_allowed_nodes(self, mode):
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
        return allowed_nodes, filter

    def compute_weights(self, y: torch.Tensor, batch_node_ids: dict, batch_seed_ids: torch.Tensor,
                        allowed_nodes: torch.Tensor, mode: str):
        # Weights
        weights = (y != -1) if y.dim() == 1 else (y != -1).all(1)
        weights = weights & np.isin(batch_node_ids[self.head_node_type], allowed_nodes)
        weights = torch.tensor(weights, dtype=torch.float)

        # Higher weights for sampled focal nodes in `n_idx`
        if batch_seed_ids is not None:
            seed_node_idx = np.isin(batch_node_ids[self.head_node_type], batch_seed_ids, invert=True)
            weights[seed_node_idx] = weights[seed_node_idx] * 0.2 if "train" in mode else 0.0
        return weights

    def sample(self, local_seed_nids, mode):
        if not isinstance(local_seed_nids, torch.Tensor) and not isinstance(local_seed_nids, dict):
            local_seed_nids = torch.tensor(local_seed_nids)

        # Sample subgraph
        batch_size, n_id, adjs = self.graph_sampler.sample(local_seed_nids)

        # Sample neighbors and return `sampled_local_nodes` as the set of all nodes traversed (in local index)
        local_sampled_nids = self.graph_sampler.get_local_nodes(adjs, n_id)

        # Ensure the sampled nodes only either belongs to training, validation, or testing set
        allowed_nodes, do_filter = self.get_allowed_nodes(mode)

        if do_filter:
            node_mask = np.isin(local_sampled_nids[self.head_node_type], allowed_nodes)
            local_sampled_nids[self.head_node_type] = local_sampled_nids[self.head_node_type][node_mask]

        # `global_node_index` here actually refers to the 'local' type-specific index of the original graph
        X = {"adjs": adjs,
             "batch_size": batch_size,
             "n_id": n_id,
             "global_node_index": local_sampled_nids,
             "x_dict": {}}

        # X["edge_index"] = self.graph_sampler.get_multi_edge_index_dict(adjs=adjs,
        #                                                                n_id=n_id,
        #                                                                local_sampled_nodes=local_sampled_nids)

        # x_dict attributes
        if hasattr(self, "x_dict") and len(self.x_dict) > 0:
            X["x_dict"] = {node_type: self.x_dict[node_type][local_sampled_nids[node_type]] \
                           for node_type in self.x_dict if node_type in local_sampled_nids}

        # y_dict
        assert torch.isclose(self.graph_sampler.global2local[n_id][:batch_size], local_seed_nids).all()

        if hasattr(self, "y_dict"):
            y = self.y_dict[self.head_node_type][self.graph_sampler.global2local[n_id]][:batch_size]
        else:
            y = None

        if y.dim() == 2 and y.size(1) == 1:
            y = y.squeeze(-1)

        # batch_seed_ids = self.graph_sampler.get_nid_relabel_dict(sampled_local_nodes)[self.head_node_type][self.graph_sampler.get_global_nidx(n_idx)]
        # print("node_ids_dict", node_ids_dict)
        # print("batch_seed_ids", batch_seed_ids)

        # weights = self.compute_weights(y, batch_node_ids=batch_nodes, batch_seed_ids=None,
        #                                allowed_nodes=allowed_nodes, mode=mode)
        weights = None
        return X, y, weights

    def khop_sampler(self, n_idx, mode):
        if not isinstance(n_idx, torch.Tensor) and not isinstance(n_idx, dict):
            n_idx = torch.tensor(n_idx)

        # Sample subgraph
        batch_size, n_id, adjs = self.graph_sampler.sample(n_idx)

        # Sample neighbors and return `sampled_local_nodes` as the set of all nodes traversed (in local index)
        sampled_local_nodes = self.graph_sampler.get_local_nodes(adjs, n_id)

        # Ensure the sampled nodes only either belongs to training, validation, or testing set
        allowed_nodes, filter = self.get_allowed_nodes(mode)

        if filter:
            node_mask = np.isin(sampled_local_nodes[self.head_node_type], allowed_nodes)
            sampled_local_nodes[self.head_node_type] = sampled_local_nodes[self.head_node_type][node_mask]

        # `global_node_index` here actually refers to the 'local' type-specific index of the original graph
        X = {"edge_index_dict": {},
             "global_node_index": sampled_local_nodes,
             "x_dict": {}}

        edge_index_dict = self.graph_sampler.get_edge_index_dict(adjs=adjs,
                                                                 n_id=n_id,
                                                                 sampled_local_nodes=sampled_local_nodes,
                                                                 filter_nodes=filter)
        X["edge_index_dict"] = edge_index_dict

        # Get higher-order relations
        next_edge_index_dict = edge_index_dict
        for t in range(len(self.neighbor_sizes) - 1):
            next_edge_index_dict = join_edge_indexes(next_edge_index_dict,
                                                     edge_index_dict,
                                                     sampled_local_nodes,
                                                     edge_sampling=True)

            X["edge_index_dict"].update(next_edge_index_dict)

        # x_dict attributes
        if hasattr(self, "x_dict") and len(self.x_dict) > 0:
            X["x_dict"] = {node_type: self.x_dict[node_type][X["global_node_index"][node_type]] \
                           for node_type in self.x_dict if node_type in X["global_node_index"]}

        # y_dict
        node_ids_dict = self.get_unique_nodes(X["edge_index"][1], source=False, target=True)

        if hasattr(self, "y_dict"):
            y = self.y_dict[self.head_node_type][n_idx].squeeze(-1)
        else:
            y = None

        if y.dim() == 2 and y.size(1) == 1:
            y = y.squeeze(-1)

        weights = self.compute_weights(y, batch_node_ids=node_ids_dict, batch_seed_ids=n_idx,
                                       allowed_nodes=allowed_nodes, mode=mode)

        return X, y, weights
