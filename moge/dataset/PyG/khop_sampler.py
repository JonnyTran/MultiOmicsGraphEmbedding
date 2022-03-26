from collections import OrderedDict

from torch_geometric.utils.hetero import group_hetero_graph

from moge.dataset import HeteroGraphDataset


class KHopHeteroSampler():
    def __init__(self, neighbor_sizes, edge_index_dict, num_nodes_dict, node_types, head_node_type):
        self.head_node_type = head_node_type

        # Ensure head_node_type is first item in num_nodes_dict, since NeighborSampler.sample() function takes in index only the first
        num_nodes_dict = OrderedDict(
            [(node_type, num_nodes_dict[node_type]) for node_type in node_types])

        self.edge_index, self.edge_type, self.node_type, self.local_node_idx, self.local2global, self.key2int = \
            group_hetero_graph(edge_index_dict, num_nodes_dict)

        self.int2node_type = {type_int: node_type for node_type, type_int in self.key2int.items() if
                              node_type in node_types}
        self.int2edge_type = {type_int: edge_type for edge_type, type_int in self.key2int.items() if
                              edge_type in edge_index_dict}

    def sample(self, global_n_idx):
        pass

    def get_global_nidx(self, node_ids):
        pass

    def get_nodes_dict(self, adjs, n_id):
        pass

    def get_edge_index_dict(self, adjs, n_id, sampled_local_nodes: dict, filter_nodes: bool):
        pass


class KHopGenerator(HeteroGraphDataset):
    def __init__(self, dataset, neighbor_sizes, node_types=None, metapaths=None, head_node_type=None, edge_dir=True,
                 resample_train: float = None, add_reverse_metapaths=True, inductive=True):
        super().__init__(dataset, node_types, metapaths, head_node_type, edge_dir, resample_train,
                         add_reverse_metapaths, inductive)

        self.neighbor_sizes = neighbor_sizes
        super(KHopGenerator, self).__init__(dataset, node_types, metapaths, head_node_type, edge_dir, resample_train,
                                            add_reverse_metapaths, inductive)

        if self.use_reverse:
            self.add_reverse_edge_index(self.edge_index_dict)

        self.graph_sampler = KHopHeteroSampler(neighbor_sizes, self.edge_index_dict, self.num_nodes_dict,
                                               self.node_types, self.head_node_type)
