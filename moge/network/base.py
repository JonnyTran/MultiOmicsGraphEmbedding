from abc import abstractmethod
from collections import OrderedDict


class Network(object):
    def __init__(self, networks: list) -> None:
        self.networks = networks
        self.preprocess_graph()
        self.node_list = self.get_node_list()

    def get_node_list(self):
        node_list = list(OrderedDict.fromkeys([node for network in self.networks for node in network.nodes]))
        return node_list

    @abstractmethod
    def preprocess_graph(self):
        raise NotImplementedError

    @abstractmethod
    def add_edges(self, edgelist, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def import_edgelist_file(self, file, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_edgelist(self, node_list, inclusive=True, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_adjacency_matrix(self, edge_types: list, node_list=None, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_graph_laplacian(self, edge_types: list, node_list=None, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_edge(self, i, j, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def remove_edges_from(self, edgelist, **kwargs):
        raise NotImplementedError

    def slice_adj(self, adj, nodes_A, nodes_B=None):
        if nodes_B is None:
            idx = [self.node_list.index(node) for node in nodes_A]
            return adj[idx, :][:, idx]
        else:
            idx_A = [self.node_list.index(node) for node in nodes_A]
            idx_B = [self.node_list.index(node) for node in nodes_B]
            return adj[idx_A, :][:, idx_B]
