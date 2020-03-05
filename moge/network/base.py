from abc import abstractmethod
from collections import OrderedDict

import numpy as np


class Network(object):
    def __init__(self, networks: list) -> None:
        """
        A class that manages multiple graphs and the nodes between those graphs. Inheriting this class will run .process_network() and get_node_list()
        :param networks (list): a list of Networkx Graph's
        """
        self.networks = networks
        self.process_network()
        self.remove_bad_nodes()
        self.node_list = self.get_node_list()

    def get_node_list(self):
        if isinstance(self.networks, dict):
            node_list = list(
                OrderedDict.fromkeys([node for network in self.networks.values() for node in network.nodes]))
        elif isinstance(self.networks, list):
            node_list = list(OrderedDict.fromkeys([node for network in self.networks for node in network.nodes]))

        return node_list

    def remove_bad_nodes(self):
        bad_nodes = [node for node in self.get_node_list()
                     if node is None or node == np.nan or \
                     type(node) != str or \
                     node == ""]

        for network in self.networks:
            network.remove_nodes_from(bad_nodes)

    @abstractmethod
    def process_network(self):
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
