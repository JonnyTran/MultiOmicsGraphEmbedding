from abc import abstractmethod
from typing import List, Union, Tuple, Any, Dict, Set

import networkx as nx
import pandas as pd

SEQ_DTYPE = "long"
SEQUENCE_COL = "sequence"


class Network(object):
    def __init__(self, networks: Dict[Tuple[str], nx.Graph]) -> None:
        """
        A class that manages multiple graphs and the nodes between those graphs. Inheriting this class will run .process_network() and get_node_list()
        :param networks (list): a list of Networkx Graph's
        """
        self.networks = networks
        self.process_network()
        self.node_list = self.get_all_nodes()

    def get_all_nodes(self) -> Set[str]:
        if isinstance(self.networks, dict):
            node_set = {node for network in self.networks.values() for node in network.nodes}
        elif isinstance(self.networks, list):
            node_set = {node for network in self.networks for node in network.nodes}
        else:
            node_set = {}

        return node_set

    def get_connected_nodes(self, layer: Union[str, Tuple[str, str, str]]):
        degrees = self.networks[layer].degree()
        return [node for node, deg in degrees if deg > 0]

    def remove_nodes_from(self, nodes: Union[List[str], Dict[str, Set[str]]]) -> None:
        nan_nodes = [node for node in self.get_all_nodes()
                     if pd.isna(node) or type(node) != str or len(node) == 0]

        if isinstance(nodes, dict):
            for metapath, g in self.networks.values():
                if not any(ntype in nodes for ntype in {metapath[0], metapath[-1]}): continue
                g.remove_nodes_from(nodes[metapath[0]])
                g.remove_nodes_from(nodes[metapath[-1]])
                if nan_nodes:
                    g.remove_nodes_from(nan_nodes)

        elif isinstance(nodes, list):
            for g in self.networks.values() if isinstance(self.networks, dict) else self.networks:
                g.remove_nodes_from(nodes)
                if nan_nodes:
                    g.remove_nodes_from(nan_nodes)

    @abstractmethod
    def process_network(self):
        raise NotImplementedError

    @abstractmethod
    def add_nodes(self, nodes: List[str], ntype: str, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def add_edges(self, edgelist: List[Union[Tuple[str, str], Tuple[str, str, Dict[str, Any]]]], **kwargs):
        raise NotImplementedError

    @abstractmethod
    def import_edgelist_file(self, filepath, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_edgelist(self, node_list: List[str], inclusive=True, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_adjacency_matrix(self, edge_types: List, node_list=None, method="GAT", output="csr", **kwargs):
        """
        Retrieves the adjacency matrix of a subnetwork with `edge_types` edge types  and `node_list` nodes. The adjacency
        matrix is preprocessed for `method`, e.g. adding self-loops in GAT, and is converted to a sparse matrix of `output` type.

        :param edge_types: a list of edge types to retrieve.
        :param node_list: list of nodes.
        :param method: one of {"GAT", "GCN"}, default: "GAT".
        :param output: one of {"csr", "coo", "dense"}, default "csr":
        :param kwargs:
        """
        raise NotImplementedError

    @abstractmethod
    def get_graph_laplacian(self, edge_types: List, node_list=None, **kwargs):
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


