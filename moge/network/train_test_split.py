import copy
import random

import networkx as nx
import numpy as np


class NetworkTrainTestSplit():
    def __init__(self) -> None:
        self.train_network = None
        self.test_network = None
        self.val_network = None

    def split_train_test_edges(self,
                               node_list=None,
                               databases=["miRTarBase", "BioGRID", "lncRNome", "lncBase", "LncReg"],
                               test_frac=.05, val_frac=.01, seed=0, verbose=False):
        print("full_network", self.G.number_of_nodes(), self.G.number_of_edges()) if verbose else None
        G_train = self.G.copy()

        test_edges, val_edges = mask_test_edges(self,
                                                node_list=node_list,
                                                databases=databases,
                                                test_frac=test_frac, val_frac=val_frac, seed=seed, verbose=verbose)
        G_train.remove_edges_from(test_edges)
        G_train.remove_edges_from(val_edges)

        self.train_network = copy.copy(self)
        self.train_network.annotations = self.annotations
        self.train_network.G = G_train

        self.test_network = copy.copy(self)
        self.test_network.annotations = self.annotations
        self.test_network.G = nx.from_edgelist(edgelist=test_edges, create_using=nx.DiGraph)

        self.val_network = copy.copy(self)
        self.val_network.annotations = self.annotations
        self.val_network.G = nx.from_edgelist(edgelist=val_edges, create_using=nx.DiGraph)

        print("train_network", self.train_network.G.number_of_nodes(),
              self.train_network.G.number_of_edges()) if verbose else None
        print("test_network", self.test_network.G.number_of_nodes(),
              self.test_network.G.number_of_edges()) if verbose else None
        print("val_network", self.val_network.G.number_of_nodes(),
              self.val_network.G.number_of_edges()) if verbose else None

    def split_train_test_nodes(self,
                               node_list,
                               test_frac=.05, val_frac=.01, seed=0, verbose=False):
        """
        Randomly remove nodes from node_list with test_frac  and val_frac. Then, collect the edges with types in edge_types
        into the val_edges_dict and test_edges_dict. Edges not in the edge_types will be added back to the graph.

        :param self: HeterogeneousNetwork
        :param node_list: a list of nodes to split from
        :param edge_types: edges types to remove
        :param test_frac: fraction of edges to remove from training set to add to test set
        :param val_frac: fraction of edges to remove from training set to add to validation set
        :param seed:
        :param verbose:
        :return: network, val_edges_dict, test_edges_dict
        """
        print("full_network", self.G.number_of_nodes(), self.G.number_of_edges()) if verbose else None
        network_train, test_edges, val_edges, \
        test_nodes, val_nodes = mask_test_edges_by_nodes(self, node_list,
                                                         test_frac=test_frac, val_frac=val_frac, seed=seed,
                                                         verbose=verbose)
        self.train_network = copy.copy(self)
        self.train_network.annotations = self.annotations
        # self.train_network.node_list = [node for node in self.node_list if node in network_train.nodes()]
        self.train_network.G = network_train

        self.test_network = copy.copy(self)
        self.test_network.annotations = self.annotations
        self.test_network.G = nx.DiGraph()
        self.test_network.G.add_nodes_from(test_nodes)
        self.test_network.G.add_edges_from(test_edges)

        self.val_network = copy.copy(self)
        self.val_network.annotations = self.annotations
        self.val_network.G = nx.DiGraph()
        self.val_network.G.add_nodes_from(val_nodes)
        self.val_network.G.add_edges_from(val_edges)

        print("train_network", self.train_network.G.number_of_nodes(),
              self.train_network.G.number_of_edges()) if verbose else None
        print("test_network", self.test_network.G.number_of_nodes(),
              self.test_network.G.number_of_edges()) if verbose else None
        print("val_network", self.val_network.G.number_of_nodes(),
              self.val_network.G.number_of_edges()) if verbose else None

    def get_train_generator(self, generator, **kwargs):
        kwargs['network'] = self.train_network
        print(kwargs)
        return generator(**kwargs)

    def get_test_generator(self, generator, **kwargs):
        kwargs['network'] = self.test_network
        return generator(**kwargs)


def mask_test_edges_by_nodes(network, node_list,
                             test_frac=.1, val_frac=.05,
                             seed=0, verbose=False):
    g = network.G.copy()
    nodes_dict = network.nodes

    g.remove_nodes_from(list(nx.isolates(g)))
    no_of_edges_before = g.number_of_edges()
    no_of_nodes_before = g.number_of_nodes()

    test_nodes_size = int(len(node_list) * test_frac)
    val_nodes_size = int(len(node_list) * val_frac)

    # Sample nodes then create a set of edges induced by the sampled nodes
    random.seed(seed)
    test_nodes = []
    for node_type, nodes in nodes_dict.items():
        node_type_ratio = len(nodes) / len(node_list)
        test_nodes.extend(random.sample(nodes, int(test_nodes_size * node_type_ratio)))
    test_edges = [(u, v, d) for u, v, d in g.edges(test_nodes, data=True)]

    val_nodes = []
    for node_type, nodes in nodes_dict.items():
        node_type_ratio = len(nodes) / len(node_list)
        val_nodes.extend(random.sample(nodes, int(val_nodes_size * node_type_ratio)))
    val_edges = [(u, v, d) for u, v, d in g.edges(val_nodes, data=True)]

    g.remove_nodes_from(test_nodes)
    g.remove_nodes_from(val_nodes)

    print('removed', no_of_edges_before - g.number_of_edges(), "edges, and ",
          no_of_nodes_before - g.number_of_nodes(), "nodes.") if verbose else None

    return g, test_edges, val_edges, test_nodes, val_nodes


def mask_test_edges(network, node_list=None, databases=None,
                    test_frac=.10, val_frac=.05,
                    seed=0, verbose=False):
    g = network.G

    edges_to_remove = g.edges(data=True)
    if databases:
        edges_to_remove = [(u, v, d) for u, v, d in g.edges(data=True) if d["database"] in databases]
    if node_list:
        edges_to_remove = [(u, v, d) for u, v, d in edges_to_remove if (u in node_list) and (v in node_list)]
    print("edges_to_remove", len(edges_to_remove)) if verbose else None

    # Avoid removing edges in the MST
    temp_graph = nx.Graph(incoming_graph_data=edges_to_remove)
    mst_edges = nx.minimum_spanning_edges(temp_graph, data=False, ignore_nan=True)
    edges_to_remove = [(u, v, d) for u, v, d in edges_to_remove if ~((u, v) in mst_edges or (v, u) in mst_edges)]
    print("edges_to_remove (after MST)", len(edges_to_remove)) if verbose else None

    np.random.seed(seed)
    np.random.shuffle(edges_to_remove)

    test_edges_size = int(len(edges_to_remove) * test_frac)
    val_edges_size = int(len(edges_to_remove) * val_frac)

    test_edges = edges_to_remove[0: test_edges_size]
    val_edges = edges_to_remove[test_edges_size: test_edges_size + val_edges_size]

    print("test_edges_size", len(test_edges)) if verbose else None
    print("val_edges_size", len(val_edges)) if verbose else None
    return test_edges, val_edges
