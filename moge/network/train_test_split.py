import copy
import random
from collections import OrderedDict

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sps
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.model_selection import IterativeStratification


def filter_y_multilabel(network, y_label="go_id", min_count=2):
    nodes_index = network.annotations[["Transcript sequence", y_label]].dropna().index

    label_counts = {}
    for items in network.annotations.loc[nodes_index, y_label].str.split("|"):
        for item in items:
            label_counts[item] = label_counts.setdefault(item, 0) + 1

    label_counts = pd.Series(label_counts)
    labels_filter = label_counts[label_counts < min_count].index
    print("labels_filter", len(labels_filter))

    y_labels = network.annotations.loc[nodes_index, y_label].str.split("|")
    y_labels = y_labels.map(lambda go_terms: [item for item in go_terms if item not in labels_filter])

    return y_labels


def stratify_train_test(y_label, n_splits=10):
    y_label_bin = MultiLabelBinarizer().fit_transform(y_label)

    k_fold = IterativeStratification(n_splits=n_splits, order=1)
    for train, test in k_fold.split(y_label.index.to_list(), sps.lil_matrix(y_label_bin)):
        train_nodes = list(y_label.index[train])
        test_nodes = list(y_label.index[test])
        return train_nodes, test_nodes

class NetworkTrainTestSplit():
    def __init__(self) -> None:
        self.training = None
        self.testing = None
        self.validation = None

    def split_train_test_edges(self, directed: bool,
                               node_list=None,
                               databases=["miRTarBase", "BioGRID", "lncRNome", "lncBase", "LncReg"],
                               test_frac=.05, val_frac=.01, seed=0, verbose=False):
        if directed:
            print("full_network directed", self.G.number_of_nodes(), self.G.number_of_edges()) if verbose else None
        else:
            print("full_network undirected", self.G_u.number_of_nodes(),
                  self.G_u.number_of_edges()) if verbose else None

        if directed:
            G_train = self.G.copy()
        else:
            G_train = self.G_u.copy()

        test_edges, val_edges = mask_test_edges(self,
                                                node_list=node_list,
                                                databases=databases,
                                                test_frac=test_frac, val_frac=val_frac, seed=seed, verbose=verbose)
        G_train.remove_edges_from(test_edges)
        G_train.remove_edges_from(val_edges)

        self.training = copy.copy(self)
        self.training.annotations = self.annotations

        if directed:
            self.training.G = G_train
        else:
            self.training.G_u = G_train

        self.testing = copy.copy(self)
        self.testing.annotations = self.annotations
        if directed:
            self.testing.G = nx.from_edgelist(edgelist=test_edges, create_using=nx.DiGraph)
        else:
            self.testing.G_u = nx.from_edgelist(edgelist=test_edges, create_using=nx.Graph)

        if val_frac > 0:
            self.validation = copy.copy(self)
            self.validation.annotations = self.annotations
            if directed:
                self.validation.G = nx.from_edgelist(edgelist=val_edges, create_using=nx.DiGraph)
            else:
                self.validation.G_u = nx.from_edgelist(edgelist=val_edges, create_using=nx.Graph)

        if directed:
            print("train_network", self.training.G.number_of_nodes(),
                  self.training.G.number_of_edges()) if verbose else None
            print("test_network", self.testing.G.number_of_nodes(),
                  self.testing.G.number_of_edges()) if verbose else None
            print("val_network", self.validation.G.number_of_nodes(),
                  self.validation.G.number_of_edges()) if verbose and val_frac > 0 else None
        else:
            print("train_network", self.training.G_u.number_of_nodes(),
                  self.training.G_u.number_of_edges()) if verbose else None
            print("test_network", self.testing.G.number_of_nodes(),
                  self.testing.G_u.number_of_edges()) if verbose else None
            print("val_network", self.validation.G.number_of_nodes(),
                  self.validation.G_u.number_of_edges()) if verbose and val_frac > 0 else None

    def split_train_test_nodes(self, directed: bool, node_list,
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
        if directed:
            print("full_network", self.G.number_of_nodes(), self.G.number_of_edges()) if verbose else None
        else:
            print("full_network", self.G_u.number_of_nodes(), self.G_u.number_of_edges()) if verbose else None

        network_train, test_edges, val_edges, \
        test_nodes, val_nodes = mask_test_edges_by_nodes(network=self, directed=directed, node_list=node_list,
                                                         test_frac=test_frac, val_frac=val_frac, seed=seed,
                                                         verbose=verbose)
        self.training = copy.copy(self)
        self.training.annotations = self.annotations
        self.training.node_list = [node for node in self.node_list if node in network_train.nodes()]
        self.training.node_list = list(OrderedDict.fromkeys(self.training.node_list))

        if directed:
            self.training.G = network_train
        else:
            self.training.G_u = network_train

        # Test network
        self.testing = copy.copy(self)
        self.testing.annotations = self.annotations
        self.testing.node_list = list(OrderedDict.fromkeys(test_nodes))
        if directed:
            self.testing.G = nx.DiGraph()
            self.testing.G.add_nodes_from(test_nodes)
            self.testing.G.add_edges_from(test_edges)
        else:
            self.testing.G_u = nx.Graph()
            self.testing.G_u.add_nodes_from(test_nodes)
            self.testing.G_u.add_edges_from(test_edges)

        if val_frac > 0:
            self.validation = copy.copy(self)
            self.validation.annotations = self.annotations
            self.validation.node_list = list(OrderedDict.fromkeys(val_nodes))
            if directed:
                self.validation.G = nx.DiGraph()
                self.validation.G.add_nodes_from(val_nodes)
                self.validation.G.add_edges_from(val_edges)
            else:
                self.validation.G_u = nx.Graph()
                self.validation.G_u.add_nodes_from(val_nodes)
                self.validation.G_u.add_edges_from(val_edges)

        if directed:
            print("train_network", self.training.G.number_of_nodes(),
                  self.training.G.number_of_edges()) if verbose else None
            print("test_network", self.testing.G.number_of_nodes(),
                  self.testing.G.number_of_edges()) if verbose else None
            print("val_network", self.validation.G.number_of_nodes(),
                  self.validation.G.number_of_edges()) if verbose and val_frac > 0 else None
        else:
            print("train_network", self.training.G_u.number_of_nodes(),
                  self.training.G_u.number_of_edges()) if verbose else None
            print("test_network", self.testing.G.number_of_nodes(),
                  self.testing.G_u.number_of_edges()) if verbose else None
            print("val_network", self.validation.G.number_of_nodes(),
                  self.validation.G_u.number_of_edges()) if verbose and val_frac > 0 else None

    def split_train_test_stratified(self, directed: bool, stratify_label: str, n_splits=8, verbose=False):
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
        if directed:
            print("full_network", self.G.number_of_nodes(), self.G.number_of_edges()) if verbose else None
        else:
            print("full_network", self.G_u.number_of_nodes(), self.G_u.number_of_edges()) if verbose else None

        y_label = filter_y_multilabel(self, y_label=stratify_label, min_count=n_splits)
        train_nodes, test_nodes = stratify_train_test(y_label, n_splits=n_splits)

        network_train, network_test = split_graph(self, directed=directed, train_nodes=train_nodes,
                                                  test_nodes=test_nodes)
        self.training = copy.copy(self)
        self.training.annotations = self.annotations
        self.training.node_list = [node for node in self.node_list if node in network_train.nodes()]
        self.training.node_list = list(OrderedDict.fromkeys(self.training.node_list))

        if directed:
            self.training.G = network_train
        else:
            self.training.G_u = network_train

        # Test network
        self.testing = copy.copy(self)
        self.testing.annotations = self.annotations
        self.testing.node_list = list(OrderedDict.fromkeys(test_nodes))
        if directed:
            self.testing.G = network_test
        else:
            self.testing.G_u = network_test

        if directed:
            print("train_network", self.training.G.number_of_nodes(),
                  self.training.G.number_of_edges()) if verbose else None
            print("test_network", self.testing.G.number_of_nodes(),
                  self.testing.G.number_of_edges()) if verbose else None
        else:
            print("train_network", self.training.G_u.number_of_nodes(),
                  self.training.G_u.number_of_edges()) if verbose else None
            print("test_network", self.testing.G.number_of_nodes(),
                  self.testing.G_u.number_of_edges()) if verbose else None

    def get_train_generator(self, generator, **kwargs):
        kwargs['network'] = self.training
        return generator(**kwargs)

    def get_test_generator(self, generator, **kwargs):
        kwargs['network'] = self.testing
        return generator(**kwargs)




def mask_test_edges_by_nodes(network, directed, node_list,
                             test_frac=0.10, val_frac=0.0,
                             seed=0, verbose=False):
    if directed:
        g = network.G.copy()
    else:
        g = network.G_u.copy()
    nodes_dict = network.nodes

    g.remove_nodes_from(list(nx.isolates(g)))
    no_of_edges_before = g.number_of_edges()
    no_of_nodes_before = g.number_of_nodes()

    test_nodes_size = int(len(node_list) * test_frac)
    val_nodes_size = int(len(node_list) * val_frac)

    random.seed(seed)
    # Sample nodes then create a set of edges induced by the sampled nodes
    test_nodes = []
    for node_type, nodes in nodes_dict.items():
        node_type_ratio = len(nodes) / len(node_list)
        test_nodes.extend(random.sample(list(nodes), int(test_nodes_size * node_type_ratio)))
    test_edges = [(u, v, d) for u, v, d in g.edges(test_nodes, data=True) if (u in test_nodes and v in test_nodes)]
    g.remove_nodes_from(test_nodes)
    print("test nodes", len(test_nodes), ", edges", len(test_edges))

    val_nodes = []
    for node_type, nodes in nodes_dict.items():
        node_type_ratio = len(nodes) / len(node_list)
        val_nodes.extend(random.sample(list(nodes), int(val_nodes_size * node_type_ratio)))
    val_edges = [(u, v, d) for u, v, d in g.edges(val_nodes, data=True) if (u in val_nodes and v in val_nodes)]
    g.remove_nodes_from(val_nodes)
    print("val nodes", len(val_nodes), ", edges", len(val_edges))

    print('removed', no_of_edges_before - g.number_of_edges(), "edges, and ",
          no_of_nodes_before - g.number_of_nodes(), "nodes.") if verbose else None

    return g, test_edges, val_edges, test_nodes, val_nodes


def split_graph(network, directed, train_nodes, test_nodes,
                seed=0, verbose=False):
    if directed:
        g_train = network.G.copy()
    else:
        g_train = network.G_u.copy()

    g_train.remove_nodes_from(list(nx.isolates(g_train)))
    no_of_edges_before = g_train.number_of_edges()
    no_of_nodes_before = g_train.number_of_nodes()

    random.seed(seed)
    # Sample nodes then create a set of edges induced by the sampled nodes
    g_test = g_train.subgraph(test_nodes)
    g_train = g_train.subgraph(train_nodes)
    print("test nodes", len(g_test.number_of_nodes()), ", edges", len(g_test.number_of_edges()))

    print('removed', no_of_edges_before - g_train.number_of_edges(), "edges, and ",
          no_of_nodes_before - g_train.number_of_nodes(), "nodes.") if verbose else None

    return g_train, g_test


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
