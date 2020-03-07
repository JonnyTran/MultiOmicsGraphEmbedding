import copy
import random
from abc import abstractmethod
from collections import OrderedDict

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sps
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.model_selection import IterativeStratification

from moge.generator.sequences import SEQUENCE_COL
from moge.network.attributed import AttributedNetwork


def filter_y_multilabel(annotations, y_label="go_id", min_count=2, dropna=False, delimiter="|"):
    if dropna:
        nodes_index = annotations[[SEQUENCE_COL] + y_label].dropna().index
    else:
        nodes_index = annotations[[SEQUENCE_COL]].dropna().index

    labels_filter = get_labels_filter(annotations, nodes_index, y_label, min_count, delimiter)
    print("labels_filtered:", len(labels_filter))

    y_labels = annotations.loc[nodes_index, y_label].str.split(delimiter)
    y_labels = y_labels.map(
        lambda go_terms: [item for item in go_terms if item not in labels_filter] if type(go_terms) == list else [])

    return y_labels, labels_filter


def get_labels_filter(annotations, node_list, y_label, min_count, delimiter="|"):
    label_counts = {}
    for items in annotations.loc[node_list, y_label].str.split(delimiter):
        if type(items) != list: continue
        for item in items:
            label_counts[item] = label_counts.setdefault(item, 0) + 1
    label_counts = pd.Series(label_counts)
    labels_filter = label_counts[label_counts < min_count].index
    return labels_filter


def stratify_train_test(y_label, n_splits=10, seed=42):
    y_label_bin = MultiLabelBinarizer().fit_transform(y_label)

    k_fold = IterativeStratification(n_splits=n_splits, order=1, random_state=seed)
    for train, test in k_fold.split(y_label.index.to_list(), sps.lil_matrix(y_label_bin)):
        train_nodes = list(y_label.index[train])
        test_nodes = list(y_label.index[test])
        return train_nodes, test_nodes

class TrainTestSplit():
    def __init__(self: AttributedNetwork) -> None:
        self.training = None
        self.testing = None
        self.validation = None

    @abstractmethod
    def split_train_test_edges(self: AttributedNetwork,
                               node_list=None,
                               databases=["miRTarBase", "BioGRID", "lncRNome", "lncBase", "LncReg"],
                               test_frac=.05, val_frac=.01, seed=0, verbose=False):
        raise NotImplementedError()

    @abstractmethod
    def split_train_test_nodes(self: AttributedNetwork, node_list,
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
        raise NotImplementedError()

    @abstractmethod
    def split_train_test_stratified(self: AttributedNetwork, stratify_label: str, stratify_omic=True, n_splits=5,
                                    dropna=False, seed=42, verbose=False):
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
        raise NotImplementedError()

    def set_val_annotations(self, val_nodes=None):
        self.validation = copy.copy(self)
        self.validation.annotations = self.annotations
        if val_nodes:
            self.validation.node_list = list(OrderedDict.fromkeys(val_nodes))
        self.validation.feature_transformer = self.feature_transformer

    def set_testing_annotations(self, test_nodes):
        self.testing = copy.copy(self)
        self.testing.annotations = self.annotations
        if test_nodes:
            self.testing.node_list = list(OrderedDict.fromkeys(test_nodes))
        self.testing.feature_transformer = self.feature_transformer

    def set_training_annotations(self, nodelist):
        self.training = copy.copy(self)
        self.training.annotations = self.annotations
        if nodelist:
            self.training.node_list = list(OrderedDict.fromkeys([node for node in self.node_list if node in nodelist]))
        self.training.feature_transformer = self.feature_transformer

    def get_train_generator(self, generator, **kwargs):
        kwargs['network'] = self.training
        return generator(**kwargs)

    def get_test_generator(self, generator, **kwargs):
        kwargs['network'] = self.testing
        return generator(**kwargs)


def mask_test_edges_by_nodes(network, directed, node_list, test_frac=0.10, val_frac=0.0,
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


def split_graph(network, directed, train_nodes, test_nodes, verbose=False):
    if directed:
        g = network.G.copy()
    else:
        g = network.G_u.copy()

    g.remove_nodes_from(list(nx.isolates(g)))
    no_of_edges_before = g.number_of_edges()
    no_of_nodes_before = g.number_of_nodes()

    g_test = g.subgraph(test_nodes)
    print("test nodes", g_test.number_of_nodes(), ", edges", g_test.number_of_edges())
    g_train = g.subgraph(train_nodes)
    print("train nodes", g_train.number_of_nodes(), ", edges", g_train.number_of_edges())

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
