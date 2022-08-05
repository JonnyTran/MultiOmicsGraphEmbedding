import random
from abc import abstractmethod
from typing import List, Tuple, Dict, Any

import networkx as nx
import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.preprocessing import MultiLabelBinarizer


def stratify_train_test(y_label: pd.DataFrame, test_size: float, seed=42):
    y_label_bin = MultiLabelBinarizer().fit_transform(y_label)

    stratify = MultilabelStratifiedShuffleSplit(test_size=test_size, random_state=0)

    for train, test in stratify.split(y_label.index.to_list(), y_label_bin):
        print("train", len(train), "test", len(test))
        train_nodes = list(y_label.index[train])
        test_nodes = list(y_label.index[test])
        yield train_nodes, test_nodes

class TrainTestSplit():
    def __init__(self) -> None:
        self.training = None
        self.testing = None
        self.validation = None

    @abstractmethod
    def split_edges(self, node_list=None, test_frac=.05, val_frac=.01, seed=0, verbose=False):
        raise NotImplementedError()

    @abstractmethod
    def split_nodes(self, node_list, test_frac=.05, val_frac=.01, seed=0, verbose=False):
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
    def split_stratified(self, stratify_label: str, stratify_omic=True, n_splits=5, dropna=False, seed=42,
                         verbose=False):
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

    def get_train_generator(self, generator, split_idx=None, **kwargs):
        if not hasattr(self, "training"):
            raise Exception("Must run split_train_test on the network first.")

        if split_idx is not None:
            assert split_idx < len(self.train_test_splits)
            node_list = self.train_test_splits[split_idx][0]
        else:
            node_list = self.training.node_list

        kwargs['network'] = self
        kwargs['node_list'] = node_list

        gen_inst = generator(**kwargs)
        self.tokenizer = gen_inst.seq_tokenizer

        return gen_inst

    def get_test_generator(self, generator, split_idx=None, **kwargs):
        if not hasattr(self, "testing"):
            raise Exception("Must run split_train_test on the network first.")

        if split_idx is not None:
            assert split_idx < len(self.train_test_splits)
            node_list = self.train_test_splits[split_idx][1]
        else:
            node_list = self.testing.node_list

        kwargs['network'] = self
        kwargs['node_list'] = node_list

        # A feature to ensure the test data has the same tokenizer as the train data
        if hasattr(self, "tokenizer"):
            kwargs["tokenizer"] = self.tokenizer
        gen_inst = generator(**kwargs)

        return gen_inst

    def label_edge_trainvalidtest(self, edges: List[Tuple[str, str]], train=False, valid=False, test=False) \
            -> List[Tuple[str, str, Dict[str, Any]]]:

        train_mask = np.where(train, np.ones(len(edges), dtype=bool), np.zeros(len(edges), dtype=bool))
        valid_mask = np.where(valid, np.ones(len(edges), dtype=bool), np.zeros(len(edges), dtype=bool))
        test_mask = np.where(test, np.ones(len(edges), dtype=bool), np.zeros(len(edges), dtype=bool))
        edge_attr = [{'train_mask': train, 'valid_mask': valid, 'test_mask': test} \
                     for train, valid, test in zip(train_mask, valid_mask, test_mask)]
        edges = [(u, v, d) for (u, v), d in zip(edges, edge_attr)]
        return edges

    def label_node_trainvalidtest(self, node_dict: Dict[str, List[str]], train=False, valid=False, test=False) \
            -> Dict[str, Dict[str, Dict[str, Any]]]:
        mask = {'train_mask': train, 'valid_mask': valid, 'test_mask': test}
        # node_attr_dict = {node: {'train_mask': train, 'valid_mask': valid, 'test_mask': test} \
        #                   for ntype, node_list in node_dict.items() for node in node_list}
        node_attr_dict = {key: {node: mask[key] \
                                for ntype, node_list in node_dict.items() \
                                for node in node_list} \
                          for key in ["train_mask", "valid_mask", "test_mask"]}

        return node_attr_dict

    def set_node_traintestvalid_mask(self, train_nodes: Dict[str, List[str]], valid_nodes: Dict[str, List[str]],
                                     test_nodes: Dict[str, List[str]]) \
            -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, List[str]]]:
        incident_nodes = {ntype: list(set(train_nodes[ntype] if ntype in train_nodes else []) |
                                      set(valid_nodes[ntype] if ntype in valid_nodes else []) |
                                      set(test_nodes[ntype] if ntype in test_nodes else [])) \
                          for ntype, nodelist in self.nodes.items()}
        nonincident_nodes = {ntype: np.setdiff1d(nodelist, incident_nodes[ntype]) \
                             for ntype, nodelist in self.nodes.items()}

        # Add non-incident nodes to the train nodes, since they do not belong in the valid_nodes or test_nodes
        train_nodes = {ntype: np.union1d(nids, nonincident_nodes[ntype]) \
                       for ntype, nids in train_nodes.items()}
        train_nodes = train_nodes | {ntype: train_ids for ntype, train_ids in nonincident_nodes.items() \
                                     if ntype not in train_nodes}
        valid_nodes = {ntype: np.setdiff1d(nids, train_nodes[ntype]) \
                       for ntype, nids in valid_nodes.items()}
        test_nodes = {ntype: np.setdiff1d(np.setdiff1d(nids, train_nodes[ntype]), valid_nodes[ntype]) \
                      for ntype, nids in test_nodes.items()}

        for metapath in self.networks.keys():
            for nodes_dict in [train_nodes, valid_nodes, test_nodes]:
                node_attr_dict = self.label_node_trainvalidtest(nodes_dict, train=nodes_dict is train_nodes,
                                                                valid=nodes_dict is valid_nodes,
                                                                test=nodes_dict is test_nodes)
                for mask_name, node_mask in node_attr_dict.items():
                    nx.set_node_attributes(self.networks[metapath], values=node_mask, name=mask_name)

        print("train nodes", sum(len(nids) for nids in train_nodes.values()),
              "valid nodes", sum(len(nids) for nids in valid_nodes.values()),
              "test nodes", sum(len(nids) for nids in test_nodes.values()))

        return train_nodes, valid_nodes, test_nodes

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


def split_network_by_nodes(g, train_nodes, test_nodes, verbose=False):
    g.remove_nodes_from(list(nx.isolates(g)))
    no_of_edges_before = g.number_of_edges()
    no_of_nodes_before = g.number_of_nodes()

    g_test = g.subgraph(test_nodes)
    print("test nodes", g_test.number_of_nodes(), ", edges", g_test.number_of_edges()) if verbose else None
    g_train = g.subgraph(train_nodes)
    print("train nodes", g_train.number_of_nodes(), ", edges", g_train.number_of_edges()) if verbose else None

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
