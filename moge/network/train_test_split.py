import pprint
import random
from abc import abstractmethod
from collections import defaultdict
from typing import List, Tuple, Dict, Any, Set

import networkx as nx
import numpy as np
import pandas as pd
import tqdm
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from logzero import logger
from networkx.classes.reportviews import EdgeView
from sklearn.preprocessing import MultiLabelBinarizer

from moge.model.utils import tensor_sizes


def stratify_train_test(y_label: pd.DataFrame, test_size: float, seed=42):
    y_label_bin = MultiLabelBinarizer().fit_transform(y_label)

    stratify = MultilabelStratifiedShuffleSplit(test_size=test_size, random_state=0)

    for train, test in stratify.split(y_label.index.to_list(), y_label_bin):
        print("train", len(train), "test", len(test))
        train_nodes = list(y_label.index[train])
        test_nodes = list(y_label.index[test])
        yield train_nodes, test_nodes


class TrainTestSplit():
    nodes: pd.Series
    networks: Dict[str, nx.Graph]

    def __init__(self) -> None:
        self.training = None
        self.testing = None
        self.validation = None

        self.train_nodes = defaultdict(set)
        self.valid_nodes = defaultdict(set)
        self.test_nodes = defaultdict(set)

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

    def update_traintest_nodes_set(self, train_nodes: Dict[str, Set[str]], valid_nodes: Dict[str, Set[str]],
                                   test_nodes: Dict[str, Set[str]], verbose=False) -> None:
        """
        Given a subset of nodes in train/valid/test, mark all nodes in the HeteroNetwork to be either train/valid/test.
        Args:
            train_nodes ():
            valid_nodes ():
            test_nodes ():

        Returns:

        """
        logger.info(pprint.pformat(tensor_sizes(
            dict(train_nodes=train_nodes, valid_nodes=valid_nodes, test_nodes=test_nodes)))) if verbose else None

        train_nodes, valid_nodes, test_nodes = defaultdict(set, train_nodes), \
                                               defaultdict(set, valid_nodes), \
                                               defaultdict(set, test_nodes)
        if hasattr(self, "train_nodes"):
            train_nodes = {ntype: self.train_nodes[ntype].union(train_nodes[ntype]) \
                           for ntype in set(self.train_nodes.keys()).union(train_nodes.keys())}
        if hasattr(self, "valid_nodes"):
            valid_nodes = {ntype: self.valid_nodes[ntype].union(valid_nodes[ntype]) \
                           for ntype in set(self.valid_nodes.keys()).union(valid_nodes.keys())}
        if hasattr(self, "test_nodes"):
            test_nodes = {ntype: self.test_nodes[ntype].union(test_nodes[ntype]) \
                          for ntype in set(self.test_nodes.keys()).union(test_nodes.keys())}

        self.train_nodes, self.valid_nodes, self.test_nodes = defaultdict(set, train_nodes), \
                                                              defaultdict(set, valid_nodes), \
                                                              defaultdict(set, test_nodes)

    def set_edge_traintest_mask(self, train_nodes: Dict[str, Set[str]] = None, valid_nodes: Dict[str, Set[str]] = None,
                                test_nodes: Dict[str, Set[str]] = None,
                                exclude_metapaths: List[Tuple[str, str, str]] = None):
        if train_nodes is None:
            train_nodes = self.train_nodes
        if valid_nodes is None:
            valid_nodes = self.valid_nodes
        if test_nodes is None:
            test_nodes = self.test_nodes

        # Set train/valid/test mask of edges on hetero graph if they're incident to the train/valid/test nodes
        for metapath, nxgraph in tqdm.tqdm(self.networks.items(), desc=f"Set train/valid/test_mask on edge types"):
            if exclude_metapaths is not None and metapath in exclude_metapaths:
                continue
            edge_attrs = self.get_all_edges_mask(nxgraph.edges, metapath=metapath, train_nodes=train_nodes,
                                                 valid_nodes=valid_nodes, test_nodes=test_nodes)
            nx.set_edge_attributes(nxgraph, edge_attrs)

        # Add non-incident nodes to the train nodes, since they do not belong in the valid_nodes or test_nodes
        incident_nodes = {ntype: set(train_nodes[ntype] if ntype in train_nodes else []) |
                                 set(valid_nodes[ntype] if ntype in valid_nodes else []) |
                                 set(test_nodes[ntype] if ntype in test_nodes else []) \
                          for ntype in self.nodes.keys()}
        nonincident_nodes = {ntype: set(nodelist).difference(incident_nodes[ntype]) \
                             for ntype, nodelist in self.nodes.items()}
        nonincident_nodes = defaultdict(set, {k: v for k, v in nonincident_nodes.items() if v})

        train_nodes = {train_nodes[ntype].union(nonincident_nodes[ntype]) \
                       for ntype in set(train_nodes.keys()).union(nonincident_nodes.keys())}
        valid_nodes = {valid_nodes[ntype].union(nonincident_nodes[ntype]) \
                       for ntype in set(valid_nodes.keys()).union(nonincident_nodes.keys())}
        test_nodes = {test_nodes[ntype].union(nonincident_nodes[ntype]) \
                      for ntype in set(test_nodes.keys()).union(nonincident_nodes.keys())}

        # Set train/valid/test_mask on node types
        for metapath in tqdm.tqdm(self.networks.keys(), desc=f"Set train/valid/test_mask on node types"):
            for nodes_dict in [train_nodes, valid_nodes, test_nodes]:
                node_attr_dict = self.get_node_mask(nodes_dict,
                                                    train=nodes_dict is train_nodes,
                                                    valid=nodes_dict is valid_nodes, test=nodes_dict is test_nodes)
                for mask_name, node_mask in node_attr_dict.items():
                    nx.set_node_attributes(self.networks[metapath], values=node_mask, name=mask_name)

        self.train_valid_test_counts = pd.DataFrame(
            tensor_sizes(train_nodes=train_nodes | self.nodes.drop(train_nodes).map(len).to_dict(),
                         valid_nodes=valid_nodes | self.nodes.drop(valid_nodes).map(len).to_dict(),
                         test_nodes=test_nodes | self.nodes.drop(test_nodes).map(len).to_dict()),
            dtype='int').T

    def get_node_mask(self, node_dict: Dict[str, Set[str]], train=False, valid=False, test=False) \
            -> Dict[str, Dict[str, Dict[str, Any]]]:
        mask = {'train_mask': train, 'valid_mask': valid, 'test_mask': test}

        node_attrs = {key: {node: mask[key] \
                            for ntype, node_list in node_dict.items() \
                            for node in node_list} \
                      for key in ["train_mask", "valid_mask", "test_mask"]}

        return node_attrs

    def get_all_edges_mask(self, edgelist: EdgeView, metapath: Tuple[str, str, str],
                           train_nodes: Dict[str, Set[str]], valid_nodes: Dict[str, Set[str]],
                           test_nodes: Dict[str, Set[str]]) \
            -> Dict[Tuple[str, str, str], Dict[str, Dict[str, Any]]]:
        """

        Args:
            edgelist ():
            metapath ():
            train_nodes ():
            valid_nodes ():
            test_nodes ():

        Returns:

        """
        train_nodes = defaultdict(set, train_nodes)
        valid_nodes = defaultdict(set, valid_nodes)
        test_nodes = defaultdict(set, test_nodes)

        def _get_edge_mask(u: str, v: str) -> Dict[str, Any]:
            head_type, tail_type = metapath[0], metapath[-1]
            train = u in train_nodes[head_type] and v in train_nodes[tail_type]
            valid = u in valid_nodes[head_type] or v in valid_nodes[tail_type]
            test = u in test_nodes[head_type] or v in test_nodes[tail_type]
            if not valid and not test and not train:
                train = valid = test = True
            return {'train_mask': train, 'valid_mask': valid, 'test_mask': test}

        edge_attrs = {edge_tup: _get_edge_mask(edge_tup[0], edge_tup[1]) \
                      for edge_tup, _ in edgelist.items()}

        return edge_attrs


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
