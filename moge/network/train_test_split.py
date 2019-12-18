import random

import networkx as nx
import numpy as np

from moge.network.heterogeneous_network import HeterogeneousNetwork


class NetworkTrainTestSplit():
    def __init__(self) -> None:
        self.train_network = None
        self.test_network = None
        self.val_network = None

    def split_train_test_edges(self: HeterogeneousNetwork,
                               node_list, edge_types=["u", "d", "u_n"],
                               databases=["miRTarBase", "BioGRID", "lncRNome", "lncBase", "LncReg"],
                               test_frac=.05, val_frac=.01, seed=0, verbose=True):
        network_train = self.G

        test_edges, val_edges = mask_test_edges(self,
                                                node_list=node_list,
                                                edge_types=edge_types,
                                                databases=databases,
                                                test_frac=test_frac, val_frac=val_frac, seed=seed, verbose=verbose)
        network_train.remove_edges_from(test_edges)
        network_train.remove_edges_from(val_edges)
        print("Removed", len(test_edges), "test, and", len(val_edges), "val, type", edge_types, "edges")

        return network_train, test_edges, val_edges

    def split_train_test_nodes(self: HeterogeneousNetwork,
                               node_list, edge_types=["u", "d", "u_n"],
                               test_frac=.05, val_frac=.01, seed=0, verbose=True):
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
        network_train, test_edges, val_edges, \
        test_nodes, val_nodes = mask_test_edges_by_nodes(self, node_list,
                                                         edge_types=edge_types,
                                                         test_frac=test_frac, val_frac=val_frac, seed=seed,
                                                         verbose=verbose)

        self.G = network_train
        self.node_list = [node for node in self.node_list if node in network_train.nodes()]
        print("validation edges", len(val_edges))
        print("test edges", len(test_edges))

        return self, test_edges, val_edges, test_nodes, val_nodes


def mask_test_edges_by_nodes(network: HeterogeneousNetwork, node_list, edge_types=["u", "d"],
                             test_frac=.1, val_frac=.05,
                             seed=0, verbose=False):
    if verbose == True:
        print('preprocessing...')

    g = network.G.copy()
    nodes_dict = network.nodes

    g.remove_nodes_from(list(nx.isolates(g)))
    no_of_edges_before = g.number_of_edges()
    no_of_nodes_before = g.number_of_nodes()

    test_nodes_size = int(len(node_list) * test_frac)
    val_nodes_size = int(len(node_list) * val_frac)

    test_nodes = []
    test_edges = []
    test_edges_add_back = []  # Edges to retain in the training network (even if nodes are removed)
    for node_type, nodes in nodes_dict.items():
        node_type_ratio = len(nodes) / len(node_list)
        test_nodes.extend(random.sample(nodes, int(test_nodes_size * node_type_ratio)))

    for u, v, d in g.edges(test_nodes, data=True):
        if d["type"] in edge_types:
            test_edges.append((u, v, d))
        else:
            test_edges_add_back.append((u, v, d))

    val_nodes = []
    val_edges = []
    val_edges_add_back = []
    for node_type, nodes in nodes_dict.items():
        node_type_ratio = len(nodes) / len(node_list)
        val_nodes.extend(random.sample(nodes, int(val_nodes_size * node_type_ratio)))

    for u, v, d in g.edges(val_nodes, data=True):
        if d["type"] in edge_types:
            val_edges.append((u, v, d))
        else:
            val_edges_add_back.append((u, v, d))

    g.remove_nodes_from(test_nodes)
    g.remove_nodes_from(val_nodes)

    g.add_edges_from(test_edges_add_back)
    g.add_edges_from(val_edges_add_back)

    if verbose == True:
        print('removed', no_of_edges_before - g.number_of_edges(), "edges, and ",
              no_of_nodes_before - g.number_of_nodes(), "nodes.")

    return g, test_edges, val_edges, test_nodes, val_nodes


def mask_test_edges(network: HeterogeneousNetwork, node_list, edge_types=["u", "d"],
                    databases=["miRTarBase", "BioGRID", "lncRNome", "lncBase", "LncReg"],
                    test_frac=.10, val_frac=.05,
                    seed=0, verbose=False):
    if verbose == True:
        print('preprocessing...')

    g = network.G

    edges_to_remove = [(u, v, d) for u, v, d in g.edges(data=True) if
                       d["type"] in edge_types and d["database"] in databases]
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
    print("test_edges_size", test_edges_size) if verbose else None
    print("val_edges_size", val_edges_size) if verbose else None

    test_edges = edges_to_remove[0: test_edges_size]
    val_edges = edges_to_remove[test_edges_size: test_edges_size + val_edges_size]

    return test_edges, val_edges
