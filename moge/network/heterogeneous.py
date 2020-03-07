import networkx as nx
import scipy.sparse as sp

from moge.evaluation.utils import sample_edges
from moge.network.attributed import AttributedNetwork, MODALITY_COL
from moge.network.semantic_similarity import *
from moge.network.train_test_split import TrainTestSplit, mask_test_edges, mask_test_edges_by_nodes, \
    filter_y_multilabel, \
    split_network_by_nodes, stratify_train_test

UNDIRECTED = False
DIRECTED = True

EPSILON = 1e-16


class HeterogeneousNetwork(AttributedNetwork, TrainTestSplit):
    def __init__(self, multiomics: MultiOmics, modalities=None, annotations=True):
        """
        This class manages a networkx multiplex graph consisting of heterogeneous gene nodes, node attributes, and heterogeneous edge types.

        :param multiomics: The MultiOmics (OpenOmics) data to import.
        :param modalities: Default None, import all modalities. A list of omics data to import (e.g. ["MessengerRNA", "LncRNA"]). Each modality represents a
            node type in a heterogeneous network.
        """
        if modalities:
            self.modalities = modalities
        else:
            self.modalities = self.multiomics.omics_list

        self.G = nx.DiGraph()
        self.G_u = nx.Graph()

        networks = {DIRECTED: self.G,
                    UNDIRECTED: self.G_u}

        super(HeterogeneousNetwork, self).__init__(networks=networks, multiomics=multiomics,
                                                   annotations=annotations, )

    def process_network(self):
        self.nodes = {}
        self.node_to_modality = {}

        for modality in self.modalities:
            for network in self.networks.values():
                network.add_nodes_from(self.multiomics[modality].get_genes_list(), modality=modality)

            self.nodes[modality] = self.multiomics[modality].get_genes_list()

            for gene in self.multiomics[modality].get_genes_list():
                if modality in self.node_to_modality:
                    self.node_to_modality[gene] = [self.node_to_modality[gene], ] + [modality]
                else:
                    self.node_to_modality[gene] = modality
            print(modality, " nodes:", len(self.nodes[modality]))
        print("Total nodes:", len(self.get_node_list()))

    def add_edges(self, edgelist, directed, **kwargs):
        if directed:
            self.G.add_edges_from(edgelist, type="d", **kwargs)
        else:
            self.G_u.add_edges_from(edgelist, type="u", **kwargs)
        print(len(edgelist), "edges added.")

    def import_edgelist_file(self, file, directed):
        if directed:
            self.G.add_edges_from(nx.read_edgelist(file, data=True, create_using=nx.DiGraph()).edges(data=True))
        else:
            self.G_u.add_edges_from(nx.read_edgelist(file, data=True, create_using=nx.Graph()).edges(data=True))

    def get_adjacency_matrix(self, edge_types: list, node_list=None, databases=None, sample_negative=0.0):
        """
        Returns an adjacency matrix from edges with type specified in :param edge_types: and nodes specified in
        :param edge_types: A list of edge types letter codes in ["d", "u", "u_n"]
        :param node_list: A list of node names
        :return: A csr_matrix sparse adjacency matrix
        """
        if node_list is None:
            node_list = self.node_list

        if type(edge_types) == list:
            if "d" in edge_types:
                is_directed = True
            elif "u" in edge_types or "u_n" in edge_types:
                is_directed = False
        elif type(edge_types) == str:
            if "d" == edge_types:
                is_directed = True
            elif "u" == edge_types or "u_n" == edge_types:
                is_directed = False

        if databases is not None and is_directed:
            edge_list = [(u, v) for u, v, d in self.G.edges(nbunch=node_list, data=True) if
                         'database' in d and d['database'] in databases]
            adj = nx.adjacency_matrix(nx.DiGraph(incoming_graph_data=edge_list), nodelist=node_list)
        elif is_directed:
            adj = nx.adjacency_matrix(self.G.subgraph(nodes=node_list), nodelist=node_list)
            if sample_negative:
                adj = self.sample_random_negative_edges(adj.astype(float), negative_sampling_ratio=sample_negative)
        elif not is_directed and (("u" in edge_types and "u_n" in edge_types) or "u" in edge_types):
            adj = nx.adjacency_matrix(self.G_u.subgraph(nodes=node_list), nodelist=node_list)
        elif not is_directed and ("u_n" == edge_types or "u_n" in edge_types):
            edge_list = [(u, v) for u, v, d in self.G_u.edges(nbunch=node_list, data=True) if
                         d['type'] in edge_types]
            adj = nx.adjacency_matrix(nx.Graph(incoming_graph_data=edge_list), nodelist=node_list)

        # Eliminate self-edges
        adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
        return adj.astype(float)

    def get_graph_laplacian(self, edge_types: list, node_list=None, databases=None):
        """
        Returns an adjacency matrix from edges with type specified in :param edge_types: and nodes specified in
        :param edge_types: A list of edge types letter codes in ["d", "u", "u_n"]
        :param node_list: A list of node names
        :return: A csr_matrix sparse adjacency matrix
        """
        if "d" in edge_types:
            directed = True
        elif "u" in edge_types:
            directed = False
        else:
            raise Exception("edge_types must be either 'd' or 'u'")

        if hasattr(self, "laplacian_adj"):
            laplacian_adj = self.laplacian_adj

        else:
            if databases is not None:
                if directed:
                    edge_list = [(u, v) for u, v, d in self.G.edges(nbunch=self.node_list, data=True) if
                                 'database' in d and d['database'] in databases]
                    laplacian_adj = nx.directed_laplacian_matrix(nx.DiGraph(incoming_graph_data=edge_list),
                                                                 nodelist=self.node_list)
                else:
                    edge_list = [(u, v) for u, v, d in self.G_u.edges(nbunch=self.node_list, data=True) if
                                 d['type'] in edge_types]
                    laplacian_adj = nx.normalized_laplacian_matrix(nx.Graph(incoming_graph_data=edge_list),
                                                                   nodelist=self.node_list)
            elif directed:
                laplacian_adj = nx.directed_laplacian_matrix(self.G.subgraph(nodes=self.node_list),
                                                             nodelist=self.node_list)
            elif not directed:
                laplacian_adj = nx.normalized_laplacian_matrix(self.G_u.subgraph(nodes=self.node_list),
                                                               nodelist=self.node_list)
            else:
                raise Exception()
            self.laplacian_adj = laplacian_adj

        if node_list is None or node_list == self.node_list:
            return laplacian_adj
        elif set(node_list) < set(self.node_list):
            return self.slice_adj(laplacian_adj, node_list, None)
        elif not (set(node_list) < set(self.node_list)):
            raise Exception("A node in node_l is not in self.node_list.")

        return laplacian_adj

    def get_edge(self, i, j, is_directed=True):
        if is_directed:
            return self.G.get_edge_data(i, j)
        else:
            return self.G_u.get_edge_data(i, j)

    def sample_random_negative_edges(self, pos_adj, negative_sampling_ratio):
        pos_rows, pos_cols = pos_adj.nonzero()
        Ed_count = len(pos_rows)
        sample_neg_count = int(Ed_count * negative_sampling_ratio)

        neg_rows, neg_cols = np.where(pos_adj.todense() == 0)
        sample_indices = np.random.choice(neg_rows.shape[0], sample_neg_count, replace=False)
        pos_adj[neg_rows[sample_indices], neg_cols[sample_indices]] = EPSILON
        assert pos_adj.count_nonzero() > Ed_count, "Did not add any sampled negative edges {}+{} > {}".format(
            pos_adj.count_nonzero(), sample_neg_count, Ed_count)
        return pos_adj

    def get_edgelist(self, node_list, inclusive=True, databases=None):
        if databases is not None:
            edgelist = [(u, v) for u, v, d in self.G.edges(nbunch=node_list, data=True) if
                        'database' in d and d['database'] in databases]
        else:
            edgelist = self.G.edges(nbunch=node_list, data=False)

        if inclusive:
            edgelist = [(u, v) for u, v in edgelist if (u in node_list and v in node_list)]

        return edgelist

    def add_sampled_undirected_negative_edges(self, n_edges, modalities=[]):
        nodes_A = self.nodes[modalities[0]]
        nodes_B = self.nodes[modalities[1]]
        edges_ebunch = sample_edges(nodes_A, nodes_B, n_edges=n_edges, edge_type="u_n")

        print("Number of negative sampled edges between", modalities, "added:", len(edges_ebunch))
        self.G_u.add_edges_from(edges_ebunch)

    def get_subgraph(self, modalities=["MicroRNA", "LncRNA", "MessengerRNA"], edge_type="d"):
        if modalities == None:
            modalities = self.modalities

        nodes = []
        for modality in modalities:
            nodes.extend(self.nodes[modality])

        if edge_type == "d":
            return self.G.subgraph(nodes)  # returned subgraph is not mutable
        elif edge_type == "u":
            return self.G_u.subgraph(nodes)  # returned subgraph is not mutable

    def remove_extra_nodes(self):
        self.G = self.get_subgraph(self.modalities).copy()

    def remove_edges_from(self, edgelist, is_directed):
        if is_directed:
            self.G.remove_edges_from(edgelist)
        else:
            self.G_u.remove_edges_from(edgelist)

    def get_non_zero_degree_nodes(self, is_directed):
        if is_directed:
            return [k for k, v in self.G.degree() if v > 0]
        else:
            return [k for k, v in self.G_u.degree() if v > 0]

    def split_edges(self: AttributedNetwork, directed: bool,
                    node_list=None,
                    databases=["miRTarBase", "BioGRID", "lncRNome", "lncBase", "LncReg"],
                    test_frac=.05, val_frac=.01, seed=0, verbose=False):
        if directed:
            print("full_network directed", self.G.number_of_nodes(), self.G.number_of_edges()) if verbose else None
        else:
            print("full_network undirected", self.G_u.number_of_nodes(),
                  self.G_u.number_of_edges()) if verbose else None

        if directed:
            network_train = self.G.copy()
        else:
            network_train = self.G_u.copy()

        test_edges, val_edges = mask_test_edges(self,
                                                node_list=node_list,
                                                databases=databases,
                                                test_frac=test_frac, val_frac=val_frac, seed=seed, verbose=verbose)
        network_train.remove_edges_from(test_edges)
        network_train.remove_edges_from(val_edges)

        self.set_training_data(network_train.nodes())
        if directed:
            self.training.G = network_train
        else:
            self.training.G_u = network_train

        self.set_testing_data(None)
        if directed:
            self.testing.G = nx.from_edgelist(edgelist=test_edges, create_using=nx.DiGraph)
        else:
            self.testing.G_u = nx.from_edgelist(edgelist=test_edges, create_using=nx.Graph)

        if val_frac > 0:
            self.set_val_data(None)
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

    def split_nodes(self, directed: bool, node_list,
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
        self.set_training_annotations(network_train.nodes())
        if directed:
            self.training.G = network_train
        else:
            self.training.G_u = network_train

        self.set_testing_annotations(test_nodes)
        if directed:
            self.testing.G = nx.DiGraph()
            self.testing.G.add_nodes_from(test_nodes)
            self.testing.G.add_edges_from(test_edges)
        else:
            self.testing.G_u = nx.Graph()
            self.testing.G_u.add_nodes_from(test_nodes)
            self.testing.G_u.add_edges_from(test_edges)

        if val_frac > 0:
            self.set_val_annotations(val_nodes)
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

    def split_stratified(self, directed: bool, stratify_label: str, stratify_omic=True, n_splits=5,
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
        if directed:
            print("full_network", self.G.number_of_nodes(), self.G.number_of_edges()) if verbose else None
        else:
            print("full_network", self.G_u.number_of_nodes(), self.G_u.number_of_edges()) if verbose else None

        y_label, _ = filter_y_multilabel(annotations=self.annotations, y_label=stratify_label, min_count=n_splits,
                                         dropna=dropna, delimiter=self.delimiter)
        if stratify_omic:
            y_omic = self.annotations.loc[y_label.index, MODALITY_COL].str.split("|")
            y_label = y_label + y_omic

        train_nodes, test_nodes = stratify_train_test(y_label=y_label, n_splits=n_splits, seed=seed)

        if directed:
            g = self.G.copy()
        else:
            g = self.G_u.copy()
        network_train, network_test = split_network_by_nodes(g, train_nodes=train_nodes, test_nodes=test_nodes,
                                                             verbose=verbose)

        self.set_training_annotations(train_nodes)
        if directed:
            self.training.G = network_train
        else:
            self.training.G_u = network_train

        # Test network
        self.set_testing_annotations(test_nodes)
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
