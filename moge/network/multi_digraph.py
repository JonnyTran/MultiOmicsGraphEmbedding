from argparse import Namespace

import networkx as nx
from moge.criterion.utils import sample_edges
from moge.network.attributed import AttributedNetwork, MODALITY_COL
from moge.network.similarity import *
from moge.network.train_test_split import TrainTestSplit, mask_test_edges, mask_test_edges_by_nodes, \
    stratify_train_test
from moge.network.utils import parse_labels

from openomics import MultiOmics

UNDIRECTED = False
DIRECTED = True

EPSILON = 1e-16


class MultiDigraphNetwork(AttributedNetwork, TrainTestSplit):
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

        super(MultiDigraphNetwork, self).__init__(networks=networks, multiomics=multiomics,
                                                  annotations=annotations, )

    def process_network(self):
        self.nodes = {}
        self.node_to_modality = {}

        for modality in self.modalities:
            for network in self.networks.values():
                network.add_nodes_from(self.multiomics[modality].get_genes_list(), modality=modality)

            self.nodes[modality] = self.multiomics[modality].get_genes_list()

            for gene in self.multiomics[modality].get_genes_list():
                self.node_to_modality[gene] = self.node_to_modality.setdefault(gene, []) + [modality, ]
            print(modality, " nodes:", len(self.nodes[modality]))
        print("Total nodes:", len(self.get_all_nodes()))
        self.nodes = pd.Series(self.nodes)
        self.node_to_modality = pd.Series(self.node_to_modality)

    def add_edges(self, edgelist, directed, **kwargs):
        if directed:
            self.G.add_edges_from(edgelist, type="d", **kwargs)
        else:
            self.G_u.add_edges_from(edgelist, type="u", **kwargs)
        print(len(edgelist), "edges added.")

    def import_edgelist_file(self, filepath, directed):
        if directed:
            self.G.add_edges_from(nx.read_edgelist(filepath, data=True, create_using=nx.DiGraph()).edges(data=True))
        else:
            self.G_u.add_edges_from(nx.read_edgelist(filepath, data=True, create_using=nx.Graph()).edges(data=True))

    def get_adjacency_matrix(self, edge_types: list, node_list=None, databases=None, sample_negative=0.0, method="GAT",
                             output="csr"):
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

        # if method == "GAT":
        #     adj = adj + sps.csr_matrix(np.eye(adj.shape[0]))  # Add self-loops

        if output == "csr":
            return adj.astype(float)
        elif output == "coo":
            adj = adj.tocoo(copy=False)
            return np.vstack((adj.row, adj.col)).astype("long")
        elif output == "dense":
            return adj.todense()
        else:
            raise Exception("Output must be one of {csr, coo, dense}")

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
        elif set(node_list) <= set(self.node_list):
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

        y_label = parse_labels(df=self.annotations, column=stratify_label, min_count=n_splits, dropna=dropna,
                               delimiter=self.delimiter)
        if stratify_omic:
            y_omic = self.annotations.loc[y_label.index, MODALITY_COL].str.split("|")
            y_label = y_label + y_omic

        self.train_test_splits = list(stratify_train_test(y_label=y_label, n_splits=n_splits, seed=seed))

        self.training = Namespace()
        self.testing = Namespace()
        self.training.node_list = self.train_test_splits[0][0]
        self.testing.node_list = self.train_test_splits[0][1]

    def get_correlation_edges(self, modality, node_list, threshold=0.7):
        # Filter similarity adj by correlation
        correlations = compute_expression_correlation(self.multiomics, modalities=[modality],
                                                      node_list=node_list, absolute_corr=True,
                                                      return_distance=False,
                                                      squareform=True)

        # Selects positive edges with high affinity in the affinity matrix
        similarity_filtered = np.triu(correlations >= threshold, k=1)  # A True/False matrix
        edgelist = [(node_list[x], node_list[y], {"weight": correlations.iloc[x, y]}) for x, y in
                    zip(*np.nonzero(similarity_filtered))]

        return edgelist

    def add_affinity_edges(self, modality, node_list, features=None, weights=None, nanmean=True,
                           similarity_threshold=0.7, dissimilarity_threshold=0.1,
                           negative_sampling_ratio=2.0, max_positive_edges=None,
                           tissue_expression=False, histological_subtypes=[],
                           pathologic_stages=[],
                           epsilon=EPSILON, tag="affinity"):
        """
        Computes similarity measures between genes within the same modality, and add them as undirected edges to the network if the similarity measures passes the threshold

        :param modality: E.g. ["GE", "MIR", "LNC"]
        :param similarity_threshold: a hard-threshold to select positive edges with affinity value more than it
        :param dissimilarity_threshold: a hard-threshold to select negative edges with affinity value less than
        :param negative_sampling_ratio: the number of negative edges in proportion to positive edges to select
        :param histological_subtypes: the patients' cancer subtype group to calculate correlation from
        :param pathologic_stages: the patient's cancer stage group to calculate correlations from
        """
        annotations = self.multiomics[modality].get_annotations()

        annotation_affinities_df = pd.DataFrame(
            data=compute_annotation_affinities(annotations, node_list=node_list, modality=modality,
                                               correlation_dist=None, nanmean=nanmean,
                                               features=features, weights=weights, squareform=True),
            index=node_list)

        # Selects positive edges with high affinity in the affinity matrix
        similarity_filtered = np.triu(annotation_affinities_df >= similarity_threshold, k=1)  # A True/False matrix
        sim_edgelist_ebunch = [(node_list[x], node_list[y], annotation_affinities_df.iloc[x, y]) for x, y in
                               zip(*np.nonzero(similarity_filtered))]
        # Sample
        if max_positive_edges is not None:
            sample_indices = np.random.choice(a=range(len(sim_edgelist_ebunch)),
                                              size=min(max_positive_edges, len(sim_edgelist_ebunch)), replace=False)
            sim_edgelist_ebunch = [(u, v, d) for i, (u, v, d) in enumerate(sim_edgelist_ebunch) if i in sample_indices]
            self.G_u.add_weighted_edges_from(sim_edgelist_ebunch, type="u", tag=tag)
        else:
            self.G_u.add_weighted_edges_from(sim_edgelist_ebunch, type="u", tag=tag)

        print(len(sim_edgelist_ebunch), "undirected positive edges (type='u') added.")

        # Select negative edges at affinity close to zero in the affinity matrix
        max_negative_edges = int(negative_sampling_ratio * len(sim_edgelist_ebunch))
        dissimilarity_filtered = np.triu(annotation_affinities_df <= dissimilarity_threshold, k=1)

        dissimilarity_index_rows, dissimilarity_index_cols = np.nonzero(dissimilarity_filtered)
        sample_indices = np.random.choice(a=dissimilarity_index_rows.shape[0],
                                          size=min(max_negative_edges, dissimilarity_index_rows.shape[0]),
                                          replace=False)
        # adds 1e-8 to keeps from 0.0 edge weights, which doesn't get picked up in nx.adjacency_matrix()
        dissim_edgelist_ebunch = [(node_list[x], node_list[y], min(annotation_affinities_df.iloc[x, y], epsilon)) for
                                  i, (x, y) in
                                  enumerate(zip(dissimilarity_index_rows[sample_indices],
                                                dissimilarity_index_cols[sample_indices])) if i < max_negative_edges]
        self.G_u.add_weighted_edges_from(dissim_edgelist_ebunch, type="u_n", tag=tag)

        print(len(dissim_edgelist_ebunch), "undirected negative edges (type='u_n') added.")
        return annotation_affinities_df
