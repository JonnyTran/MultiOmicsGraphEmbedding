from collections import OrderedDict

import networkx as nx
import scipy.sparse as sp

from moge.evaluation.utils import sample_edges
from moge.network.attributed import AttributedNetwork
from moge.network.omics_distance import *
from moge.network.train_test_split import NetworkTrainTestSplit

EPSILON = 1e-16


class HeterogeneousNetwork(AttributedNetwork, NetworkTrainTestSplit):
    def __init__(self, modalities: list, multi_omics_data: MultiOmics, process_annotations=True):
        """
        This class manages a networkx graph consisting of heterogeneous gene nodes, and heterogeneous edge types.

        :param modalities: A list of omics data to import (e.g. ["GE", "LNC"]). Each modalities has a list of genes
        :param multi_omics_data: The multiomics data to import
        """
        self.multi_omics_data = multi_omics_data
        self.G = nx.DiGraph()
        self.G_u = nx.Graph()
        self.modalities = modalities

        self.preprocess_graph()

        super(HeterogeneousNetwork, self).__init__(multi_omics_data=multi_omics_data,
                                                   process_annotations=process_annotations)

        self.node_list = self.get_node_list()

    def get_node_list(self):
        node_list = list(OrderedDict.fromkeys(list(self.G.nodes) + list(self.G_u.nodes)))
        return node_list

    def preprocess_graph(self):
        self.nodes = {}
        self.node_to_modality = {}

        bad_nodes = [node for node in self.get_node_list()
                     if node is None or node == np.nan or \
                     type(node) != str or \
                     node == "" or " " in node \
                     ]
        self.G.remove_nodes_from(bad_nodes)
        self.G_u.remove_nodes_from(bad_nodes)

        for modality in self.modalities:
            self.G.add_nodes_from(self.multi_omics_data[modality].get_genes_list(), modality=modality)
            self.G_u.add_nodes_from(self.multi_omics_data[modality].get_genes_list(), modality=modality)
            self.nodes[modality] = self.multi_omics_data[modality].get_genes_list()

            for gene in self.multi_omics_data[modality].get_genes_list():
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
        elif databases is not None:
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
            laplacian_adj = nx.directed_laplacian_matrix(self.G.subgraph(nodes=self.node_list), nodelist=self.node_list)
        elif not directed:
            laplacian_adj = nx.normalized_laplacian_matrix(self.G_u.subgraph(nodes=self.node_list),
                                                           nodelist=self.node_list)
        else:
            raise Exception()

        if node_list is None or node_list == self.node_list:
            self.laplacian_adj = laplacian_adj
            return laplacian_adj
        elif set(node_list) < set(self.node_list):
            return self._select_adj_indices(laplacian_adj, node_list, None)
        elif not (set(node_list) < set(self.node_list)):
            raise Exception("A node in node_l is not in self.node_list.")

        return laplacian_adj

    def _select_adj_indices(self, adj, node_list_A, node_list_B=None):
        if node_list_B is None:
            idx = [self.node_list.index(node) for node in node_list_A]
            return adj[idx, :][:, idx]
        else:
            idx_A = [self.node_list.index(node) for node in node_list_A]
            idx_B = [self.node_list.index(node) for node in node_list_B]
            return adj[idx_A, :][:, idx_B]

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

    def get_edgelist(self, node_list, databases=None, inclusive=True):
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

    def add_sampled_undirected_negative_edges_from_correlation(self, modalities=[], correlation_threshold=0.2,
                                                               histological_subtypes=[],
                                                               pathologic_stages=[]):
        """
        Sample edges with experssion values absolute-value correlations near zero, indicating no relationships

        :param modalities:
        :param correlation_threshold:
        :return:
        """
        nodes_A = self.nodes[modalities[0]]
        nodes_B = self.nodes[modalities[1]]
        node_list = [node for node in self.node_list if node in nodes_A or node in nodes_B]

        # Filter similarity adj by correlation
        correlation_dist = compute_expression_correlation_dists(self.multi_omics_data, modalities=modalities,
                                                                node_list=node_list,
                                                                histological_subtypes=histological_subtypes,
                                                                pathologic_stages=pathologic_stages,
                                                                squareform=True)
        correlation_dist = 1 - correlation_dist
        correlation_dist = np.abs(correlation_dist)

        similarity_filtered = np.triu(correlation_dist <= correlation_threshold, k=1)  # A True/False matrix
        sim_edgelist_ebunch = [(node_list[x], node_list[y], correlation_dist.iloc[x, y]) for x, y in
                               zip(*np.nonzero(similarity_filtered))]
        print(sim_edgelist_ebunch[0:10])
        self.G.add_weighted_edges_from(sim_edgelist_ebunch, type="u_n")
        print(len(sim_edgelist_ebunch), "undirected positive edges (type='u') added.")


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
