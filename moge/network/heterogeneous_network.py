import dask.dataframe as dd
from dask.threaded import get
import scipy.sparse as sp
import networkx as nx

from moge.network.omics_distance import *


class HeterogeneousNetwork():
    def __init__(self, modalities:list, multi_omics_data:MultiOmicsData):
        """
        This class manages a networkx graph consisting of heterogeneous gene nodes, and heterogeneous edge types.

        :param modalities: A list of omics data to import (e.g. ["GE", "LNC"]). Each modalities has a list of genes
        :param multi_omics_data: The multiomics data to import
        """
        self.modalities = modalities
        self.multi_omics_data = multi_omics_data
        self.G = nx.DiGraph()

        self.preprocess_graph()

    def preprocess_graph(self):
        self.nodes = {}
        self.node_to_modality = {}

        self.node_list = []
        for modality in self.modalities:
            self.G.add_nodes_from(self.multi_omics_data[modality].get_genes_list(), modality=modality)
            self.nodes[modality] = self.multi_omics_data[modality].get_genes_list()

            for gene in self.multi_omics_data[modality].get_genes_list():
                self.node_to_modality[gene] = modality

            print(modality, " nodes:", len(self.nodes[modality]))
            self.node_list.extend(self.multi_omics_data[modality].get_genes_list())

        print("Total nodes:", len(self.node_list))



    def add_directed_edges_from_edgelist(self, edgelist, modalities=None):
        if not (modalities is None):
            source_genes = set([edge[0] for edge in edgelist])
            target_genes = set([edge[1] for edge in edgelist])

            source_genes_matched = set(self.nodes[modalities[0]]) & source_genes
            target_genes_matched = set(self.nodes[modalities[1]]) & target_genes

            print("Adding edgelist with", len(source_genes), "total unique", modalities[0], "genes (source), but only matching", len(source_genes_matched), "nodes")
            print("Adding edgelist with", len(target_genes), "total unique", modalities[1], "genes (target), but only matching", len(target_genes_matched), "nodes")
            print(len(edgelist), "edges added.")

        self.G.add_edges_from(edgelist, type="d")

    def import_edgelist_file(self, file, is_directed):
        if is_directed:
            self.G.add_edges_from(nx.read_edgelist(file, data=True, create_using=nx.DiGraph()).edges(data=True))
        else:
            self.G.add_edges_from(nx.read_edgelist(file, data=True, create_using=nx.Graph()).edges(data=True))

    def get_adjacency_matrix(self, edge_types=["u", "d"], node_list=None):
        """
        Returns an adjacency matrix from edges with type specified in :param edge_types: and nodes specified in
         :param node_list:.

        :param edge_types: A list of edge types letter codes to ex
        :param node_list: A list of node names
        :return: A csr_matrix sparse adjacency matrix
        """
        if node_list == None:
            node_list = self.node_list

        if edge_types == None:
            edge_list = self.G.edges(data=True)
        else:
            if type(edge_types) == list:
                edge_list = [(u, v, d) for u, v, d in self.G.edges(data=True) if d['type'] in edge_types]
            else:
                edge_list = [(u, v, d) for u, v, d in self.G.edges(data=True) if d['type'] == edge_types]

        # Also add reverse edges for undirected edges
        if 'u' == edge_types:
            if type(edge_types) == list:
                edge_list.extend([(v, u, d) for u, v, d in edge_list if d['type'] in edge_types])
            else:
                edge_list.extend([(v, u, d) for u, v, d in edge_list if d['type'] == edge_types])

        adj = nx.adjacency_matrix(nx.DiGraph(incoming_graph_data=edge_list), nodelist=node_list)
        # Eliminate self-edges
        adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
        return adj

    def get_edge(self, i, j):
        return self.G.get_edge_data(i, j)

    def get_subgraph(self, modalities=["MIR", "LNC", "GE"]):
        if modalities==None:
            modalities = self.modalities

        nodes = []
        for modality in modalities:
            nodes.extend(self.nodes[modality])

        return self.G.subgraph(nodes) # returned subgraph is not mutable

    def add_edges_from_nodes_similarity(self, modality, node_list, features=None, similarity_threshold=0.7,
                                        dissimilarity_threshold=0.1, negative_sampling_ratio=2.0):
        """
        Computes similarity measures between genes within the same modality, and add them as undirected edges to the network if the similarity measures passes the threshold

        :param modality: E.g. ["GE", "MIR", "LNC"]
        :param similarity_threshold: a hard-threshold on the similarity measure
        :param data:
        """
        genes_info = self.multi_omics_data[modality].get_genes_info()

        similarity_adj_df = pd.DataFrame(
            compute_annotation_similarity(genes_info, node_list=node_list, modality=modality,
                                          features=features, squareform=True), index=node_list)

        # Selects edges from the affinity matrix
        similarity_filtered = np.triu(similarity_adj_df >= similarity_threshold, k=1) # A True/False matrix
        sim_edgelist_ebunch = [(node_list[x], node_list[y], similarity_adj_df.iloc[x, y]) for x, y in
                               zip(*np.nonzero(similarity_filtered))]
        self.G.add_weighted_edges_from(sim_edgelist_ebunch, type="u")
        print(len(sim_edgelist_ebunch), "undirected positive edges (type='u') added.")

        max_negative_edges = negative_sampling_ratio * len(sim_edgelist_ebunch)
        dissimilarity_filtered = np.triu(similarity_adj_df <= dissimilarity_threshold, k=1)
        # adds 1e-8 to keeps from 0.0 edge weights, which doesn't get picked up in nx.adjacency_matrix()
        dissim_edgelist_ebunch = [(node_list[x], node_list[y], similarity_adj_df.iloc[x, y] + 1e-8) for i, (x, y) in
                                  enumerate(zip(*np.nonzero(dissimilarity_filtered))) if i < max_negative_edges]
        self.G.add_weighted_edges_from(dissim_edgelist_ebunch, type="u_n")

        print(len(dissim_edgelist_ebunch), "undirected negative edges (type='u_n') added.")

    def remove_extra_nodes(self):
        self.G = self.get_subgraph(self.modalities)

    def remove_edges_from(self, edgelist):
        self.G.remove_edges_from(edgelist)


    def set_node_similarity_training_adjacency(self, adj):
        self.adj_similarity_train = adj

    def set_regulatory_edges_training_adjacency(self, adj):
        self.adj_regulatory_train = adj

    def get_non_zero_degree_nodes(self):
        return [k for k, v in self.G.degree() if v > 0]


    def get_combined_gene_info(self, modalities):
        pass
