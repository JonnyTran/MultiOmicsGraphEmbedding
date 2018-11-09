from collections import OrderedDict

import networkx as nx
import scipy.sparse as sp

from moge.evaluation.utils import sample_edges
from moge.network.omics_distance import *


class HeterogeneousNetwork():
    def __init__(self, modalities:list, multi_omics_data:MultiOmicsData, process_genes_info=True):
        """
        This class manages a networkx graph consisting of heterogeneous gene nodes, and heterogeneous edge types.

        :param modalities: A list of omics data to import (e.g. ["GE", "LNC"]). Each modalities has a list of genes
        :param multi_omics_data: The multiomics data to import
        """
        self.modalities = modalities
        self.multi_omics_data = multi_omics_data
        self.G = nx.DiGraph()
        self.affinities = {}

        self.preprocess_graph()
        if process_genes_info:
            self.process_genes_info()

    def preprocess_graph(self):
        self.nodes = {}
        self.node_to_modality = {}

        self.node_list = []
        for modality in self.modalities:
            self.G.add_nodes_from(self.multi_omics_data[modality].get_genes_list(), modality=modality)
            self.nodes[modality] = self.multi_omics_data[modality].get_genes_list()
            self.nodes[modality] = [node for node in self.nodes[modality]]

            for gene in self.multi_omics_data[modality].get_genes_list():
                self.node_to_modality[gene] = modality

            print(modality, " nodes:", len(self.nodes[modality]))
            self.node_list.extend(self.multi_omics_data[modality].get_genes_list())

        print("Total nodes:", len(self.node_list))

    def process_genes_info(self):
        GE = self.multi_omics_data.GE.get_genes_info()
        MIR = self.multi_omics_data.MIR.get_genes_info()
        LNC = self.multi_omics_data.LNC.get_genes_info()

        GE["Family"] = GE["gene_family"] if "Family" not in GE else None
        MIR["Family"] = MIR["miR family"] if "Family" not in MIR else None
        LNC["Family"] = LNC["Rfams"] if "Family" not in LNC else None

        self.genes_info = pd.concat([GE, MIR, LNC], join="inner", copy=True)
        self.genes_info["Family"] = self.genes_info["Family"].str.split("|", expand=True)[
            0]  # TODO Selects only first family annotation if an RNA belongs to multiple
        print("Genes info columns:", self.genes_info.columns.tolist())
        self.genes_info = self.genes_info[~self.genes_info.index.duplicated()]

        self.node_seq_list = self.genes_info[self.genes_info["Transcript sequence"].notnull()].index.tolist()
        print("Number of nodes without seq removed:", len(self.node_list) - len(self.node_seq_list))

        self.node_list = [node for node in self.node_list if node in self.node_seq_list]
        print("Total nodes (filtered):", len(self.node_list))

        self.node_list = list(OrderedDict.fromkeys(self.node_list))


    def add_directed_edges_from_edgelist(self, edgelist, modalities, database,
                                         correlation_weights=False, threshold=None):
        if not (modalities is None):
            source_genes = set([edge[0] for edge in edgelist])
            target_genes = set([edge[1] for edge in edgelist])

            source_genes_matched = set(self.nodes[modalities[0]]) & source_genes
            target_genes_matched = set(self.nodes[modalities[1]]) & target_genes

            print("Adding edgelist with", len(source_genes), "total unique", modalities[0],
                  "genes (source), but only matching", len(source_genes_matched), "nodes")

            print("Adding edgelist with", len(target_genes), "total unique", modalities[1],
                  "genes (target), but only matching", len(target_genes_matched), "nodes")

        if correlation_weights == False:
            self.G.add_edges_from(edgelist, type="d", source=modalities[0], target=modalities[1], database=database)
            print(len(edgelist), "edges added.")
        else:
            node_list = [node for node in self.node_list if
                         node in self.nodes[modalities[0]] or node in self.nodes[modalities[1]]]
            correlation_df = compute_expression_correlation_dists(self.multi_omics_data, modalities=modalities,
                                                                  node_list=node_list, absolute_corr=True,
                                                                  return_distance=False,
                                                                  squareform=True)

            edgelist_weighted = [(edge[0], edge[1], {"weight": correlation_df.loc[edge[0], edge[1]]}) for edge in
                                 edgelist if
                                 edge[0] in node_list and edge[1] in node_list]

            if threshold is not None:
                no_edges = len(edgelist_weighted)
                edgelist_weighted = [(u, v, d) for u, v, d in edgelist_weighted if d["weight"] >= threshold]
                print("Filtered out", no_edges, "-", len(edgelist_weighted), "edges by correlation weight.")

            self.G.add_edges_from(edgelist_weighted, type="d", source=modalities[0], target=modalities[1],
                                  database=database)
            print(len(edgelist_weighted),
                  "weighted (directed interaction) edges added. Note: only added edges that are in the modalities:",
                  modalities)


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
            edge_list = self.G.edges(nbunch=node_list, data=True)
        else:
            if type(edge_types) == list:
                edge_list = [(u, v, d) for u, v, d in self.G.edges(data=True) if d['type'] in edge_types]
            else:
                edge_list = [(u, v, d) for u, v, d in self.G.edges(data=True) if d['type'] == edge_types]

        # Also add reverse edges for undirected edges
        if 'u' in edge_types or "u_n" in edge_types:
            edge_list.extend([(v, u, d) for u, v, d in edge_list if d['type'] in edge_types])

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

    def get_edgelist(self, edge_types, node_list, inclusive=True):
        edgelist = [(u, v) for u, v, d in self.G.edges(nbunch=node_list, data=True) if
                    d['type'] in edge_types]
        if inclusive:
            edgelist = [(u, v) for u, v in edgelist if (u in node_list and v in node_list)]

        return edgelist


    def add_edges_from_nodes_similarity(self, modality, node_list, features=None, weights=None,
                                        nanmean=True,
                                        similarity_threshold=0.7, dissimilarity_threshold=0.1,
                                        negative_sampling_ratio=2.0, max_positive_edges=None,
                                        compute_correlation=True, histological_subtypes=[], pathologic_stages=[],
                                        epsilon=1e-16, tag="affinity"):
        """
        Computes similarity measures between genes within the same modality, and add them as undirected edges to the
network if the similarity measures passes the threshold

        :param modality: E.g. ["GE", "MIR", "LNC"]
        :param similarity_threshold: a hard-threshold to select positive edges with affinity value more than it
        :param dissimilarity_threshold: a hard-threshold to select negative edges with affinity value less than
        :param negative_sampling_ratio: the number of negative edges in proportion to positive edges to select
        :param histological_subtypes: the patients' cancer subtype group to calculate correlation from
        :param pathologic_stages: the patient's cancer stage group to calculate correlations from
        """
        genes_info = self.multi_omics_data[modality].get_genes_info()

        # Filter similarity adj by correlation
        if compute_correlation:
            correlation_dist = compute_expression_correlation_dists(self.multi_omics_data, modalities=[modality],
                                                                    node_list=node_list, absolute_corr=True,
                                                                    histological_subtypes=histological_subtypes,
                                                                    pathologic_stages=pathologic_stages,
                                                                    squareform=False)
        else:
            correlation_dist = None

        annotation_affinities_df = pd.DataFrame(
            data=compute_annotation_affinities(genes_info, node_list=node_list, modality=modality,
                                               correlation_dist=correlation_dist, nanmean=nanmean,
                                               features=features, weights=weights, squareform=True),
            index=node_list)

        # Selects positive edges with high affinity in the affinity matrix
        similarity_filtered = np.triu(annotation_affinities_df >= similarity_threshold, k=1) # A True/False matrix
        sim_edgelist_ebunch = [(node_list[x], node_list[y], annotation_affinities_df.iloc[x, y]) for x, y in
                               zip(*np.nonzero(similarity_filtered))]
        # Sample
        if max_positive_edges is not None:
            sample_indices = np.random.choice(a=range(len(sim_edgelist_ebunch)),
                                              size=min(max_positive_edges, len(sim_edgelist_ebunch)), replace=False)
            sim_edgelist_ebunch = [(u, v, d) for i, (u, v, d) in enumerate(sim_edgelist_ebunch) if i in sample_indices]
            self.G.add_weighted_edges_from(sim_edgelist_ebunch, type="u", tag=tag)
        else:
            self.G.add_weighted_edges_from(sim_edgelist_ebunch, type="u", tag=tag)

        print(len(sim_edgelist_ebunch), "undirected positive edges (type='u') added.")




        # Select negative edges at affinity close to zero in the affinity matrix
        max_negative_edges = int(negative_sampling_ratio * len(sim_edgelist_ebunch))
        dissimilarity_filtered = np.triu(annotation_affinities_df <= dissimilarity_threshold, k=1)

        dissimilarity_index_rows, dissimilarity_index_cols = np.nonzero(dissimilarity_filtered)
        sample_indices = np.random.choice(a=dissimilarity_index_rows.shape[0],
                                          size=min(max_negative_edges, dissimilarity_index_rows.shape[0]),
                                          replace=False)
        # adds 1e-8 to keeps from 0.0 edge weights, which doesn't get picked up in nx.adjacency_matrix()
        dissim_edgelist_ebunch = [(node_list[x], node_list[y], annotation_affinities_df.iloc[x, y] + epsilon) for i, (x, y) in
                                  enumerate(zip(dissimilarity_index_rows[sample_indices],
                                                dissimilarity_index_cols[sample_indices])) if i < max_negative_edges]
        self.G.add_weighted_edges_from(dissim_edgelist_ebunch, type="u_n", tag=tag)

        print(len(dissim_edgelist_ebunch), "undirected negative edges (type='u_n') added.")
        return annotation_affinities_df


    def add_sampled_negative_edges(self, n_edges, modalities=[]):
        nodes_A = self.nodes[modalities[0]]
        nodes_B = self.nodes[modalities[1]]
        edges_ebunch = sample_edges(nodes_A, nodes_B, n_edges=n_edges, edge_type="u_n")

        print("Number of negative sampled edges between", modalities, "added:", len(edges_ebunch))
        self.G.add_edges_from(edges_ebunch)

    def add_sampled_negative_edges_from_correlation(self, modalities=[], correlation_threshold=0.2,
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

    def remove_edges_from(self, edgelist):
        self.G.remove_edges_from(edgelist)


    def set_node_similarity_training_adjacency(self, adj):
        self.adj_similarity_train = adj

    def set_regulatory_edges_training_adjacency(self, adj):
        self.adj_regulatory_train = adj

    def get_non_zero_degree_nodes(self):
        return [k for k, v in self.G.degree() if v > 0]




