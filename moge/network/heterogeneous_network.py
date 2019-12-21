from collections import OrderedDict

import networkx as nx
import scipy.sparse as sp
from sklearn import preprocessing

from moge.evaluation.utils import sample_edges
from moge.network.omics_distance import *
from moge.network.train_test_split import NetworkTrainTestSplit

EPSILON = 1e-16


def get_rename_dict(dataframe, alias_col_name):
    dataframe = dataframe[dataframe[alias_col_name].notnull()]
    b = pd.DataFrame(dataframe[alias_col_name].str.split('|').tolist(), index=dataframe.index).stack()
    b = b.reset_index(level=0)
    b.columns = ['index', 'alias']
    b.index = b["alias"]
    b = b.reindex()
    return pd.Series(b["index"]).to_dict()


class HeterogeneousNetwork(NetworkTrainTestSplit):
    def __init__(self, modalities: list, multi_omics_data: MultiOmics, process_annotations=True):
        """
        This class manages a networkx graph consisting of heterogeneous gene nodes, and heterogeneous edge types.

        :param modalities: A list of omics data to import (e.g. ["GE", "LNC"]). Each modalities has a list of genes
        :param multi_omics_data: The multiomics data to import
        """
        self.modalities = modalities
        self.multi_omics_data = multi_omics_data
        self.G = nx.DiGraph()
        self.G_u = nx.Graph()

        self.preprocess_graph()

        if process_annotations:
            self.process_annotations()
            self.process_feature_tranformer()

        super(HeterogeneousNetwork, self).__init__()

    def get_node_list(self):
        node_list = list(OrderedDict.fromkeys(list(self.G.nodes) + list(self.G_u.nodes)))
        return node_list
    node_list = property(get_node_list)

    def preprocess_graph(self):
        self.nodes = {}
        self.node_to_modality = {}

        bad_nodes = [node for node in self.get_node_list() if node is None or \
                     type(node) != str or \
                     node == "" or \
                     " " in node
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

    def process_annotations(self):
        annotations_list = []

        for modality in self.modalities:
            gene_info = self.multi_omics_data[modality].get_annotations()
            annotations_list.append(gene_info)

        self.annotations = pd.concat(annotations_list, join="inner", copy=True)
        self.annotations = self.annotations[~self.annotations.index.duplicated(keep='first')]
        print("Annotation columns:", self.annotations.columns.tolist())

        self.annotations['transcript_start'] = pd.to_numeric(
            self.annotations['transcript_start'].str.split("|", expand=True)[0])
        self.annotations['transcript_end'] = pd.to_numeric(
            self.annotations['transcript_end'].str.split("|", expand=True)[0])
        self.annotations['transcript_length'] = pd.to_numeric(
            self.annotations['transcript_length'].str.split("|", expand=True)[0])

    def process_feature_tranformer(self):
        self.feature_transformer = {}
        for label in self.annotations.columns:
            if label == 'Transcript sequence':
                continue

            if self.annotations[label].dtypes == np.object and self.annotations[label].str.contains("|").any():
                self.feature_transformer[label] = preprocessing.MultiLabelBinarizer()
                features = self.annotations.loc[self.node_list, label].dropna().str.split("|")
                self.feature_transformer[label].fit(features)

            elif self.annotations[label].dtypes == int or self.annotations[label].dtypes == float:
                self.feature_transformer[label] = preprocessing.StandardScaler()
                features = self.annotations.loc[self.node_list, label].dropna()
                self.feature_transformer[label].fit(features.to_numpy().reshape(-1, 1))

            else:
                self.feature_transformer[label] = preprocessing.MultiLabelBinarizer()
                features = self.annotations.loc[self.node_list, label].dropna()
                self.feature_transformer[label].fit(features.to_numpy().reshape(-1, 1))

    def add_directed_edges(self, edgelist, modalities, database,
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
            edge_list = [(u, v, d) for u, v, d in self.G.edges(nbunch=node_list, data=True) if
                         'database' in d and d['database'] in databases]
            adj = nx.adjacency_matrix(nx.DiGraph(incoming_graph_data=edge_list), nodelist=node_list)
        elif is_directed:
            adj = nx.adjacency_matrix(self.G.subgraph(nodes=node_list), nodelist=node_list)
            if sample_negative:
                adj = self.sample_random_negative_edges(adj.astype(float), negative_sampling_ratio=sample_negative)
        elif not is_directed and (("u" in edge_types and "u_n" in edge_types) or "u" in edge_types):
            adj = nx.adjacency_matrix(self.G_u.subgraph(nodes=node_list), nodelist=node_list)
        elif not is_directed and ("u_n" == edge_types or "u_n" in edge_types):
            edge_list = [(u, v, d) for u, v, d in self.G_u.edges(nbunch=node_list, data=True) if
                         d['type'] in edge_types]
            adj = nx.adjacency_matrix(nx.Graph(incoming_graph_data=edge_list), nodelist=node_list)

        # Eliminate self-edges
        adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
        return adj.astype(float)

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

    def get_edge(self, i, j, is_directed=True):
        if is_directed:
            return self.G.get_edge_data(i, j)
        else:
            return self.G_u.get_edge_data(i, j)

    def get_subgraph(self, modalities=["MIR", "LNC", "GE"], edge_type="d"):
        if modalities==None:
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

    def add_undirected_edges_from_attibutes(self, modality, node_list, features=None, weights=None,
                                            nanmean=True,
                                            similarity_threshold=0.7, dissimilarity_threshold=0.1,
                                            negative_sampling_ratio=2.0, max_positive_edges=None,
                                            compute_correlation=True, tissue_expression=False, histological_subtypes=[],
                                            pathologic_stages=[],
                                            epsilon=EPSILON, tag="affinity"):
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
        annotations = self.multi_omics_data[modality].get_annotations()

        # Filter similarity adj by correlation
        if compute_correlation:
            correlation_dist = compute_expression_correlation_dists(self.multi_omics_data, modalities=[modality],
                                                                    node_list=node_list, absolute_corr=True,
                                                                    return_distance=True,
                                                                    histological_subtypes=histological_subtypes,
                                                                    pathologic_stages=pathologic_stages,
                                                                    squareform=False,
                                                                    tissue_expression=tissue_expression)
        else:
            correlation_dist = None

        annotation_affinities_df = pd.DataFrame(
            data=compute_annotation_affinities(annotations, node_list=node_list, modality=modality,
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
