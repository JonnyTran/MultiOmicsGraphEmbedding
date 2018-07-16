import dask.dataframe as dd
from dask.threaded import get
import scipy.sparse as sp

from moge.network.omics_distance import *


class HeterogeneousNetwork():
    def __init__(self, modalities:list, multi_omics_data:MultiOmicsData):
        self.modalities = modalities
        self.multi_omics_data = multi_omics_data
        self.G = nx.DiGraph()

        self.preprocess_graph()

    def preprocess_graph(self):
        self.nodes = {}
        self.node_list = []
        for modality in self.modalities:
            self.G.add_nodes_from(self.multi_omics_data[modality].get_genes_list(), modality=modality)
            self.nodes[modality] = self.multi_omics_data[modality].get_genes_list()

            print(modality, " nodes:", len(self.nodes[modality]))
            self.node_list.extend(self.multi_omics_data[modality].get_genes_list())

        print("Total nodes:", len(self.node_list))


    def add_edges_from_edgelist(self, edgelist, modalities=None):
        if not (modalities is None):
            source_genes = set(pd.DataFrame(edgelist)[0].tolist())
            target_genes = set(pd.DataFrame(edgelist)[1].tolist())

            source_genes_matched = set(self.nodes[modalities[0]]) & source_genes
            target_genes_matched = set(self.nodes[modalities[1]]) & target_genes

            print("Adding edgelist with", len(source_genes), "total unique", modalities[0], "genes (source), but only matching", len(source_genes_matched), "nodes")
            print("Adding edgelist with", len(target_genes), "total unique", modalities[1], "genes (target), but only matching", len(target_genes_matched), "nodes")
            print(len(edgelist), "edges added.")

        self.G.add_edges_from(edgelist, type="d")

    def import_edgelist_file(self, file, directed):
        if directed:
            self.G.add_edges_from(nx.read_edgelist(file, data=True, create_using=nx.DiGraph()).edges(data=True))
        else:
            self.G.add_edges_from(nx.read_edgelist(file, data=True, create_using=nx.Graph()).edges(data=True))

    def get_adjacency_matrix(self, edge_type=["u", "d"], node_list=None, get_training_data=False):
        """
        Get adjacency matrix, and remove diagonal elements
        :return:
        """
        if node_list==None:
            node_list = self.node_list

        if edge_type == None:
            edge_list = self.G.edges(data=True)
        else:
            if get_training_data:
                if "u" in edge_type:
                    return self.adj_similarity_train
                elif 'd' in edge_type:
                    return self.adj_regulatory_train
            else:
                edge_list = [(u, v, d) for u, v, d in self.G.edges(data=True) if d['type'] in edge_type]

        if 'u' in edge_type:
            undirected_edge_list = [(v, u, d) for u, v, d in edge_list if d['type'] == 'u']
            edge_list.extend(undirected_edge_list)

        adj = nx.adjacency_matrix(nx.DiGraph(incoming_graph_data=edge_list), nodelist=node_list)

        # Eliminate self-edges
        adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
        adj.eliminate_zeros()
        return adj

    def get_edge(self, i, j):
        return self.G.get_edge_data(i, j)

    def get_subgraph(self, modalities=["MIR", "LNC", "GE"]):
        if modalities==None:
            modalities = self.modalities

        nodes = []
        for modality in modalities:
            nodes.extend(self.nodes[modality])

        return self.G.subgraph(nodes)


    def add_edges_from_nodes_similarity(self, modality, features=None, similarity_threshold=0.7, data=True):
        """
        Computes similarity measures between genes within the same modality, and add them as undirected edges to the network if the similarity measures passes the threshold

        :param modality: E.g. "GE", "MIR", "LNC"
        :param similarity_threshold: a hard-threshold on the similarity measure
        :param data:
        """
        genes_info = self.multi_omics_data[modality].get_genes_info()

        similarity_adj_df = pd.DataFrame(compute_annotation_similarity(genes_info, modality=modality, features=features, squareform=True), index=self.multi_omics_data[modality].get_genes_list())

        similarity_filtered = similarity_adj_df.loc[:, :] >= similarity_threshold
        index = similarity_adj_df.index

        edgelist_ebunch = [(index[x], index[y], similarity_adj_df.iloc[x, y]) for x, y in zip(*np.nonzero(similarity_filtered.values))]

        self.G.add_weighted_edges_from(edgelist_ebunch, type="u")
        print(len(edgelist_ebunch), "edges added.")

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



    @DeprecationWarning
    def compute_corr_graph_(self, modalities_pairs):
        self.C = nx.Graph()
        self.C.add_nodes_from(self.G.nodes())

        for modalities_pair in modalities_pairs:
            all_assoc = []

            for m in self.nodes[modalities_pair[0]]:
                for t in self.nodes[modalities_pair[1]]:
                    all_assoc.append((m, t))

            all_assoc = pd.DataFrame(all_assoc, columns=["source", "target"])

            edges_added = self._map(putative_assocs=all_assoc, map_func=self._correlation, apply_func=self._add_edge_to_corr_graph, modalities=modalities_pair)

            print(modalities_pairs, "edges added:", edges_added)


    def _map(self, putative_assocs, map_func, apply_func, modalities, pathologic_stages=[], histological_subtypes=[], n_jobs=4):
        edges_added = 0

        X_multiomics, _ = self.multi_omics_data.load_data(modalities, pathologic_stages=pathologic_stages, histological_subtypes=histological_subtypes)

        if putative_assocs is not None:
            putative_dd = dd.from_pandas(putative_assocs, npartitions=n_jobs)

            result = putative_dd.map_partitions(map_func, meta=putative_dd, X_multiomics=X_multiomics, modalities=modalities).compute(get=get)

            for res_partition in result:
                for tup in res_partition:
                    apply_func(tup[0], tup[1], dys=tup[2])
                    edges_added += 1
        # return apply_func(result)
        return edges_added

    def _correlation(self, df, X_multiomics, modalities, threshold=None):
        result = []
        for row in df.iterrows():
            s = row[1]['source']
            t = row[1]['target']

            s_data = X_multiomics[modalities[0]][s]
            t_data = X_multiomics[modalities[1]][t]
            n_samples = s_data.shape[0]

            corr = np.dot(s_data - np.mean(s_data), t_data - np.mean(t_data)) / ((n_samples - 1) * np.std(s_data) * np.std(t_data))

            if "MIR" == modalities[0] and "GE" == modalities[1]:
                if corr < 0:
                    result.append((s, t, corr))
            else:
                result.append((s, t, corr))

        return result

    def _add_edge_to_corr_graph(self, source, target, corr):
        self.C.add_edge(source, target, weight=corr)

    def _geneInfo(self, df, modalities):
        pass


    def _test(self):
        df = pd.DataFrame(
            data={"source": ['hsa-let-7c', 'hsa-let-7d', 'hsa-let-7e', 'hsa-let-7f-1', 'hsa-let-7f-2', 'hsa-let-7g',
                             'hsa-let-7i'], "target": ['A2ML1', 'A4GALT', 'A4GNT', 'AAAS', 'ABCA2', 'ABCA3', 'ABCA4']})

        result = self._map(df, map_func=self._correlation, apply_func=None, modalities=["MIR", "GE"])
        print(result)


    # def calc_dys_A_B(df, miRNA_A, miRNA_B, gene_A, gene_B):
    #     result = []
    #     for row in df.iterrows():
    #         m = row[1]['MiRBase ID']
    #         t = row[1]['Gene Symbol']
    #         miRNA_gene_A_corr = np.dot(miRNA_A[m] - np.mean(miRNA_A[m]),
    #                                    gene_A[t] - np.mean(gene_A[t])) / \
    #                             ((n_A - 1) * np.std(miRNA_A[m]) * np.std(gene_A[t]))
    #         miRNA_gene_B_corr = np.dot(miRNA_B[m] - np.mean(miRNA_B[m]),
    #                                    gene_B[t] - np.mean(gene_B[t])) / \
    #                             ((n_B - 1) * np.std(miRNA_B[m]) * np.std(gene_B[t]))
    #         dys = miRNA_gene_A_corr - miRNA_gene_B_corr
    #         p_value = self.z_to_p_value(self.fisher_r_to_z(miRNA_gene_A_corr, n_A, miRNA_gene_B_corr, n_B))
    #
    #         if p_value <= p_threshold and (miRNA_gene_A_corr < 0 or miRNA_gene_B_corr < 0):
    #             result.append((m, t, p_value))
    #     return result

