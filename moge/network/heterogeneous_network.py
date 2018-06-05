import os

import numpy as np
import pandas as pd
import networkx as nx
import dask.dataframe as dd
from dask.threaded import get

from TCGAMultiOmics.multiomics import MultiOmicsData


class HeterogeneousNetwork():
    def __init__(self, modalities:list, multi_omics_data:MultiOmicsData):
        self.modalities = modalities
        self.multi_omics_data = multi_omics_data
        self.G = nx.DiGraph()

        self.preprocess_graph()

    def preprocess_graph(self):
        self.nodes = {}
        self.all_nodes = []
        for modality in self.modalities:
            self.G.add_nodes_from(self.multi_omics_data[modality].get_genes_list())
            self.nodes[modality] = self.multi_omics_data[modality].get_genes_list()
            self.all_nodes.extend(self.multi_omics_data[modality].get_genes_list())

    def add_edges_from_modality(self, modality):
        self.G.add_edges_from(self.multi_omics_data[modality].network.edges(data=True))

    def add_edges_from_edgelist(self, edgelist):
        self.G.add_edges_from(edgelist)

    def get_affinity_matrix(self):
        return nx.adjacency_matrix(self.G)

    def get_edge(self, i, j):
        return self.G.get_edge_data(i, j)

    def get_subgraph(self, modalities):
        nodes = []
        for modality in modalities:
            nodes.extend(self.nodes[modality])

        return self.G.subgraph(nodes)

    def compute_multiomics_correlations(self, modalities, pathologic_stages=[], histological_subtypes=[]):
        X_multiomics, y = self.multi_omics_data.load_data(modalities=modalities, pathologic_stages=pathologic_stages, histological_subtypes=histological_subtypes)

        X_multiomics_concat = pd.concat([X_multiomics[m] for m in modalities], axis=1)
        X_multiomics_corr = np.corrcoef(X_multiomics_concat,  rowvar=False)

        cols = X_multiomics_concat.columns
        X_multiomics_corr_df = pd.DataFrame(X_multiomics_corr, columns=cols, index=cols)

        return X_multiomics_corr_df

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


    def _merge_results(self, result):
        pass

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

