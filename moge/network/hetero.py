from argparse import Namespace

import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, Tuple

from moge.network.attributed import AttributedNetwork, MODALITY_COL, filter_y_multilabel
from moge.network.train_test_split import TrainTestSplit, stratify_train_test

from openomics.utils.df import concat_uniques
from openomics import MultiOmics
from moge.data.dgl.node_generator import DGLNodeSampler


class HeteroNetwork(AttributedNetwork, TrainTestSplit):
    def __init__(self, multiomics: MultiOmics, node_types: list, layers: Dict[Tuple[str]: nx.Graph],
                 annotations=True, ) -> None:
        """
        :param multiomics: MultiOmics object containing annotations
        :param node_types: Node types
        :param layers: A dict of edge types tuple and networkx.Graph/Digraph containing heterogeneous edges
        :param annotations: Whether to process annotation data, default True
        """
        self.multiomics = multiomics
        self.node_types = node_types
        self.networks: Dict[nx.Graph] = {}

        networks = {}
        for src_etype_dst, GraphClass in layers.items():
            networks[src_etype_dst] = GraphClass()

        super(HeteroNetwork, self).__init__(networks=networks, multiomics=multiomics,
                                            annotations=annotations)

    def process_network(self):
        self.nodes = {}
        self.node_to_modality = {}

        for source_target, network in self.networks.items():
            for node_type in self.node_types:
                network.add_nodes_from(self.multiomics[node_type].get_genes_list(), modality=node_type)

        for node_type in self.node_types:
            self.nodes[node_type] = self.multiomics[node_type].get_genes_list()

            for gene in self.multiomics[node_type].get_genes_list():
                self.node_to_modality[gene] = self.node_to_modality.setdefault(gene, []) + [node_type, ]
            print(node_type, " nodes:", len(self.nodes[node_type]), self.multiomics[node_type].gene_index)

        print("Total nodes:", len(self.get_node_list()))
        self.nodes = pd.Series(self.nodes)
        self.node_to_modality = pd.Series(self.node_to_modality)

    def process_annotations(self):
        self.annotations = {}
        for modality in self.node_types:
            annotation = self.multiomics[modality].get_annotations()
            self.annotations[modality] = annotation

        self.annotations = pd.Series(self.annotations)
        print("All annotation columns (union):",
              {col for _, annotations in self.annotations.items() for col in annotations.columns.tolist()})

    def process_feature_tranformer(self, delimiter="\||;", filter_label=None, min_count=0, verbose=False):
        self.delimiter = delimiter
        if not hasattr(self, "all_annotations"):
            annotations_list = []

            for modality in self.node_types:
                annotation = self.multiomics[modality].get_annotations()
                annotation["omic"] = modality
                annotations_list.append(annotation)

            self.all_annotations = pd.concat(annotations_list, join="inner", copy=True)
            self.all_annotations = self.all_annotations.groupby(self.all_annotations.index).agg(
                {k: concat_uniques for k in self.all_annotations.columns})

        print("Annotation columns:", self.all_annotations.columns.tolist()) if verbose else None
        self.feature_transformer = self.get_feature_transformers(self.all_annotations, self.node_list, delimiter,
                                                                 filter_label,
                                                                 min_count, verbose=verbose)

    def add_edges(self, edgelist, layer: (str, str, str), database, **kwargs):
        source = layer[0]
        target = layer[-1]
        self.networks[layer].add_edges_from(edgelist, source=source, target=target, database=database, **kwargs)
        print(len(edgelist), "edges added to self.networks[{}]".format(layer))

    def get_adjacency_matrix(self, edge_types: (str, str), node_list=None, method="GAT", output="dense"):
        """

        :param edge_types: either a tuple(str, ...) or [tuple(str, ...), tuple(str, ...)]
        :param node_list (list):
        :return:
        """
        if node_list is None:
            node_list = self.node_list

        if isinstance(edge_types, tuple):
            assert edge_types in self.networks
            adj = self.get_layer_adjacency_matrix(edge_types, node_list, method=method, output=output)

        elif isinstance(edge_types, list) and isinstance(edge_types[0], tuple):
            assert self.networks.issuperset(edge_types)
            adj = {}
            for layer in edge_types:
                adj[layer] = self.get_layer_adjacency_matrix(layer, node_list)
        else:
            raise Exception("edge_types '{}' must be one of {}".format(edge_types, self.node_types))

        return adj

    def get_layer_adjacency_matrix(self, edge_type, node_list=None, method="GAT", output="csr"):
        if edge_type in self.layers_adj:
            adjacency_matrix = self.layers_adj[edge_type]

        # Get adjacency and caches the matrix
        else:
            adjacency_matrix = nx.adjacency_matrix(self.networks[edge_type],
                                                   nodelist=self.node_list)
            # if method == "GAT":
            #     adjacency_matrix = adjacency_matrix + sps.csr_matrix(
            #         np.eye(adjacency_matrix.shape[0]))  # Add self-loops

            self.layers_adj[edge_type] = adjacency_matrix

        if node_list is None or node_list == self.node_list:
            pass
        elif set(node_list) <= set(self.node_list):
            adjacency_matrix = self.slice_adj(adjacency_matrix, node_list, None)
        elif set(node_list) > set(self.node_list):
            raise Exception(f"A node in node_l is not in self.node_list : {set(node_list) - set(self.node_list)}")

        if output == "csr":
            return adjacency_matrix.astype(float)
        elif output == "coo":
            adjacency_matrix = adjacency_matrix.tocoo(copy=True)
            return np.vstack((adjacency_matrix.row, adjacency_matrix.col)).astype("long")
        elif output == "dense":
            return adjacency_matrix.todense()
        else:
            raise Exception("Output must be one of {csr, coo, dense}")

    def split_stratified(self, stratify_label: str, stratify_omic=True, n_splits=5,
                         dropna=False, seed=42, verbose=False):
        y_label = filter_y_multilabel(annotations=self.all_annotations, y_label=stratify_label,
                                      min_count=n_splits,
                                      dropna=dropna, delimiter=self.delimiter)
        if stratify_omic:
            y_omic = self.all_annotations.loc[y_label.index,
                                              MODALITY_COL].str.split("\||:")
            y_label = y_label + y_omic

        print("y_label", y_label.shape)

        self.train_test_splits = list(stratify_train_test(y_label=y_label, n_splits=n_splits, seed=seed))

        self.training = Namespace()
        self.testing = Namespace()
        self.training.node_list = self.train_test_splits[0][0]
        self.testing.node_list = self.train_test_splits[0][1]

    def get_aggregated_network(self):
        G = nx.compose_all(list(self.networks.values()))
        return G
