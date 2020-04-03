import networkx as nx
import numpy as np
import pandas as pd
from openomics.utils.df import concat_uniques

from moge.network.attributed import AttributedNetwork, filter_y_multilabel
from moge.network.train_test_split import TrainTestSplit, stratify_train_test, \
    split_network_by_nodes


class MultiplexAttributedNetwork(AttributedNetwork, TrainTestSplit):
    def __init__(self, multiomics, modalities: list, layers: {(str, str): nx.Graph}, annotations=True, ) -> None:
        """

        :param multiomics:
        :param modalities:
        :param layers:
        :param annotations:
        """
        self.modalities = modalities
        self.layers = layers
        self.layers_adj = {}

        networks = {}
        for source_target, graph_class in layers.items():
            networks[source_target] = graph_class()

        super(MultiplexAttributedNetwork, self).__init__(networks=networks, multiomics=multiomics,
                                                         annotations=annotations)

    def process_network(self):
        self.nodes = {}
        self.node_to_modality = {}

        for source_target, network in self.networks.items():
            for modality in self.modalities:
                network.add_nodes_from(self.multiomics[modality].get_genes_list(), modality=modality)

        for modality in self.modalities:
            self.nodes[modality] = self.multiomics[modality].get_genes_list()

            for gene in self.multiomics[modality].get_genes_list():
                self.node_to_modality[gene] = self.node_to_modality.setdefault(gene, []) + [modality, ]

            print(modality, " nodes:", len(self.nodes[modality]))
        print("Total nodes:", len(self.get_node_list()))
        self.nodes = pd.Series(self.nodes)
        self.node_to_modality = pd.Series(self.node_to_modality)

    def process_annotations(self):
        self.annotations_dict = {}
        for modality in self.modalities:
            annotation = self.multiomics[modality].get_annotations()
            self.annotations_dict[modality] = annotation

        self.annotations_dict = pd.Series(self.annotations_dict)
        print("All annotation columns (union):",
              {col for _, annotations in self.annotations_dict.items() for col in annotations.columns.tolist()})

    def process_feature_tranformer(self, delimiter="\||;", min_count=0):
        self.delimiter = delimiter
        if not hasattr(self, "all_annotations"):
            annotations_list = []

            for modality in self.modalities:
                annotation = self.multiomics[modality].get_annotations()
                annotation["omic"] = modality
                annotations_list.append(annotation)

            self.annotations = pd.concat(annotations_list, join="inner", copy=True)
            self.annotations = self.annotations.groupby(self.annotations.index).agg(
                {k: concat_uniques for k in self.annotations.columns})

        print("Annotation columns:", self.annotations.columns.tolist())
        self.feature_transformer = self.get_feature_transformers(self.annotations, self.node_list, delimiter,
                                                                 min_count)

    def add_edges(self, edgelist, layer: (str, str, str), database, **kwargs):
        source = layer[0]
        target = layer[1]
        self.networks[layer].add_edges_from(edgelist, source=source, target=target, database=database, **kwargs)
        print(len(edgelist), "edges added to self.networks[{}]".format(layer))

    def get_adjacency_matrix(self, edge_types: (str, str), node_list=None, ):
        """

        :param edge_types: either a tuple(str, ...) or [tuple(str, ...), tuple(str, ...)]
        :param node_list (list):
        :return:
        """
        if node_list is None:
            node_list = self.node_list

        # edge_list = [(u, v) for u, v, d in self.networks[edge_types].edges(nbunch=node_list, data=True)]
        if isinstance(edge_types, tuple):
            assert edge_types in self.networks
            adj = self.get_layer_adjacency_matrix(edge_types, node_list)

        elif isinstance(edge_types, list) and isinstance(edge_types[0], tuple):
            assert self.networks.issuperset(edge_types)
            adj = {}
            for layer in edge_types:
                adj[layer] = self.get_layer_adjacency_matrix(layer, node_list)
        else:
            raise Exception("edge_types '{}' must be one of {}".format(edge_types, self.layers))

        # Eliminate self-edges
        # adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
        return adj.astype(float)

    def get_layer_adjacency_matrix(self, edge_type, node_list=None):
        if edge_type in self.layers_adj:
            adjacency_matrix = self.layers_adj[edge_type]

        # Get adjacency and caches the matrix
        else:
            adjacency_matrix = nx.adjacency_matrix(self.networks[edge_type],
                                                   nodelist=self.node_list)
            adjacency_matrix = adjacency_matrix + np.eye(adjacency_matrix.shape[0])  # Add self-loops
            self.layers_adj[edge_type] = adjacency_matrix

        if node_list is None or node_list == self.node_list:
            return adjacency_matrix
        elif set(node_list) < set(self.node_list):
            return self.slice_adj(adjacency_matrix, node_list, None)
        elif not (set(node_list) < set(self.node_list)):
            raise Exception("A node in node_l is not in self.node_list.")

        return adjacency_matrix

    def split_stratified(self, stratify_label: str, stratify_label_2=None, n_splits=5,
                         dropna=False, seed=42, verbose=False):
        y_label = filter_y_multilabel(annotations=self.annotations, y_label=stratify_label, min_count=n_splits,
                                      dropna=dropna, delimiter=self.delimiter)
        if stratify_label_2:
            y_omic = self.annotations.loc[y_label.index, stratify_label_2].str.split(self.delimiter)
            y_label = y_label + y_omic

        train_nodes, test_nodes = stratify_train_test(y_label=y_label, n_splits=n_splits, seed=seed)

        self.set_training_annotations(train_nodes)
        self.set_testing_annotations(test_nodes)

        self.training.networks = {}
        self.testing.networks = {}
        for layer, network in self.networks.items():
            network_train, network_test = split_network_by_nodes(network, train_nodes=train_nodes,
                                                                 test_nodes=test_nodes, verbose=False)
            self.training.networks[layer] = network_train
            self.testing.networks[layer] = network_test

            print("Layer {} train_network".format(str(layer)), self.training.networks[layer].number_of_nodes(),
                  self.training.networks[layer].number_of_edges()) if verbose else None
            print("Layer {} test_network".format(str(layer)), self.testing.networks[layer].number_of_nodes(),
                  self.testing.networks[layer].number_of_edges()) if verbose else None

    def get_aggregated_network(self):
        G = nx.compose_all(list(self.networks.values()))
        return G
