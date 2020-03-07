import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn import preprocessing

from moge.network.attributed import AttributedNetwork, SEQUENCE_COL
from moge.network.train_test_split import TrainTestSplit, get_labels_filter


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

        networks = {}
        for source_target, graph_class in self.layers.items():
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
                if modality in self.node_to_modality:
                    self.node_to_modality[gene] = [self.node_to_modality[gene], ] + [modality]
                else:
                    self.node_to_modality[gene] = modality

            print(modality, " nodes:", len(self.nodes[modality]))
        print("Total nodes:", len(self.get_node_list()))

    def process_annotations(self):
        self.annotations = {}
        for modality in self.modalities:
            annotation = self.multiomics[modality].get_annotations()
            self.annotations[modality] = annotation

        print("All annotation columns (union):",
              {col for _, annotations in self.annotations.items() for col in annotations.columns.tolist()})

    def process_feature_tranformer(self, delimiter="|", min_count=0):
        annotations_list = []

        for modality in self.modalities:
            annotation = self.multiomics[modality].get_annotations()
            annotation["omic"] = modality
            annotations_list.append(annotation)

        self.all_annotations = pd.concat(annotations_list, join="inner", copy=True)
        self.all_annotations = self.all_annotations[~self.all_annotations.index.duplicated(keep='first')]
        print("Annotation columns:", self.all_annotations.columns.tolist())

        self.feature_transformer = {}
        for label in self.all_annotations.columns:
            if label == SEQUENCE_COL:
                continue

            if self.all_annotations[label].dtypes == np.object and self.all_annotations[label].str.contains(delimiter,
                                                                                                            regex=False).any():
                print(
                    "INFO: Label {} is split by delim '{}' transformed by MultiLabelBinarizer".format(label, delimiter))
                self.feature_transformer[label] = preprocessing.MultiLabelBinarizer()
                features = self.all_annotations.loc[self.node_list, label].dropna(axis=0).str.split(delimiter)
                if min_count:
                    labels_filter = get_labels_filter(self, features.index, label, min_count=min_count)
                    features = features.map(lambda labels: [item for item in labels if item not in labels_filter])
                self.feature_transformer[label].fit(features)

            elif self.all_annotations[label].dtypes == int or self.all_annotations[label].dtypes == float:
                print("INFO: Label {} is transformed by StandardScaler".format(label))
                self.feature_transformer[label] = preprocessing.StandardScaler()
                features = self.all_annotations.loc[self.node_list, label].dropna(axis=0)
                self.feature_transformer[label].fit(features.to_numpy().reshape(-1, 1))
            else:
                print("INFO: Label {} is transformed by MultiLabelBinarizer".format(label))
                self.feature_transformer[label] = preprocessing.MultiLabelBinarizer()
                features = self.all_annotations.loc[self.node_list, label].dropna(axis=0)
                self.feature_transformer[label].fit(features.to_numpy().reshape(-1, 1))

    def add_edges(self, edgelist, source, target, database, **kwargs):
        self.networks[(source, target)].add_edges_from(edgelist, source=source, target=target, database=database,
                                                       **kwargs)
        print(len(edgelist), "edges added to self.layers[({}, {})]".format(source, target))

    def get_adjacency_matrix(self, edge_types: (str, str), node_list=None, ):
        if node_list is None:
            node_list = self.node_list

        # edge_list = [(u, v) for u, v, d in self.networks[edge_types].edges(nbunch=node_list, data=True)]
        if isinstance(edge_types, tuple):
            assert edge_types in self.networks
            adj = nx.adjacency_matrix(self.networks[edge_types], nodelist=node_list)

        elif isinstance(edge_types, list) and isinstance(edge_types[0], tuple):
            assert self.networks.issuperset(edge_types)
            adj = nx.adjacency_matrix(self.networks[edge_types[0]], nodelist=node_list)
            for edge_type in edge_types[1:]:
                adj = adj + nx.adjacency_matrix(self.networks[edge_type],
                                                nodelist=node_list)  # TODO some edges may have weight > 1

        # Eliminate self-edges
        adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
        return adj.astype(float)
