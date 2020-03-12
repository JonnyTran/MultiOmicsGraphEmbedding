import networkx as nx
import pandas as pd

from moge.network.attributed import AttributedNetwork, MODALITY_COL
from moge.network.train_test_split import TrainTestSplit, filter_y_multilabel, stratify_train_test, \
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
        self.annotations = {}
        for modality in self.modalities:
            annotation = self.multiomics[modality].get_annotations()
            self.annotations[modality] = annotation

        self.annotations = pd.Series(self.annotations)
        print("All annotation columns (union):",
              {col for _, annotations in self.annotations.items() for col in annotations.columns.tolist()})

    def process_feature_tranformer(self, delimiter="\||;", min_count=0):
        self.delimiter = delimiter

        annotations_list = []

        for modality in self.modalities:
            annotation = self.multiomics[modality].get_annotations()
            annotation["omic"] = modality
            annotations_list.append(annotation)

        self.all_annotations = pd.concat(annotations_list, join="inner", copy=True)
        self.all_annotations = self.all_annotations[~self.all_annotations.index.duplicated(keep='first')]
        print("Annotation columns:", self.all_annotations.columns.tolist())

        self.feature_transformer = self.get_feature_transformers(self.all_annotations, self.node_list, delimiter,
                                                                 min_count)

    def add_edges(self, edgelist, source, target, database, **kwargs):
        self.networks[(source, target)].add_edges_from(edgelist, source=source, target=target, database=database,
                                                       **kwargs)
        print(len(edgelist), "edges added to self.networks[({}, {})]".format(source, target))

    def get_adjacency_matrix(self, edge_types: (str, str), node_list=None, ):
        if node_list is None:
            node_list = self.node_list

        # edge_list = [(u, v) for u, v, d in self.networks[edge_types].edges(nbunch=node_list, data=True)]
        if isinstance(edge_types, tuple):
            assert edge_types in self.networks
            adj = nx.adjacency_matrix(self.networks[edge_types], nodelist=node_list)

        elif isinstance(edge_types, list) and isinstance(edge_types[0], tuple):
            assert self.networks.issuperset(edge_types)
            adj = {}
            for edge_type in edge_types:
                adj[edge_type] = nx.adjacency_matrix(self.networks[edge_type],
                                                     nodelist=node_list)  # TODO some edges may have weight > 1
        else:
            raise Exception("edge_types '{}' must be one of {}".format(edge_types, self.layers))

        # Eliminate self-edges
        # adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
        return adj.astype(float)

    def split_stratified(self, stratify_label: str, stratify_omic=True, n_splits=5,
                         dropna=False, seed=42, verbose=False):
        y_label, _ = filter_y_multilabel(annotations=self.all_annotations, y_label=stratify_label, min_count=n_splits,
                                         dropna=dropna, delimiter=self.delimiter)
        if stratify_omic:
            y_omic = self.all_annotations.loc[y_label.index, MODALITY_COL].str.split(
                "|")  # Need to return a Series of []'s
            y_label = y_label + y_omic

        train_nodes, test_nodes = stratify_train_test(y_label=y_label, n_splits=n_splits, seed=seed)

        self.set_training_annotations(train_nodes)
        self.set_testing_annotations(test_nodes)

        self.training.networks = {}
        self.testing.networks = {}
        for layer, network in self.networks.items():
            network_train, network_test = split_network_by_nodes(network, train_nodes=train_nodes,
                                                                 test_nodes=test_nodes, verbose=verbose)
            self.training.networks[layer] = network_train
            self.testing.networks[layer] = network_test

            print("{} layer train_network".format(str(layer)), self.training.networks[layer].number_of_nodes(),
                  self.training.networks[layer].number_of_edges()) if verbose else None
            print("{} layer test_network".format(str(layer)), self.testing.networks[layer].number_of_nodes(),
                  self.testing.networks[layer].number_of_edges()) if verbose else None
