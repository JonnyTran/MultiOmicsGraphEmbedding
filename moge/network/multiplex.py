from networkx import Graph

from moge.network.attributed import AttributedNetwork
from moge.network.train_test_split import TrainTestSplit


class MultiplexAttributedNetwork(AttributedNetwork, TrainTestSplit):
    def __init__(self, multiomics, modalities: list, layers: {(str, str): Graph}, annotations=True, ) -> None:
        self.modalities = modalities
        self.layers = layers

        networks = {}
        for layer, graph_class in self.layers.items():
            networks[layer] = graph_class()

        super(MultiplexAttributedNetwork, self).__init__(networks=networks, multiomics=multiomics,
                                                         modalities=modalities,
                                                         annotations=annotations)

    def process_network(self):
        self.nodes = {}
        self.node_to_modality = {}

        for network in self.networks.values():
            for modality in self.modalities:
                network.add_nodes_from(self.multiomics[modality].get_genes_list(), modality=modality)

        self.nodes[modality] = self.multiomics[modality].get_genes_list()

        for gene in self.multiomics[modality].get_genes_list():
            self.node_to_modality[gene] = modality

        print(modality, " nodes:", len(self.nodes[modality]))
        print("Total nodes:", len(self.get_node_list()))

    def process_annotations(self):
        self.annotations = {}
        for modality in self.modalities:
            annotation = self.multiomics[modality].get_annotations()

            self.annotations[modality] = annotation

        print("Annotation columns:",
              {modality: annotations.columns.tolist() for modality, annotations in self.annotations.items()})

    def add_edges(self, edgelist, layer):
        pass

    def get_layer(self, modality_A, modality_B):
        pass

    def get_adjacency_matrix(self, edge_types: list, node_list=None, **kwargs):
        pass
