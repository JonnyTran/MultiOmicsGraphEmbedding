import networkx as nx
import numpy as np
import scipy.sparse as sp

from moge.network.attributed import AttributedNetwork
from moge.network.train_test_split import TrainTestSplit


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
                if modality in self.node_to_modality[gene]:
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

        print("Annotation columns:",
              {modality: annotations.columns.tolist() for modality, annotations in self.annotations.items()})

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
