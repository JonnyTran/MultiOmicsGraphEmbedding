from networkx import Graph

from moge.network.attributednetwork import MultiplexAttributedNetwork
from moge.network.base import Network
from moge.network.semantic_similarity import *
from moge.network.train_test_split import TrainTestSplit


class MultiplexHeterogeneousNetwork(Network, MultiplexAttributedNetwork, TrainTestSplit):
    def __init__(self, modalities: list, layers: {str: Graph}, multiomics: MultiOmics, process_annotations=True):
        self.modalities = modalities
        self.layers = layers

        super(MultiplexHeterogeneousNetwork, self).__init__(multiomics=multiomics,
                                                            process_annotations=process_annotations)

    def add_edges(self, edgelist, **kwargs):
        pass

    def get_adjacency_matrix(self, edge_types: list, node_list=None, **kwargs):
        pass
