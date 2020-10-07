from ogb.nodeproppred import DglNodePropPredDataset

from moge.generator.network import HeteroNetDataset


class DGLNodeSampler(HeteroNetDataset):
    def __init__(self, dataset: DglNodePropPredDataset, neighbor_sizes, node_types=None, metapaths=None,
                 head_node_type=None, directed=True, resample_train=None, add_reverse_metapaths=True, inductive=False):
        super().__init__(dataset, neighbor_sizes, node_types, metapaths, head_node_type, directed, resample_train,
                         add_reverse_metapaths, inductive)

    def process_DglNodeDataset_hetero(self, dataset):
        graph, labels = dataset[0]
        self._name = dataset.name

        if self.node_types is None:
            self.node_types = graph.ntypes

        self.num_nodes_dict = {ntype: graph.num_nodes(ntype) for ntype in self.node_types}

        self.y_dict = labels

        if self.head_node_type is None:
            if self.y_dict is not None:
                self.head_node_type = list(self.y_dict.keys())[0]
            else:
                self.head_node_type = self.node_types[0]

        self.metapaths = graph.canonical_etypes

        split_idx = dataset.get_idx_split()
        self.training_idx, self.validation_idx, self.testing_idx = split_idx["train"][self.head_node_type], \
                                                                   split_idx["valid"][self.head_node_type], \
                                                                   split_idx["test"][self.head_node_type]

        self.G = graph

    def get_collate_fn(self, collate_fn: str, mode=None):
        return super().get_collate_fn(collate_fn, mode)

    def sample(self, iloc, mode):
        return super().sample(iloc, mode)
