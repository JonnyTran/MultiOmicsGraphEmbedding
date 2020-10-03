from moge.generator.PyG.network import HeteroNetDataset


class DGLHeteroDataset(HeteroNetDataset):

    def __init__(self, dataset, node_types=None, metapaths=None, head_node_type=None, directed=True,
                 resample_train: float = None, add_reverse_metapaths=True, inductive=True):
        super().__init__(dataset, node_types, metapaths, head_node_type, directed, resample_train,
                         add_reverse_metapaths, inductive)
