from moge.generator.PyG.node_sampler import HeteroNeighborSampler
import dgl


class DGLHeteroDataset(HeteroNeighborSampler):

    def __init__(self, dataset, node_types=None, metapaths=None, head_node_type=None, directed=True,
                 resample_train: float = None, add_reverse_metapaths=True, inductive=True):
        super().__init__(dataset, node_types, metapaths, head_node_type, directed, resample_train,
                         add_reverse_metapaths, inductive)

        self.G = dgl.heterograph({m: e.t().numpy().tolist() for m, e in dataset.edge_index_dict.items()}, )
        for ntype, ndata in self.x_dict.items():
            self.G.ndata[ntype] = {"x": ndata}

        for ntype, y_labels in self.y_dict.items():
            self.G.ndata[ntype] = {"y": y_labels}
