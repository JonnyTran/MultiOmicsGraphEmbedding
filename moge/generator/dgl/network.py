from moge.generator.PyG.node_sampler import HeteroNeighborSampler
import dgl


class DGLHeteroDataset(HeteroNeighborSampler):

    def __init__(self, dataset, neighbor_sizes, node_types=None, metapaths=None, head_node_type=None, directed=True,
                 resample_train=None, add_reverse_metapaths=True, inductive=False):
        super().__init__(dataset, neighbor_sizes, node_types, metapaths, head_node_type, directed, resample_train,
                         add_reverse_metapaths, inductive)

        self.G = dgl.heterograph({m: e.t().numpy().tolist() for m, e in self.edge_index_dict.items()}, )

        for ntype, features in self.x_dict.items():
            self.G.nodes[ntype].data["x"] = features

        for ntype, labels in self.y_dict.items():
            self.G.nodes[ntype].data["y"] = labels
