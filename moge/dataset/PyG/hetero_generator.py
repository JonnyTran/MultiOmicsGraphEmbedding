from typing import List, Tuple, Union, Dict

import torch
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader

from moge.dataset.graph import HeteroGraphDataset


class HeteroDataSampler(HeteroGraphDataset):

    def __init__(self, dataset: HeteroData,
                 neighbor_sizes: Union[List[int], Dict[str, List[int]]] = [128, 128],
                 node_types: List[str] = None, metapaths: List[Tuple[str, str, str]] = None, head_node_type: str = None,
                 edge_dir: str = "in", reshuffle_train: float = None, add_reverse_metapaths: bool = True,
                 inductive: bool = False, **kwargs):
        super().__init__(dataset, node_types, metapaths, head_node_type, edge_dir, reshuffle_train,
                         add_reverse_metapaths, inductive, **kwargs)

        self.neighbor_sizes = neighbor_sizes

    def process_pyg_heterodata(self, hetero: HeteroData):
        self.G = hetero
        self.x_dict = {}
        self.node_types = hetero.node_types
        self.num_nodes_dict = {ntype: hetero[ntype].num_nodes for ntype in hetero.node_types}
        self.y_dict = {ntype: hetero[ntype].y for ntype in hetero.node_types}

        self.metapaths = hetero.edge_types
        self.edge_index_dict = {etype: edge_index for etype, edge_index in zip(hetero.edge_types, hetero.edge_stores)}

    @classmethod
    def from_pyg_heterodata(cls, hetero: HeteroData, labels: Union[Tensor, Dict[str, Tensor]],
                            classes: List[str],
                            train_idx: Dict[str, Tensor],
                            val_idx: Dict[str, Tensor],
                            test_idx: Dict[str, Tensor], directed=True, **kwargs):
        self = cls(dataset=hetero, metapaths=hetero.edge_types, add_reverse_metapaths=False,
                   edge_dir="in" if directed else "out", **kwargs)
        self.classes = classes
        self._name = ""

        self.x_dict = {}
        self.node_types = hetero.node_types
        self.num_nodes_dict = {ntype: hetero[ntype].num_nodes for ntype in hetero.node_types}
        self.y_dict = {ntype: hetero[ntype].y for ntype in hetero.node_types}

        self.metapaths = hetero.edge_types
        self.edge_index_dict = {etype: edge_index for etype, edge_index in zip(hetero.edge_types, hetero.edge_stores)}

        self.train_idx = train_idx
        self.val_idx = val_idx
        self.test_idx = test_idx

        return self

    def sample(self, batch: HeteroData, collate_fn=False):
        X = {}
        #
        X["x_dict"] = batch.x_dict
        X["edge_index_dict"] = batch.edge_index_dict
        X["global_node_index"] = batch.nid_dict
        X['sizes'] = batch.num_nodes_dict

        y_dict = batch.y_dict

        if len(y_dict) == 1:
            y_dict = y_dict[list(y_dict.keys()).pop()]

            if y_dict.dim() == 2 and y_dict.size(1) == 1:
                y_dict = y_dict.squeeze(-1)
            elif y_dict.dim() == 1:
                weights = (y_dict >= 0).to(torch.float)

        elif len(y_dict) > 1:
            weights = {}
            for ntype, label in y_dict.items():
                if label.dim() == 2 and label.size(1) == 1:
                    y_dict[ntype] = label.squeeze(-1)

                if label.dim() == 1:
                    weights[ntype] = (y_dict >= 0).to(torch.float)
                elif label.dim() == 2:
                    weights[ntype] = (label.sum(1) > 0).to(torch.float)

        return X, y_dict, weights

    def train_dataloader(self, collate_fn=None, batch_size=128, num_workers=0, **kwargs):
        neighbor_sampler = NeighborLoader(self.G, num_neighbors=self.neighbor_sizes,
                                          batch_size=batch_size,
                                          directed=True,
                                          transform=self.sample,
                                          input_nodes=(self.head_node_type, self.G[self.head_node_type].train_mask),
                                          shuffle=True, num_workers=num_workers, **kwargs)

        return neighbor_sampler

    def valid_dataloader(self, collate_fn=None, batch_size=128, num_workers=0, **kwargs):
        neighbor_sampler = NeighborLoader(self.G, num_neighbors=self.neighbor_sizes,
                                          batch_size=batch_size,
                                          directed=False,
                                          transform=self.sample,
                                          input_nodes=(self.head_node_type, self.G[self.head_node_type].valid_mask),
                                          shuffle=True, num_workers=num_workers, **kwargs)

        return neighbor_sampler

    def test_dataloader(self, collate_fn=None, batch_size=128, num_workers=0, **kwargs):
        neighbor_sampler = NeighborLoader(self.G, num_neighbors=self.neighbor_sizes,
                                          batch_size=batch_size,
                                          directed=False,
                                          transform=self.sample,
                                          input_nodes=(self.head_node_type, self.G[self.head_node_type].test_mask),
                                          shuffle=True, num_workers=num_workers, **kwargs)

        return neighbor_sampler
