from typing import List, Tuple, Union, Dict

import torch
from ogb.nodeproppred import PygNodePropPredDataset
from torch import Tensor
from torch_geometric.data import HeteroData

from moge.dataset.graph import HeteroGraphDataset


class HeteroDataSampler(HeteroGraphDataset):

    def __init__(self, dataset: HeteroData,
                 node_types: List[str] = None, metapaths: List[Tuple[str, str, str]] = None, head_node_type: str = None,
                 edge_dir: str = "in", reshuffle_train: float = None, add_reverse_metapaths: bool = True,
                 inductive: bool = False, **kwargs):
        super().__init__(dataset, node_types, metapaths, head_node_type, edge_dir, reshuffle_train,
                         add_reverse_metapaths, inductive, **kwargs)

    @classmethod
    def from_pyg_heterodata(cls, hetero: HeteroData, labels: Union[Tensor, Dict[str, Tensor]],
                            num_classes: int,
                            train_idx: Dict[str, Tensor],
                            val_idx: Dict[str, Tensor],
                            test_idx: Dict[str, Tensor], **kwargs):

        self = cls(dataset=hetero, metapaths=hetero.edge_types, **kwargs)
        self._name = ""

        self.x_dict = {}
        self.node_types = hetero.node_types
        self.num_nodes_dict = {ntype: hetero[ntype].num_nodes for ntype in hetero.node_types}
        self.y_dict = {}

        self.edge_index_dict = {etype: edge_index for etype, edge_index in zip(hetero.edge_types, hetero.edge_stores)}

        return self

    def process_PygNodeDataset_hetero(self, dataset: PygNodePropPredDataset, ):
        data = dataset[0]
        self._name = dataset.name

        if hasattr(data, "y_dict"):
            self.y_dict = data.y_dict
        elif hasattr(data, "y"):
            self.y_dict = {self.head_node_type: data.y}
        else:
            self.y_dict = {}

        self.y_index_dict = {node_type: torch.arange(self.num_nodes_dict[node_type]) for node_type in
                             self.y_dict.keys()}

        if self.head_node_type is None:
            if hasattr(self, "y_dict"):
                self.head_node_type = list(self.y_dict.keys())[0]
            else:
                self.head_node_type = self.node_types[0]

        self.metapaths = list(self.edge_index_dict.keys())

        split_idx = dataset.get_idx_split()
        self.training_idx, self.validation_idx, self.testing_idx = split_idx["train"][self.head_node_type], \
                                                                   split_idx["valid"][self.head_node_type], \
                                                                   split_idx["test"][self.head_node_type]
