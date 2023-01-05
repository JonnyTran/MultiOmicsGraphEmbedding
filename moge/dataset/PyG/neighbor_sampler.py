#!/usr/bin/python3
from typing import Union, List, Optional, Callable, Any, Dict, Tuple

import torch
from torch import Tensor
from torch_geometric.data import Data, HeteroData
from torch_geometric.data.feature_store import FeatureStore
from torch_geometric.data.graph_store import GraphStore
from torch_geometric.loader import HGTLoader
from torch_geometric.loader.neighbor_loader import NumNeighbors, NeighborSampler, NeighborLoader, get_input_nodes
from torch_geometric.sampler import HGTSampler
from torch_geometric.sampler.base import NodeSamplerInput, HeteroSamplerOutput
from torch_geometric.sampler.utils import remap_keys
from torch_geometric.typing import InputNodes, NodeType


class NeighborSamplerX(NeighborSampler):

    def __init__(self, data: Union[Data, HeteroData, Tuple[FeatureStore, GraphStore]], num_neighbors: NumNeighbors,
                 replace: bool = False, directed: bool = True, input_type: Optional[Any] = None,
                 time_attr: Optional[str] = None, is_sorted: bool = False, share_memory: bool = False,
                 class_indices: Dict[str, Tensor] = None):
        super().__init__(data, num_neighbors, replace, directed, input_type, time_attr, is_sorted, share_memory)
        self.class_indices = class_indices

    def __call__(self, index: Union[List[int], Tensor]):
        if not isinstance(index, torch.LongTensor):
            index = torch.LongTensor(index)

        if self.data_cls != 'custom' and issubclass(self.data_cls, Data):
            return self._sparse_neighbor_sample(index) + (index.numel(),)

        elif self.data_cls == 'custom' or issubclass(self.data_cls,
                                                     HeteroData):
            query_nodes = {self.input_type: index}
            if self.class_indices is not None:
                query_nodes.update(self.class_indices)

            return self._hetero_sparse_neighbor_sample(query_nodes) + (index.numel(),)


# def __call__(self, nids: Union[LongTensor, Dict[str, Tensor]]):
#     if isinstance(nids, dict):
#         query_nodes = {ntype: torch.LongTensor(nids) if not isinstance(nids, torch.LongTensor) else nids \
#                        for ntype, nids in nids.items()}
#     else:
#         query_nodes = {self.input_type: torch.LongTensor(nids) \
#             if not isinstance(nids, torch.LongTensor) else nids}
#
#     if self.class_indices is not None:
#         query_nodes.update(self.class_indices)
#         batch_size = sum([nids.numel() for ntype, nids in query_nodes.items() if ntype not in self.class_indices])
#     else:
#         batch_size = sum([nids.numel() for ntype, nids in query_nodes.items()])
#
#     if issubclass(self.data_cls, Data):
#         sample_fn = torch.ops.torch_sparse.neighbor_sample
#         node, row, col, edge = sample_fn(
#             self.colptr,
#             self.row,
#             query_nodes,
#             self.num_neighbors,
#             self.replace,
#             self.directed,
#         )
#         return node, row, col, edge, batch_size
#
#     elif issubclass(self.data_cls, HeteroData):
#         sample_fn = torch.ops.torch_sparse.hetero_neighbor_sample
#         node_dict, row_dict, col_dict, edge_dict = sample_fn(
#             self.node_types,
#             self.edge_types,
#             self.colptr_dict,
#             self.row_dict,
#             query_nodes,
#             self.num_neighbors,
#             self.num_hops,
#             self.replace,
#             self.directed,
#         )
#         return node_dict, row_dict, col_dict, edge_dict, batch_size

class NeighborLoaderX(NeighborLoader):

    def __init__(self, data: Union[Data, HeteroData, Tuple[FeatureStore, GraphStore]], num_neighbors: NumNeighbors,
                 input_nodes: InputNodes = None, replace: bool = False, directed: bool = True,
                 time_attr: Optional[str] = None, transform: Callable = None, is_sorted: bool = False,
                 filter_per_worker: bool = False, neighbor_sampler: Optional[NeighborSampler] = None,
                 class_indices: Dict[str, Tensor] = None,
                 **kwargs):
        # Remove for PyTorch Lightning:
        if 'dataset' in kwargs:
            del kwargs['dataset']
        if 'collate_fn' in kwargs:
            del kwargs['collate_fn']

        self.data = data

        # Save for PyTorch Lightning < 1.6:
        self.num_neighbors = num_neighbors
        self.input_nodes = input_nodes
        self.replace = replace
        self.directed = directed
        self.transform = transform
        self.filter_per_worker = filter_per_worker
        self.neighbor_sampler = neighbor_sampler

        node_type, input_nodes = get_input_nodes(data, input_nodes)

        if neighbor_sampler is None:
            self.neighbor_sampler = NeighborSamplerX(
                data,
                num_neighbors,
                replace,
                directed,
                input_type=node_type,
                time_attr=time_attr,
                is_sorted=is_sorted,
                share_memory=kwargs.get('num_workers', 0) > 0,
                class_indices=class_indices,
            )

        super(NeighborLoader, self).__init__(input_nodes, collate_fn=self.collate_fn, **kwargs)


class HGTSamplerX(HGTSampler):

    def __init__(self, data: HeteroData, num_neighbors: Union[List[int], Dict[NodeType, List[int]]],
                 input_type: Optional[Any] = None, is_sorted: bool = False, share_memory: bool = False,
                 class_indices: Dict[str, Tensor] = None, ):
        """

        Args:
            data ():
            num_neighbors ():
            input_type ():
            is_sorted ():
            share_memory ():
            class_indices ():
        """
        self.class_indices = class_indices
        if self.class_indices:
            # Filter out -1 values which indicate non-matching class indices
            for ntype, idx in self.class_indices.items():
                self.class_indices[ntype] = idx[idx != -1]

        super().__init__(data, num_samples=num_neighbors, input_type=input_type, is_sorted=is_sorted,
                         share_memory=share_memory)

    def sample_from_nodes(
            self,
            index: NodeSamplerInput,
            **kwargs,
    ) -> HeteroSamplerOutput:
        index, input_nodes, _ = index
        query_nodes = {self.input_type: input_nodes}

        if self.class_indices is not None:
            query_nodes.update(self.class_indices)

        sample_fn = torch.ops.torch_sparse.hgt_sample
        out = sample_fn(
            self.colptr_dict,
            self.row_dict,
            query_nodes,
            self.num_samples,
            self.num_hops,
        )
        node, row, col, edge, batch = out + (None,)
        return HeteroSamplerOutput(
            node=node,
            row=remap_keys(row, self.to_edge_type),
            col=remap_keys(col, self.to_edge_type),
            edge=remap_keys(edge, self.to_edge_type),
            batch=batch,
            metadata=index,
        )


class HGTLoaderX(HGTLoader):
    def __init__(
            self,
            data: HeteroData,
            num_neighbors: Union[List[int], Dict[NodeType, List[int]]],
            input_nodes: Union[NodeType, Tuple[NodeType, Optional[Tensor]]],
            is_sorted: bool = False,
            transform: Callable = None,
            filter_per_worker: bool = False,
            class_indices: Dict[str, Tensor] = None,
            **kwargs,
    ):
        """

        Args:
            data ():
            num_neighbors ():
            input_nodes ():
            is_sorted ():
            transform ():
            filter_per_worker ():
            class_indices ():
            **kwargs ():
        """
        node_type, _ = get_input_nodes(data, input_nodes)

        hgt_sampler = HGTSamplerX(
            data,
            num_neighbors=num_neighbors,
            input_type=node_type,
            is_sorted=is_sorted,
            share_memory=kwargs.get('num_workers', 0) > 0,
            class_indices=class_indices,
        )

        super(HGTLoader, self).__init__(
            data=data,
            node_sampler=hgt_sampler,
            input_nodes=input_nodes,
            transform=transform,
            filter_per_worker=filter_per_worker,
            **kwargs,
        )
