#!/usr/bin/python3
import copy
from abc import ABCMeta, abstractmethod
from typing import Union, List, Optional, Callable, Any, Dict, Tuple

import torch
from torch import Tensor, LongTensor
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader.base import BaseDataLoader
from torch_geometric.loader.neighbor_loader import NumNeighbors, get_input_node_type, \
    get_input_node_indices
from torch_geometric.loader.neighbor_sampler import EdgeIndex, Adj
from torch_geometric.loader.utils import filter_data, to_hetero_csc, filter_node_store_, \
    edge_type_to_str, filter_edge_store_, to_csc
from torch_geometric.typing import InputNodes, NodeType, OptTensor
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_sparse import SparseTensor


class Sampler(metaclass=ABCMeta):
    @abstractmethod
    def sample(self, node_ids):
        """
        Args:
            node_ids:
        """
        pass

    @abstractmethod
    def get_global_nidx(self, node_ids):
        """
        Args:
            node_ids:
        """
        pass

    @abstractmethod
    def get_local_nodes(self, n_id):
        """
        Args:
            n_id:
        """
        pass


class HeteroNeighborSampler(torch.utils.data.DataLoader):
    def __init__(self, edge_index: Union[Tensor, SparseTensor],
                 sizes: List[int],
                 node_idx: Optional[Tensor] = None,
                 num_nodes: Optional[int] = None,
                 return_e_id: bool = True,
                 **kwargs):

        """
        Args:
            edge_index:
            sizes:
            node_idx:
            num_nodes:
            return_e_id (bool):
            **kwargs:
        """
        self.sizes = sizes
        self.return_e_id = return_e_id
        self.is_sparse_tensor = isinstance(edge_index, SparseTensor)
        self.__val__ = None

        # Obtain a *transposed* `SparseTensor` instance.
        edge_index = edge_index.to('cpu')
        if not self.is_sparse_tensor:
            num_nodes = maybe_num_nodes(edge_index, num_nodes)
            value = torch.arange(edge_index.size(1)) if return_e_id else None

            # Sampling source_to_target
            self.adj_t = SparseTensor(row=edge_index[1],
                                      col=edge_index[0],
                                      value=value,
                                      sparse_sizes=(num_nodes, num_nodes)).t()
        else:
            adj_t = edge_index
            if return_e_id:
                self.__val__ = adj_t.storage.value()
                value = torch.arange(adj_t.nnz())
                adj_t = adj_t.set_value(value, layout='coo')
            self.adj_t = adj_t

        self.adj_t.storage.rowptr()

        if node_idx is None:
            node_idx = torch.arange(self.adj_t.sparse_size(0))
        elif node_idx.dtype == torch.bool:
            node_idx = node_idx.nonzero(as_tuple=False).view(-1)

        super(HeteroNeighborSampler, self).__init__(
            node_idx.view(-1).tolist(), collate_fn=self.sample, **kwargs)

    def sample(self, batch):
        """
        Args:
            batch:
        """
        if not isinstance(batch, Tensor):
            batch = torch.tensor(batch)

        batch_size: int = len(batch)

        adjs = []
        n_id = batch
        for size in self.sizes:
            adj_t, n_id = self.adj_t.sample_adj(n_id, size, replace=False)
            e_id = adj_t.storage.value()
            size = adj_t.sparse_sizes()[::-1]
            if self.__val__ is not None:
                adj_t.set_value_(self.__val__[e_id], layout='coo')

            if self.is_sparse_tensor:
                adjs.append(Adj(adj_t, e_id, size))
            else:
                row, col, _ = adj_t.coo()
                edge_index = torch.stack([row, col], dim=0)
                adjs.append(EdgeIndex(edge_index, e_id, size))

        return batch_size, n_id, adjs


class NeighborSampler:
    def __init__(
            self,
            data: HeteroData,
            num_neighbors: NumNeighbors,
            replace: bool = False,
            directed: bool = True,
            input_node_type: Optional[str] = None,
    ):
        self.data_cls = data.__class__
        self.num_neighbors = num_neighbors
        self.replace = replace
        self.directed = directed

        if isinstance(data, Data):
            # Convert the graph data into a suitable format for sampling.
            self.colptr, self.row, self.perm = to_csc(data, device='cpu')
            assert isinstance(num_neighbors, (list, tuple))

        elif isinstance(data, HeteroData):
            # Convert the graph data into a suitable format for sampling.
            # NOTE: Since C++ cannot take dictionaries with tuples as key as
            # input, edge type triplets are converted into single strings.
            out = to_hetero_csc(data, device='cpu')
            self.colptr_dict, self.row_dict, self.perm_dict = out

            self.node_types, self.edge_types = data.metadata()
            if isinstance(num_neighbors, (list, tuple)):
                num_neighbors = {key: num_neighbors for key in self.edge_types}
            assert isinstance(num_neighbors, dict)
            self.num_neighbors = {
                edge_type_to_str(key): value
                for key, value in num_neighbors.items()
            }

            self.num_hops = max([len(v) for v in self.num_neighbors.values()])

            assert isinstance(input_node_type, str)
            self.input_node_type = input_node_type

        else:
            raise TypeError(f'NeighborLoader found invalid type: {type(data)}')

    def __call__(self, nids: Union[LongTensor, Dict[str, Tensor]]):
        if isinstance(nids, dict):
            query_nodes = {ntype: torch.LongTensor(nids) if not isinstance(nids, torch.LongTensor) else nids \
                           for ntype, nids in nids.items()}
        else:
            query_nodes = {self.input_node_type: torch.LongTensor(nids) \
                if not isinstance(nids, torch.LongTensor) else nids}

        batch_size = sum([nids.numel() for ntype, nids in query_nodes.items()])

        if issubclass(self.data_cls, Data):
            sample_fn = torch.ops.torch_sparse.neighbor_sample
            node, row, col, edge = sample_fn(
                self.colptr,
                self.row,
                query_nodes,
                self.num_neighbors,
                self.replace,
                self.directed,
            )
            return node, row, col, edge, batch_size

        elif issubclass(self.data_cls, HeteroData):
            sample_fn = torch.ops.torch_sparse.hetero_neighbor_sample
            node_dict, row_dict, col_dict, edge_dict = sample_fn(
                self.node_types,
                self.edge_types,
                self.colptr_dict,
                self.row_dict,
                query_nodes,
                self.num_neighbors,
                self.num_hops,
                self.replace,
                self.directed,
            )
            return node_dict, row_dict, col_dict, edge_dict, batch_size


class NeighborLoader(BaseDataLoader):
    def __init__(
            self,
            data: Union[Data, HeteroData],
            num_neighbors: NumNeighbors,
            input_nodes: InputNodes = None,
            replace: bool = False,
            directed: bool = True,
            transform: Callable = None,
            neighbor_sampler: Optional[NeighborSampler] = None,
            **kwargs,
    ):
        if 'dataset' in kwargs:
            del kwargs['dataset']
        if 'collate_fn' in kwargs:
            del kwargs['collate_fn']

        # Save for PyTorch Lightning:
        self.G = data
        self.num_neighbors = num_neighbors
        self.input_nodes = input_nodes
        self.replace = replace
        self.directed = directed
        self.transform = transform
        self.neighbor_sampler = neighbor_sampler

        if neighbor_sampler is None:
            input_node_type = get_input_node_type(input_nodes)
            self.neighbor_sampler = NeighborSampler(data, num_neighbors,
                                                    replace, directed,
                                                    input_node_type)

        return super().__init__(get_input_node_indices(data, input_nodes),
                                collate_fn=self.neighbor_sampler, **kwargs)

    def transform_fn(self, out: Any) -> Union[Data, HeteroData]:
        if isinstance(self.G, Data):
            node, row, col, edge, batch_size = out
            data = filter_data(self.G, node, row, col, edge,
                               self.neighbor_sampler.perm)
            data.batch_size = batch_size

        elif isinstance(self.G, HeteroData):
            node_dict, row_dict, col_dict, edge_dict, batch_size = out
            print({"edge_types": (len(self.G.edge_types), len(self.G.edge_stores)), "node_dict": len(node_dict),
                   "row_dict": len(row_dict), "col_dict": len(col_dict),
                   "edge_dict": len(edge_dict), 'perm_dict': len(self.neighbor_sampler.perm_dict)})

            data = self.filter_hetero_data(self.G, node_dict, row_dict, col_dict,
                                           edge_dict,
                                           perm_dict=self.neighbor_sampler.perm_dict)
            data[self.neighbor_sampler.input_node_type].batch_size = batch_size
        else:
            print(len(self.G))

        return data if self.transform is None else self.transform(data)

    @classmethod
    def filter_hetero_data(self,
                           data: HeteroData,
                           node_dict: Dict[str, Tensor],
                           row_dict: Dict[str, Tensor],
                           col_dict: Dict[str, Tensor],
                           edge_dict: Dict[str, Tensor],
                           perm_dict: Dict[str, OptTensor],
                           ) -> HeteroData:
        # Filters a heterogeneous data object to only hold nodes in `node` and
        # edges in `edge` for each node and edge type, respectively:
        out = copy.copy(data)

        for node_type in data.node_types:
            filter_node_store_(data[node_type], out[node_type],
                               node_dict[node_type])

        for edge_type in data.edge_types:
            edge_type_str = edge_type_to_str(edge_type)
            if edge_type_str not in edge_dict: continue

            filter_edge_store_(data[edge_type], out[edge_type],
                               row_dict[edge_type_str], col_dict[edge_type_str],
                               edge_dict[edge_type_str], perm_dict[edge_type_str])

        return out

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

    def sample(self, nids: Union[List[int], Tensor]):
        pass


class HGTLoader(BaseDataLoader):
    def __init__(
            self,
            data: HeteroData,
            num_neighbors: Union[List[int], Dict[NodeType, List[int]]],
            input_nodes: Union[NodeType, Tuple[NodeType, Optional[Tensor]]],
            transform: Callable = None,
            **kwargs,
    ):
        if 'collate_fn' in kwargs:
            del kwargs['collate_fn']

        if isinstance(num_neighbors, (list, tuple)):
            num_neighbors = {key: num_neighbors for key in data.node_types}

        if isinstance(input_nodes, str):
            input_nodes = (input_nodes, None)
        assert isinstance(input_nodes, (list, tuple))
        assert len(input_nodes) == 2
        assert isinstance(input_nodes[0], str)
        if input_nodes[1] is None:
            index = torch.arange(data[input_nodes[0]].num_nodes)
            input_nodes = (input_nodes[0], index)
        elif input_nodes[1].dtype == torch.bool:
            index = input_nodes[1].nonzero(as_tuple=False).view(-1)
            input_nodes = (input_nodes[0], index)

        self.G = data
        self.num_samples = num_neighbors
        self.input_nodes = input_nodes
        self.num_hops = max([len(v) for v in num_neighbors.values()])
        self.transform = transform
        self.sample_fn = torch.ops.torch_sparse.hgt_sample

        # Convert the graph data into a suitable format for sampling.
        # NOTE: Since C++ cannot take dictionaries with tuples as key as
        # input, edge type triplets are converted into single strings.
        self.colptr_dict, self.row_dict, self.perm_dict = to_hetero_csc(
            data, device='cpu')

        super().__init__(input_nodes[1].tolist(), collate_fn=self.sample,
                         **kwargs)

    def sample(self, indices: Dict[List[int], Dict[str, List[int]]]) -> HeteroData:
        if isinstance(indices, dict):
            query_nodes = {ntype: torch.tensor(nids) if not isinstance(nids, Tensor) else nids \
                           for ntype, nids in indices.items()}
        else:
            query_nodes = {self.input_nodes[0]: torch.tensor(indices) \
                if not isinstance(indices, Tensor) else indices}

        batch_size = sum([nids.numel() for ntype, nids in query_nodes.items()])

        # input_node_dict = {self.input_nodes[0]: torch.tensor(indices)}
        node_dict, row_dict, col_dict, edge_dict = self.sample_fn(
            self.colptr_dict,
            self.row_dict,
            query_nodes,
            self.num_samples,
            self.num_hops,
        )
        return node_dict, row_dict, col_dict, edge_dict, batch_size

    def transform_fn(self, out: Any) -> HeteroData:
        node_dict, row_dict, col_dict, edge_dict, batch_size = out
        data = self.filter_hetero_data(self.G, node_dict, row_dict, col_dict,
                                       edge_dict, self.perm_dict)
        data[self.input_nodes[0]].batch_size = batch_size

        return data if self.transform is None else self.transform(data)

    def filter_hetero_data(self,
                           data: HeteroData,
                           node_dict: Dict[str, Tensor],
                           row_dict: Dict[str, Tensor],
                           col_dict: Dict[str, Tensor],
                           edge_dict: Dict[str, Tensor],
                           perm_dict: Dict[str, OptTensor],
                           ) -> HeteroData:
        # Filters a heterogeneous data object to only hold nodes in `node` and
        # edges in `edge` for each node and edge type, respectively:
        out = copy.copy(data)

        for node_type in data.node_types:
            if node_type not in node_dict:
                node_dict[node_type] = torch.tensor([], dtype=torch.long)

            filter_node_store_(data[node_type], out[node_type],
                               node_dict[node_type])

        for edge_type in data.edge_types:
            edge_type_str = edge_type_to_str(edge_type)
            if edge_type_str not in row_dict:
                row_dict[edge_type_str] = torch.tensor([], dtype=torch.long)
                col_dict[edge_type_str] = torch.tensor([], dtype=torch.long)
                edge_dict[edge_type_str] = torch.tensor([], dtype=torch.long)
                perm_dict[edge_type_str] = torch.tensor([], dtype=torch.long)

            filter_edge_store_(data[edge_type], out[edge_type],
                               row_dict[edge_type_str], col_dict[edge_type_str],
                               edge_dict[edge_type_str], perm_dict[edge_type_str])

        return out

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
