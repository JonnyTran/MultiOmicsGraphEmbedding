#!/usr/bin/python3
from abc import ABCMeta, abstractmethod
from collections import defaultdict, OrderedDict
from typing import Union, List, Optional

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data.sampler import Adj, EdgeIndex, maybe_num_nodes
from torch_geometric.utils.hetero import group_hetero_graph
from torch_sparse import coalesce, SparseTensor


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
    def get_nodes_dict(self, adjs, n_id):
        """
        Args:
            adjs:
            n_id:
        """
        pass


class HeteroNeighborSampler(torch.utils.data.DataLoader):
    def __init__(self, edge_index: Union[Tensor, SparseTensor],
                 sizes: List[int], node_idx: Optional[Tensor] = None,
                 num_nodes: Optional[int] = None, return_e_id: bool = True,
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
            self.adj_t = SparseTensor(row=edge_index[1], col=edge_index[0],
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


class NeighborSampler(Sampler):
    def __init__(self, neighbor_sizes, edge_index_dict, num_nodes_dict, node_types, head_node_type):
        """
        Args:
            neighbor_sizes:
            edge_index_dict:
            num_nodes_dict:
            node_types:
            head_node_type:
        """
        self.head_node_type = head_node_type

        # Ensure head_node_type is first item in num_nodes_dict, since NeighborSampler.sample() function takes in index only the first
        num_nodes_dict = OrderedDict(
            [(node_type, num_nodes_dict[node_type]) for node_type in node_types])

        self.edge_index, self.edge_type, self.node_type, self.local_node_idx, self.local2global, self.key2int = \
            group_hetero_graph(edge_index_dict, num_nodes_dict)

        self.int2node_type = {type_int: node_type for node_type, type_int in self.key2int.items() if
                              node_type in node_types}
        self.int2edge_type = {type_int: edge_type for edge_type, type_int in self.key2int.items() if
                              edge_type in edge_index_dict}

        self.neighbor_sampler = HeteroNeighborSampler(self.edge_index, node_idx=None,
                                                      sizes=neighbor_sizes, batch_size=128,
                                                      shuffle=True)

    def sample(self, node_ids: dict):
        """
        Args:
            node_ids (dict):
        """
        local_node_idx = self.get_global_nidx(node_ids)

        batch_size, n_id, adjs = self.neighbor_sampler.sample(batch=local_node_idx)
        if not isinstance(adjs, list):
            adjs = [adjs]
        return batch_size, n_id, adjs

    def get_global_nidx(self, node_ids):
        """
        Args:
            node_ids:
        """
        if isinstance(node_ids, dict):
            n_idx_to_sample = torch.cat([self.local2global[ntype][nid] for ntype, nid in node_ids.items()], dim=0)
        else:
            n_idx_to_sample = self.local2global[self.head_node_type][node_ids]

        return n_idx_to_sample

    def get_nodes_dict(self, adjs: List[EdgeIndex], n_id):
        """
        Args:
            adjs:
            n_id:
        """
        sampled_nodes = {}
        for adj in adjs:
            for i in [0, 1]:
                node_ids = n_id[adj.edge_index[i]]
                node_types = self.node_type[node_ids]

                for node_type_id in node_types.unique():
                    mask = node_types == node_type_id
                    local_node_ids = self.local_node_idx[node_ids[mask]]
                    sampled_nodes.setdefault(self.int2node_type[node_type_id.item()], []).append(local_node_ids)

        # Concatenate & remove duplicate nodes
        sampled_nodes = {k: torch.cat(v, dim=0).unique() for k, v in sampled_nodes.items()}
        return sampled_nodes

    def get_edge_index_dict(self, adjs: List[EdgeIndex], n_id, sampled_local_nodes: dict, filter_nodes: bool):
        """Conbine all edge_index's and convert local node id to "batch node
        index" that aligns with `x_dict` and `global_node_index`

        Args:
            adjs:
            n_id:
            sampled_local_nodes (dict):
            filter_nodes (bool):
        """
        relabel_nodes = self.get_nid_relabel_dict(sampled_local_nodes)

        edge_index_dict = {}
        for adj in adjs:
            for edge_type_id in self.edge_type[adj.e_id].unique():
                metapath = self.int2edge_type[edge_type_id.item()]
                head_type, tail_type = metapath[0], metapath[-1]

                # Filter edges to correct edge_type_id
                edge_mask = self.edge_type[adj.e_id] == edge_type_id
                edge_index = adj.edge_index[:, edge_mask]

                # convert from "sampled_edge_index" to global index
                edge_index[0] = n_id[edge_index[0]]
                edge_index[1] = n_id[edge_index[1]]

                # Convert node global index -> local index -> batch index
                if head_type == tail_type:
                    edge_index = self.local_node_idx[edge_index].apply_(relabel_nodes[head_type].get)
                else:
                    edge_index[0] = self.local_node_idx[edge_index[0]].apply_(relabel_nodes[head_type].get)
                    edge_index[1] = self.local_node_idx[edge_index[1]].apply_(relabel_nodes[tail_type].get)

                # Remove edges not in sampled_local_nodes
                mask = np.isin(edge_index, [-1], assume_unique=True, invert=True).all(0)
                edge_index = edge_index[:, mask]

                # # Filter nodes for only head node type
                # if filter_nodes is True:
                #     allowed_nodes_idx = self.local2global[self.head_node_type][sampled_local_nodes[self.head_node_type]]
                #
                #     # If node_type==self.head_node_type, then remove edge_index with any nodes not in allowed_nodes_idx
                #     if head_type == self.head_node_type and tail_type == self.head_node_type:
                #         mask = np.isin(edge_index, allowed_nodes_idx, assume_unique=True).all(0)
                #         edge_index = edge_index[:, mask]
                #     elif head_type == self.head_node_type:
                #         mask = np.isin(edge_index[0], allowed_nodes_idx, assume_unique=True)
                #         edge_index = edge_index[:, mask]
                #     elif tail_type == self.head_node_type:
                #         mask = np.isin(edge_index[1], allowed_nodes_idx, assume_unique=True)
                #         edge_index = edge_index[:, mask]
                #
                # # Filter nodes from all node types
                # elif filter_nodes == 2:
                #     if head_type not in relabel_nodes or tail_type not in relabel_nodes: continue
                #
                #     allowed_nodes_idx = torch.cat([self.local2global[ntype][n_ids] \
                #                                    for ntype, n_ids in sampled_local_nodes.items()], dim=0)
                #
                #     mask = np.isin(edge_index, allowed_nodes_idx, assume_unique=True).all(0)
                #     edge_index = edge_index[:, mask]

                if edge_index.size(1) == 0: continue

                edge_index_dict.setdefault(metapath, []).append(edge_index)

        # Join edges from the adjs (from iterative layer-wise sampling)
        edge_index_dict = {metapath: torch.cat(e_index_list, dim=1) \
                           for metapath, e_index_list in edge_index_dict.items()}

        # Ensure no duplicate edges in each metapath
        edge_index_dict = {metapath: coalesce(index=edge_index, value=torch.ones_like(edge_index[0], dtype=torch.float),
                                              m=sampled_local_nodes[metapath[0]].size(0),
                                              n=sampled_local_nodes[metapath[-1]].size(0),
                                              op="max")[0] \
                           for metapath, edge_index in edge_index_dict.items()}

        return edge_index_dict

    def get_nid_relabel_dict(self, node_ids_dict):
        """
        Args:
            node_ids_dict:
        """
        relabel_nodes = {node_type: defaultdict(lambda: -1, dict(zip(node_ids_dict[node_type].numpy(),
                                                                     range(node_ids_dict[node_type].size(0))))) \
                         for node_type in node_ids_dict}
        return relabel_nodes
