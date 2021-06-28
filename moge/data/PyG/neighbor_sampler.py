#!/usr/bin/python3
from abc import ABCMeta, abstractmethod
from collections import defaultdict, OrderedDict
from typing import Union, List, Optional

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import NeighborSampler as PyGNeighborSampler
from torch_geometric.data.sampler import Adj, EdgeIndex
from torch_geometric.utils.hetero import group_hetero_graph
from torch_geometric.utils.num_nodes import maybe_num_nodes
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

        self.edge_index, self.edge_type, self.node_type, self.global2local, self.local2global, self.key2int = \
            group_hetero_graph(edge_index_dict, num_nodes_dict)

        self.int2node_type = {type_int: node_type for node_type, type_int in self.key2int.items() if
                              node_type in node_types}
        self.int2edge_type = {type_int: edge_type for edge_type, type_int in self.key2int.items() if
                              edge_type in edge_index_dict}

        self.neighbor_sampler = PyGNeighborSampler(self.edge_index, node_idx=None,
                                                   sizes=neighbor_sizes, batch_size=128,
                                                   shuffle=True)

    def sample(self, local_seed_node_ids: dict):
        """
        Args:
            local_seed_node_ids (dict):
        """
        local_node_idx = self.get_global_nidx(local_seed_node_ids)

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

    def get_local_nodes(self, n_id, filter_nodes: torch.Tensor = None):
        """
        Args:
            n_id: maps batch indices from adjs to global node ids
        """
        local_nodes = {}
        node_types = self.node_type[n_id]

        for node_type_id in node_types.unique():
            mask = node_types == node_type_id
            local_node_ids = self.global2local[n_id[mask]]

            ntype = self.int2node_type[node_type_id.item()]
            local_nodes[ntype] = local_node_ids

            # Ensure the sampled nodes only either belongs to training, validation, or testing set
            if filter_nodes is not None and (isinstance(filter_nodes, torch.Tensor) and ntype == self.head_node_type):
                node_mask = np.isin(local_nodes[ntype], filter_nodes)

                # Get the indices in n_id that were filtered out
                n_id_outlier_idx = mask.nonzero().flatten()[~node_mask]
                n_id[n_id_outlier_idx] = -1

        return local_nodes, n_id

    def get_edge_index_dict(self, adjs: List[EdgeIndex], n_id, sampled_local_nodes: dict):
        """Conbine all edge_index's across multiple layers and convert local node id to "batch node
        index" that aligns with `x_dict` and `global_node_index`

        Args:
            adjs:
            n_id:
            sampled_local_nodes (dict):
        """
        local2batch = self.get_local2batch_dict(sampled_local_nodes)

        edge_index_dict = {}
        for adj in adjs:
            for edge_type_id in self.edge_type[adj.e_id].unique():
                metapath = self.int2edge_type[edge_type_id.item()]
                head, tail = metapath[0], metapath[-1]

                # Filter edges to correct edge_type_id
                edge_mask = self.edge_type[adj.e_id] == edge_type_id
                edge_index = adj.edge_index[:, edge_mask]

                # convert from "sampled_edge_index" to global index
                edge_index[0] = n_id[edge_index[0]]
                edge_index[1] = n_id[edge_index[1]]

                # Convert node global index -> local index -> batch index
                if head == tail:
                    edge_index = self.global2local[edge_index].apply_(local2batch[head].get)
                else:
                    edge_index[0] = self.global2local[edge_index[0]].apply_(lambda x: local2batch[head].get(x, -1))
                    edge_index[1] = self.global2local[edge_index[1]].apply_(lambda x: local2batch[tail].get(x, -1))

                # Remove edges labeled as -1, which contain nodes not in sampled_local_nodes
                mask = np.isin(edge_index, [-1], assume_unique=False, invert=True).all(axis=0)
                edge_index = edge_index[:, mask]
                if edge_index.size(1) == 0: continue

                edge_index_dict.setdefault(metapath, []).append(edge_index)

        # Join edges from the adjs (from iterative layer-wise sampling)
        edge_index_dict = {metapath: torch.cat(e_index_list, dim=1) \
                           for metapath, e_index_list in edge_index_dict.items()}

        # Ensure no duplicate edges in each metapath
        edge_index_dict = {metapath: coalesce(index=edge_index,
                                              value=torch.ones_like(edge_index[0], dtype=torch.float),
                                              m=sampled_local_nodes[metapath[0]].size(0),
                                              n=sampled_local_nodes[metapath[-1]].size(0),
                                              op="add")[0] \
                           for metapath, edge_index in edge_index_dict.items()}

        return edge_index_dict

    def get_multi_edge_index_dict(self, adjs: List[EdgeIndex], n_id, local_nodes_dict: dict):
        """Conbine all edge_index's across multiple layers and convert local node id to "batch node
        index" that aligns with `x_dict` and `global_node_index`

        Args:
            adjs: global_batched edge index (local indices for the hetero graph sampler's global index, not original local index)
            n_id: global nodes ordering for adjs
            local_nodes_dict (dict): local nodes (original node ids)
        """
        local2batch = self.get_local2batch_dict(local_nodes_dict)
        local_edges_dict = [{} for i in range(len(adjs))]

        for i, adj in enumerate(adjs):
            for etype_id in self.edge_type[adj.e_id].unique():
                metapath = self.int2edge_type[etype_id.item()]
                head, tail = metapath[0], metapath[-1]

                # Filter edges to correct edge_type_id
                edge_mask = self.edge_type[adj.e_id] == etype_id
                edge_index = adj.edge_index[:, edge_mask]

                # convert from "sampled_edge_index" to global index
                edge_index = n_id[edge_index]

                # Remove edges labeled as -1, which contain nodes not in sampled_local_nodes
                mask = np.isin(edge_index, [-1], assume_unique=False, invert=True).all(axis=0)
                edge_index = edge_index[:, mask]
                if edge_index.size(1) == 0: continue

                # Convert node global index -> local index -> batch index
                # if head == tail:
                #     edge_index = self.global2local[edge_index].apply_(local2batch[head].get)
                # else:
                edge_index[0] = self.global2local[edge_index[0]].apply_(lambda x: local2batch[head].get(x, -1))
                edge_index[1] = self.global2local[edge_index[1]].apply_(lambda x: local2batch[tail].get(x, -1))

                local_edges_dict[i][metapath] = edge_index

        return local_edges_dict

    def get_local2batch_dict(self, local_node_ids: dict):
        """
        Args:
            local_node_ids:
        """
        relabel_nodes = {node_type: defaultdict(lambda: -1,
                                                dict(zip(local_node_ids[node_type].numpy(),
                                                         range(local_node_ids[node_type].size(0))))) \
                         for node_type in local_node_ids}

        # relabel_nodes = {ntype: pd.Series(data=np.arange(node_ids_dict[ntype].size(0)),
        #                                   index=node_ids_dict[ntype].numpy()) \
        #                  for ntype in node_ids_dict}
        return relabel_nodes
