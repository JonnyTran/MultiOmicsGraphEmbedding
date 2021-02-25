#!/usr/bin/python3
from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import numpy as np
import torch
import torch_geometric
from torch_geometric.utils.hetero import group_hetero_graph


class Sampler(metaclass=ABCMeta):
    @abstractmethod
    def sample(self, node_ids):
        pass

    @abstractmethod
    def get_global_nidx(self, node_ids):
        pass

    @abstractmethod
    def get_nodes_dict(self, adjs, n_id):
        pass


class NeighborSampler(Sampler):
    def __init__(self, neighbor_sizes, edge_index_dict, num_nodes_dict, node_types, head_node_type):
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

        self.neighbor_sampler = torch_geometric.data.NeighborSampler(self.edge_index, node_idx=None,
                                                                     sizes=neighbor_sizes, batch_size=128,
                                                                     shuffle=True)

    def sample(self, node_ids: dict):
        local_node_idx = self.get_global_nidx(node_ids)

        batch_size, n_id, adjs = self.neighbor_sampler.sample(batch=local_node_idx)
        if not isinstance(adjs, list):
            adjs = [adjs]
        return batch_size, n_id, adjs

    def get_global_nidx(self, node_ids):
        if isinstance(node_ids, dict):
            n_idx_to_sample = torch.cat([self.local2global[ntype][nid] for ntype, nid in node_ids.items()], dim=0)
        else:
            n_idx_to_sample = self.local2global[self.head_node_type][node_ids]

        return n_idx_to_sample

    def get_nodes_dict(self, adjs, n_id):
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

    def get_edge_index_dict(self, adjs, n_id, sampled_local_nodes: dict, filter_nodes: bool):
        """ Conbine all edge_index's and convert local node id to "batch node index" that aligns with `x_dict` and
        `global_node_index`

        :param adjs:
        :param n_id:
        :param sampled_local_nodes:
        :param filter_nodes:
        :return:
        """
        relabel_nodes = {
            node_type: dict(zip(sampled_local_nodes[node_type].numpy(),
                                range(len(sampled_local_nodes[node_type])))
                            ) for node_type in sampled_local_nodes}

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

                # Filter nodes for only head node type
                if filter_nodes is True:
                    # If node_type==self.head_node_type, then remove edge_index with nodes not in allowed_nodes_idx
                    allowed_nodes_idx = self.local2global[self.head_node_type][sampled_local_nodes[self.head_node_type]]

                    if head_type == self.head_node_type and tail_type == self.head_node_type:
                        mask = np.isin(edge_index[0], allowed_nodes_idx) & np.isin(edge_index[1], allowed_nodes_idx)
                        edge_index = edge_index[:, mask]
                    elif head_type == self.head_node_type:
                        mask = np.isin(edge_index[0], allowed_nodes_idx)
                        edge_index = edge_index[:, mask]
                    elif tail_type == self.head_node_type:
                        mask = np.isin(edge_index[1], allowed_nodes_idx)
                        edge_index = edge_index[:, mask]

                # Filter nodes from all node types
                elif filter_nodes == 2:
                    if head_type not in relabel_nodes or tail_type not in relabel_nodes: continue

                    allowed_nodes_idx = torch.cat([self.local2global[ntype][n_ids] \
                                                   for ntype, n_ids in sampled_local_nodes.items()], dim=0)

                    mask = np.isin(edge_index[0], allowed_nodes_idx) & np.isin(edge_index[1], allowed_nodes_idx)
                    edge_index = edge_index[:, mask]

                if edge_index.shape[1] == 0: continue

                # Convert node global index -> local index -> batch index
                edge_index[0] = self.local_node_idx[edge_index[0]].apply_(relabel_nodes[head_type].get)
                edge_index[1] = self.local_node_idx[edge_index[1]].apply_(relabel_nodes[tail_type].get)

                edge_index_dict.setdefault(metapath, []).append(edge_index)

        # Join edges from the adjs (from iterative layer-wise sampling)
        edge_index_dict = {metapath: torch.cat(e_index_list, dim=1) \
                           for metapath, e_index_list in edge_index_dict.items()}

        # # Ensure no duplicate edges in each metapath
        # edge_index_dict = {metapath: edge_index[:, nonduplicate_indices(edge_index)] \
        #                    for metapath, edge_index in edge_index_dict.items()}
        return edge_index_dict
