from typing import Dict, Tuple
import logging
import numpy as np
import pandas as pd
import torch

import dgl
import dgl.function as fn
from dgl.dataloading import MultiLayerNeighborSampler, MultiLayerFullNeighborSampler, NodeCollator, BlockSampler
from dgl.sampling import sample_neighbors, select_topk, random_walk

from moge.module.utils import tensor_sizes

class ImportanceSampler(BlockSampler):
    def __init__(self, fanouts, metapaths, degree_counts: pd.Series, edge_dir="in", return_eids=False):
        """
        Args:
            fanouts:
            metapaths:
            degree_counts (pd.Series):
            edge_dir:
            return_eids:
        """
        super().__init__(len(fanouts), return_eids)

        self.fanouts = fanouts
        self.edge_dir = edge_dir
        self.degree_counts = degree_counts
        self.metapaths = metapaths

    def sample_frontier(self, block_id, g: dgl.DGLGraph, seed_nodes: Dict[str, torch.Tensor]):
        """
        Args:
            block_id:
            g (dgl.DGLGraph):
            seed_nodes:
        """
        fanouts = self.fanouts[block_id]

        if self.edge_dir == "in":
            sg = dgl.in_subgraph(g, seed_nodes)
        elif self.edge_dir == "out":
            sg = dgl.out_subgraph(g, seed_nodes)

        self.assign_prob(sg, seed_nodes)
        print(sg)
        # for metapath in sg.canonical_etypes:
        #     sg.edges[metapath].data["prob"] = torch.rand((sg.edges[metapath].data[dgl.EID].size(0),))

        frontier = sample_neighbors(sg, nodes=seed_nodes, fanout=fanouts, edge_dir=self.edge_dir)

        return frontier

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        """
        Args:
            g:
            seed_nodes:
            exclude_eids:
        """
        return super().sample_blocks(g, seed_nodes, exclude_eids)

    def get_fanout(self, fanout, seed_nodes):
        """
        Args:
            fanout:
            seed_nodes:
        """
        edge_counts = pd.Series(data=1, index=self.degree_counts.head(n=1).index)
        for ntype, nids in seed_nodes.items():
            if nids.size(0) == 0: continue
            edge_counts = edge_counts.add(self.degree_counts.loc[nids.numpy(), :, ntype], fill_value=1)

        etype_counts = edge_counts.groupby("relation").mean()
        etype_counts = etype_counts / etype_counts.sum()

        new_fanouts = {
            etype: np.ceil((0.5 + etype_counts[etype_id]) * fanout).astype(int) if etype_id in etype_counts.index else 1
            for etype_id, etype in enumerate(self.metapaths)}
        return new_fanouts

    def assign_prob(self, subgraph, seed_nodes=None):
        """
        Args:
            subgraph (dgl.DGLHeteroGraph):
            seed_nodes:
        """
        print("gothere")
        for metapath in subgraph.canonical_etypes:
            head_type, tail_type = metapath[0], metapath[-1]
            relation_id = self.metapaths.index(metapath)
            print(metapath)

            src, dst = subgraph.all_edges(etype=metapath)
            if src.size(0) == 0:
                subgraph.edges[metapath].data["prob"] = torch.tensor([])
                continue

            print("src", src)
            print("nid, relation_id, head_type", relation_id, head_type)

            # subsampling_weight = src.apply_(lambda nid: self.degree_counts.get((nid, relation_id, head_type), 1)).type(
            #     torch.float)
            # print(subsampling_weight.shape)

            # if (self.degree_counts.index.get_level_values(1) < 0).any():
            #     subsampling_weight += dst.apply_(
            #         lambda nid: self.degree_counts.get((nid, -relation_id - 1, tail_type), 1)).type(
            #         torch.float)

            # subsampling_weight = torch.sqrt(1.0 / subsampling_weight)
            # subgraph.edges[metapath].data["prob"] = subsampling_weight
