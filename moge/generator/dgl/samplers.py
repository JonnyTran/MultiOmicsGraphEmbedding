from typing import Dict, Tuple
import logging
import torch

import dgl
import dgl.function as fn
from dgl.dataloading import MultiLayerNeighborSampler, MultiLayerFullNeighborSampler, NodeCollator, BlockSampler
from dgl.sampling import sample_neighbors, select_topk, PinSAGESampler, RandomWalkNeighborSampler


class ImportanceSampler(BlockSampler):
    def __init__(self, fanouts, metapaths, degree_counts: Dict[Tuple, int], edge_dir="in", return_eids=False):
        super().__init__(len(fanouts), return_eids)

        self.fanouts = fanouts
        self.edge_dir = edge_dir
        self.degree_counts = degree_counts
        self.metapaths = metapaths

    def sample_frontier(self, block_id, g: dgl.DGLGraph, seed_nodes: Dict[str, torch.Tensor]):
        # Get all inbound edges to `seed_nodes`
        if self.edge_dir == "in":
            sg = dgl.in_subgraph(g, seed_nodes)
        elif self.edge_dir == "out":
            sg = dgl.out_subgraph(g, seed_nodes)

        sg = self.assign_prob(sg, seed_nodes)

        fanout = self.fanouts[block_id]

        if fanout is None:
            frontier = sg
        else:
            frontier = sample_neighbors(sg, nodes=seed_nodes, fanout=fanout, prob="prob", edge_dir=self.edge_dir)

        return frontier

    def assign_fanout(self, subgraph: dgl.DGLHeteroGraph, seed_nodes: Dict[str, torch.Tensor]):
        pass

    def assign_prob(self, subgraph: dgl.DGLHeteroGraph, seed_nodes: Dict[str, torch.Tensor]):
        for metapath in subgraph.canonical_etypes:
            head_type, tail_type = metapath[0], metapath[-1]
            relation = self.metapaths.index(metapath)

            src, dst = subgraph.all_edges(etype=metapath)
            prob_a = src.apply_(self.degree_counts.get(()))

        return subgraph
