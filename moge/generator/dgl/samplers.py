import logging
import torch
import dgl.function as fn
from dgl.dataloading import MultiLayerNeighborSampler, MultiLayerFullNeighborSampler, NodeCollator, BlockSampler
from dgl.sampling import sample_neighbors, select_topk, PinSAGESampler, RandomWalkNeighborSampler


class ImportanceSampler(BlockSampler):
    def __init__(self, num_layers, return_eids):
        super().__init__(num_layers, return_eids)

    def sample_frontier(self, block_id, g, seed_nodes):
        pass

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        return super().sample_blocks(g, seed_nodes, exclude_eids)
