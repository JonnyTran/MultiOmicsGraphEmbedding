#!/usr/bin/python3

from collections import defaultdict

import numpy as np
import torch

from torch.utils.data import Dataset
from moge.generator import HeteroNeighborSampler, TripletSampler
from moge.module.PyG.latte import tag_negative, untag_negative, is_negative
from moge.module.sampling import negative_sample_head_tail
from moge.module.utils import tensor_sizes


class BidirectionalSampler(TripletSampler, HeteroNeighborSampler):

    def __init__(self, dataset, neighbor_sizes=[40], negative_sampling_size=128, test_negative_sampling_size=500,
                 node_types=None, metapaths=None, head_node_type=None, directed=True,
                 resample_train=None, add_reverse_metapaths=True, **kwargs):
        super().__init__(dataset, neighbor_sizes=neighbor_sizes, node_types=node_types, metapaths=metapaths,
                         head_node_type=head_node_type, directed=directed, resample_train=resample_train,
                         add_reverse_metapaths=add_reverse_metapaths, **kwargs)
        self.negative_sampling_size = negative_sampling_size
        self.test_negative_sampling_size = test_negative_sampling_size

    # def __init__(self, dataset, node_types=None, metapaths=None,
    #              negative_sampling_size=128, test_negative_sampling_size=500,
    #              head_node_type=None, directed=True,
    #              resample_train=None, add_reverse_metapaths=True):
    #     super().__init__(dataset, node_types, metapaths, head_node_type, directed, resample_train,
    #                      add_reverse_metapaths)
    #
    # self.negative_sampling_size = negative_sampling_size
    # self.test_negative_sampling_size = test_negative_sampling_size

    def get_collate_fn(self, collate_fn: str, mode=None):
        assert mode is not None, "Must pass arg `mode` at get_collate_fn(). {'train', 'valid', 'test'}"

        def collate_wrapper(iloc):
            return self.sample(iloc, mode=mode)

        return collate_wrapper

    def sample(self, e_idx, mode):
        if not isinstance(e_idx, torch.Tensor):
            e_idx = torch.tensor(e_idx)

        if "test" in mode:
            negative_sampling_size = self.test_negative_sampling_size
        elif "valid" in mode:
            negative_sampling_size = self.negative_sampling_size * 2
        else:
            negative_sampling_size = self.negative_sampling_size

        negative_sampling_size = int(negative_sampling_size / 2)

        triples = {k: v[e_idx] for k, v in self.triples.items() if not is_negative(k)}
        # Add neg edges if valid or test
        # if e_idx.max() < self.start_idx["train"]:
        #     triples.update({k: v[e_idx] for k, v in self.triples.items() if is_negative(k)})

        relation_ids_all = triples["relation"].unique()

        global_node_index = self.get_global_node_index(triples, relation_ids_all, metapaths=self.metapaths)

        # Positive edges
        pos_edges = self.get_local_edge_index(triples=triples,
                                              global_node_index=global_node_index,
                                              relation_ids_all=relation_ids_all,
                                              metapaths=self.metapaths)

        # Negative edges
        neg_head = {}
        neg_tail = {}
        for metapath, edge_index in pos_edges.items():
            neg_head[metapath] = \
                torch.randint(high=len(global_node_index[metapath[0]]),
                              size=(edge_index.shape[1], negative_sampling_size,))
            neg_tail[metapath] = \
                torch.randint(high=len(global_node_index[metapath[-1]]),
                              size=(edge_index.shape[1], negative_sampling_size,))

        # Neighbor sampling with global_node_index
        batch_nodes_global = torch.cat([self.local2global[ntype][nid] for ntype, nid in global_node_index.items()], 0)
        batch_size, n_id, adjs = self.neighbor_sampler.sample(batch_nodes_global)
        if not isinstance(adjs, list):
            adjs = [adjs]

        local2batch = {node_type: dict(zip(global_node_index[node_type].numpy(),
                                           range(len(global_node_index[node_type])))) \
                       for node_type in global_node_index}

        edge_index_dict = self.get_local_edge_index_dict(adjs=adjs, n_id=n_id,
                                                         sampled_local_nodes=global_node_index,
                                                         local2batch=local2batch,
                                                         filter_nodes=2)

        if self.use_reverse:
            self.add_reverse_edge_index(edge_index_dict)

        # Make x_dict
        if hasattr(self, "x_dict") and len(self.x_dict) > 0:
            node_feats = {node_type: self.x_dict[node_type][global_node_index[node_type]] \
                          for node_type in self.x_dict}
        else:
            node_feats = {}

        # Ensure no edges are outside of batch_nodes_ids
        # for m, edge_index in pos_edges.items():
        #     assert set(global_node_index[m[0]][edge_index[0]].tolist()) <= set(local2batch[m[0]].keys()), f"{m}, {m[0]}"
        #     assert set(global_node_index[m[-1]][edge_index[1]].tolist()) <= set(local2batch[m[-1]].keys()), f"{m}, {m[-1]}"
        #
        # for m, edge_index in edge_index_dict.items():
        #     assert set(global_node_index[m[0]][edge_index[0]].tolist()) <= set(local2batch[m[0]].keys()), f"{m}, {m[0]}"
        #     assert set(global_node_index[m[-1]][edge_index[1]].tolist()) <= set(local2batch[m[-1]].keys()), f"{m}, {m[-1]}"

        X = {"edge_index_dict": edge_index_dict,
             "edge_pos": pos_edges,
             "edge_neg_head": neg_head,
             "edge_neg_tail": neg_tail,
             "global_node_index": global_node_index,
             "x_dict": node_feats}

        return X, None, None



class TrainDataset(Dataset):
    def __init__(self, triples, nentity, nrelation, mode=None, negative_sample_size=128, count=None, true_head=None,
                 true_tail=None, entity_dict=defaultdict()):
        self.entity_dict = entity_dict
        if true_tail is None:
            self.true_tail = defaultdict(list)
        if true_head is None:
            self.true_head = defaultdict(list)
        if count is None:
            self.count = defaultdict(lambda: 4)

        self.len = len(triples['head'])
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.mode = mode

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        head, relation, tail = self.triples['head'][idx], self.triples['relation'][idx], self.triples['tail'][idx]
        head_type, tail_type = self.triples['head_type'][idx], self.triples['tail_type'][idx]
        positive_sample = [head + self.entity_dict[head_type][0],
                           relation, tail + self.entity_dict[tail_type][0]]

        subsampling_weight = self.count[(head, relation, head_type)] + self.count[(tail, -relation - 1, tail_type)]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))

        if self.mode == 'head-batch':
            negative_sample = torch.randint(self.entity_dict[head_type][0], self.entity_dict[head_type][1],
                                            (self.negative_sample_size,))
        elif self.mode == 'tail-batch':
            negative_sample = torch.randint(self.entity_dict[tail_type][0], self.entity_dict[tail_type][1],
                                            (self.negative_sample_size,))
        else:
            raise
        positive_sample = torch.LongTensor(positive_sample)

        return positive_sample, negative_sample, subsampling_weight

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        return positive_sample, negative_sample, subsample_weight


