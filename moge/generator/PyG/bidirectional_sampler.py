#!/usr/bin/python3

from collections import defaultdict

import numpy as np
import torch

from torch.utils.data import Dataset
from moge.generator.PyG.triplet_sampler import HeteroNetDataset, TripletSampler
from moge.module.PyG.latte import tag_negative, untag_negative, is_negative
from moge.module.sampling import negative_sample_head_tail


class BidirectionalSampler(TripletSampler):

    def __init__(self, dataset, node_types=None, metapaths=None,
                 negative_sampling_size=128, mode=["tail", "head"],
                 head_node_type=None, directed=True,
                 resample_train=None, add_reverse_metapaths=True):
        super().__init__(dataset, node_types, metapaths, head_node_type, directed, resample_train,
                         add_reverse_metapaths)

        self.negative_sampling_size = negative_sampling_size
        self.mode = mode

    def sample(self, e_idx, mode=None):
        if not isinstance(e_idx, torch.Tensor):
            e_idx = torch.tensor(e_idx)

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
                torch.randint(len(global_node_index[metapath[0]]), (edge_index.shape[1], self.negative_sampling_size,))
            neg_tail[metapath] = \
                torch.randint(len(global_node_index[metapath[-1]]), (edge_index.shape[1], self.negative_sampling_size,))

        # Do this last to avoid sampling negative edges on the reverse metapaths
        if self.use_reverse:
            self.add_reverse_edge_index(pos_edges)

        # Make x_dict
        if hasattr(self, "x_dict") and len(self.x_dict) > 0:
            node_feats = {node_type: self.x_dict[node_type][global_node_index[node_type]] \
                          for node_type in self.x_dict}
        else:
            node_feats = {}

        X = {"edge_index_dict": pos_edges,
             "edge_neg_head": neg_head,
             "edge_neg_tail": neg_tail,
             "global_node_index": global_node_index,
             "x_dict": node_feats}

        return X, None, None


def tag_negative_head(metapath):
    if isinstance(metapath, tuple):
        return metapath + ("neg", "head")
    elif isinstance(metapath, str):
        return metapath + "_neg_head"
    else:
        return "neg"


def tag_negative_tail(metapath):
    if isinstance(metapath, tuple):
        return metapath + ("neg", "tail")
    elif isinstance(metapath, str):
        return metapath + "_neg_tail"
    else:
        return "neg"


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


class TestDataset(Dataset):
    def __init__(self, triples, args, mode, random_sampling, entity_dict):
        self.len = len(triples['head'])
        self.triples = triples
        self.nentity = args.nentity
        self.nrelation = args.nrelation
        self.mode = mode
        self.random_sampling = random_sampling
        if random_sampling:
            self.neg_size = args.neg_size_eval_train
        self.entity_dict = entity_dict

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        head, relation, tail = self.triples['head'][idx], self.triples['relation'][idx], self.triples['tail'][idx]
        head_type, tail_type = self.triples['head_type'][idx], self.triples['tail_type'][idx]
        positive_sample = torch.LongTensor(
            (head + self.entity_dict[head_type][0], relation, tail + self.entity_dict[tail_type][0]))

        if self.mode == 'head-batch':
            if not self.random_sampling:
                negative_sample = torch.cat([torch.LongTensor([head + self.entity_dict[head_type][0]]),
                                             torch.from_numpy(
                                                 self.triples['head_neg'][idx] + self.entity_dict[head_type][0])])
            else:
                negative_sample = torch.cat([torch.LongTensor([head + self.entity_dict[head_type][0]]),
                                             torch.randint(self.entity_dict[head_type][0],
                                                           self.entity_dict[head_type][1], size=(self.neg_size,))])
        elif self.mode == 'tail-batch':
            if not self.random_sampling:
                negative_sample = torch.cat([torch.LongTensor([tail + self.entity_dict[tail_type][0]]),
                                             torch.from_numpy(
                                                 self.triples['tail_neg'][idx] + self.entity_dict[tail_type][0])])
            else:
                negative_sample = torch.cat([torch.LongTensor([tail + self.entity_dict[tail_type][0]]),
                                             torch.randint(self.entity_dict[tail_type][0],
                                                           self.entity_dict[tail_type][1], size=(self.neg_size,))])

        return positive_sample, negative_sample, self.mode

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        mode = data[0][2]

        return positive_sample, negative_sample, mode


class BidirectionalOneShotIterator(object):
    def __init__(self, dataloader_head, dataloader_tail):
        self.iterator_head = self.one_shot_iterator(dataloader_head)
        self.iterator_tail = self.one_shot_iterator(dataloader_tail)
        self.step = 0

    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            data = next(self.iterator_head)
        else:
            data = next(self.iterator_tail)
        return data

    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data
