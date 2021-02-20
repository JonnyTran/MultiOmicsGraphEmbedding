#!/usr/bin/python3

from collections import defaultdict

import numpy as np
import torch

from ogb.linkproppred import PygLinkPropPredDataset

from torch.utils.data import Dataset
from moge.generator import HeteroNeighborSampler, TripletSampler, EdgeSampler
from moge.module.PyG.latte import tag_negative, untag_negative, is_negative
from moge.module.sampling import negative_sample_head_tail
from moge.module.utils import tensor_sizes


class BidirectionalSampler(TripletSampler, HeteroNeighborSampler):

    def __init__(self, dataset: PygLinkPropPredDataset, neighbor_sizes,
                 negative_sampling_size=128, test_negative_sampling_size=500,
                 force_negative_sampling=False,
                 node_types=None, metapaths=None, head_node_type=None, directed=True,
                 resample_train=None, add_reverse_metapaths=True, **kwargs):
        super(BidirectionalSampler, self).__init__(dataset, neighbor_sizes=neighbor_sizes, node_types=node_types,
                                                   metapaths=metapaths,
                                                   head_node_type=head_node_type, directed=directed,
                                                   resample_train=resample_train,
                                                   add_reverse_metapaths=add_reverse_metapaths, **kwargs)
        self.neg_sampling_size = negative_sampling_size
        self.test_neg_sampling_size = test_negative_sampling_size
        self.force_neg_sampling = force_negative_sampling

        self.train_counts = defaultdict(lambda: 4)

        relation_counts = self.triples["relation"].bincount()
        for metapath_id, count in enumerate(relation_counts):
            self.train_counts[self.metapaths[metapath_id]] = count

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

        # Select sampling size
        if "test" in mode:
            negative_sampling_size = self.test_neg_sampling_size
        elif "valid" in mode:
            negative_sampling_size = self.test_neg_sampling_size
        else:
            negative_sampling_size = self.neg_sampling_size
        negative_sampling_size = int(negative_sampling_size)

        triples = {k: v[e_idx] for k, v in self.triples.items() if not is_negative(k)}
        # Add true neg edges if on valid or test triplet indices
        if e_idx.max() < self.start_idx["train"] and not self.force_neg_sampling:
            triples.update({k: v[e_idx] \
                            for k, v in self.triples.items() \
                            if is_negative(k)})

        relation_ids_all = triples["relation"].unique()

        # Set of all nodes from sampled triples
        triplets_node_index = self.get_global_node_index(triples, relation_ids_all, metapaths=self.metapaths)

        # Get true edges from triples
        edges_pos, edges_neg = self.get_local_edge_index(triples=triples,
                                                         global_node_index=triplets_node_index,
                                                         relation_ids_all=relation_ids_all,
                                                         metapaths=self.metapaths)

        # Whether to negative sampling
        if not edges_neg:
            head_batch = {}
            tail_batch = {}
            for metapath, edge_index in edges_pos.items():
                head_batch[metapath] = \
                    torch.randint(high=len(triplets_node_index[metapath[0]]),
                                  size=(edge_index.shape[1], negative_sampling_size,))
                tail_batch[metapath] = \
                    torch.randint(high=len(triplets_node_index[metapath[-1]]),
                                  size=(edge_index.shape[1], negative_sampling_size,))

        # Neighbor sampling with global_node_index
        batch_nodes_global = torch.cat([self.local2global[ntype][nid] for ntype, nid in triplets_node_index.items()], 0)
        batch_size, n_id, adjs = self.neighbor_sampler.sample(batch_nodes_global)
        if not isinstance(adjs, list):
            adjs = [adjs]

        sampled_local_nodes = self.get_local_node_index(adjs, n_id)

        # Merge triplets_node_index + sampled_local_nodes = global_node_index, while ensuring index order in triplets_node_index
        global_node_index = self.merge_node_index(old_node_index=triplets_node_index,
                                                  new_node_index=sampled_local_nodes)

        # Get dict to convert from global node index to batch node index
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

        # Calculate subsampling weights on each edge_pos
        edge_pos_weights = {}
        for metapath, edge_index in edges_pos:
            edge_pos_weights[metapath] = torch.ones(edge_index.shape[1]) * torch.sqrt(
                1 / torch.Tensor(self.train_counts[metapath]))

        # Build X input dict
        X = {"edge_index_dict": edge_index_dict,
             "edge_pos": edges_pos,
             "global_node_index": global_node_index,
             "x_dict": node_feats}

        if not edges_neg:
            X.update({"head-batch": head_batch, "tail-batch": tail_batch, })
        else:
            X.update({"edge_neg": edges_neg})

        return X, None, edge_pos_weights

    def merge_node_index(self, old_node_index, new_node_index):
        merged = {}
        for ntype, new_nodes in new_node_index.items():
            if ntype not in old_node_index:
                merged.setdefault(ntype, []).append(new_nodes)
            else:
                merged.setdefault(ntype, []).append(old_node_index[ntype])
                new_nodes_mask = np.isin(new_nodes, old_node_index[ntype], invert=True)
                merged[ntype].append(new_nodes[new_nodes_mask])
            merged[ntype] = torch.cat(merged[ntype], 0)
        return merged


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


