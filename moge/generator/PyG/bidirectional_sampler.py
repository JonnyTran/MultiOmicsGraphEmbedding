#!/usr/bin/python3

from collections import defaultdict
from tqdm import tqdm

import pandas as pd
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

        df = pd.DataFrame(
            {key: self.triples[key].numpy() if isinstance(self.triples[key], torch.Tensor) else self.triples[key] \
             for key in ["head", "head_type", "relation", "tail", "tail_type"]})

        head_counts = df.groupby(["head", "relation", "head_type"])["tail"].count()
        tail_counts = df.groupby(["tail", "relation", "tail_type"])["head"].count()
        tail_counts.index = tail_counts.index.set_levels(levels=-tail_counts.index.get_level_values(1) - 1,
                                                         level=1,
                                                         verify_integrity=False, )

        head_counts.index = head_counts.index.set_names(["nid", "relation", "ntype"])
        tail_counts.index = tail_counts.index.set_names(["nid", "relation", "ntype"])

        self.degree_counts = head_counts.append(tail_counts).to_dict()  # (node_id, relation, ntype): count

        # relation_counts = self.triples["relation"].bincount()
        # for metapath_id, count in enumerate(relation_counts):
        #     self.train_counts[self.metapaths[metapath_id]] = count.astype(torch.float)

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
                head_type, tail_type = metapath[0], metapath[-1]
                relation_id = self.metapaths.index(metapath)
                # head_batch[metapath] = \
                #     torch.randint(high=len(triplets_node_index[metapath[0]]),
                #                   size=(edge_index.shape[1], negative_sampling_size,))
                # tail_batch[metapath] = \
                #     torch.randint(high=len(triplets_node_index[metapath[-1]]),
                #                   size=(edge_index.shape[1], negative_sampling_size,))

                head_batch[metapath] = torch.multinomial(
                    input=self.get_degrees(triplets_node_index[metapath[0]], relation_id, head_type).type(torch.float),
                    num_samples=edge_index.shape[1] * negative_sampling_size,
                    replacement=True) \
                    .view(edge_index.shape[1], negative_sampling_size)

                tail_batch[metapath] = torch.multinomial(
                    input=self.get_degrees(triplets_node_index[metapath[-1]], -relation_id - 1, tail_type).type(
                        torch.float),
                    num_samples=edge_index.shape[1] * negative_sampling_size,
                    replacement=True) \
                    .view(edge_index.shape[1], negative_sampling_size)

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

        edge_index_dict = self.get_local_edge_index_dict(adjs=adjs,
                                                         n_id=n_id,
                                                         sampled_local_nodes=global_node_index,
                                                         local2batch=local2batch,
                                                         filter_nodes=2)

        if self.use_reverse:
            self.add_reverse_edge_index(edge_index_dict)

        # Make x_dict
        if hasattr(self, "x_dict") and len(self.x_dict) > 0:
            node_feats = {node_type: self.x_dict[node_type][global_node_index[node_type]] for node_type in self.x_dict}
        else:
            node_feats = {}

        # Calculate subsampling weights on each edge_pos
        edge_pos_weights = {}
        if hasattr(self, "degree_counts") and "train" in mode:
            for metapath, edge_index in edges_pos.items():
                head_type, tail_type = metapath[0], metapath[-1]
                relation_id = self.metapaths.index(metapath)

                head_weights = self.get_degrees(global_node_index[head_type][edge_index[0]], relation_id=relation_id,
                                                node_type=head_type)
                tail_weights = self.get_degrees(global_node_index[tail_type][edge_index[1]],
                                                relation_id=-relation_id - 1, node_type=tail_type)

                subsampling_weight = head_weights + tail_weights
                edge_pos_weights[metapath] = torch.sqrt(1.0 / torch.tensor(subsampling_weight, dtype=torch.float))

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

    def get_degrees(self, node_ids: torch.LongTensor, relation_id, node_type):
        return node_ids.apply_(lambda nid: self.degree_counts.get((nid, relation_id, node_type), 1))

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




