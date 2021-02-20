from collections import defaultdict

import numpy as np
import torch
from ogb.linkproppred import PygLinkPropPredDataset

from moge.generator.network import HeteroNetDataset
from moge.generator.PyG.node_sampler import HeteroNeighborSampler
from moge.module.PyG.latte import tag_negative, is_negative


class EdgeSampler(HeteroNetDataset):
    def __init__(self, *args, **kwargs):
        super(EdgeSampler, self).__init__(*args, **kwargs)

    def process_PygLinkDataset_homo(self, dataset: PygLinkPropPredDataset):
        data = dataset[0]
        self._name = dataset.name
        self.edge_index_dict = {("self", "edge", "self"): data.edge_index}

        if hasattr(data, "num_nodes_dict"):
            self.num_nodes_dict = data.num_nodes_dict
        else:
            self.num_nodes_dict = self.get_num_nodes_dict(self.edge_index_dict)

        if self.node_types is None:
            self.node_types = list(self.num_nodes_dict.keys())

        if hasattr(data, "x") and data.x is not None:
            self.x_dict = {self.head_node_type: data.x}
        elif hasattr(data, "x_dict") and data.x_dict is not None:
            self.x_dict = data.x_dict
        else:
            self.x_dict = {}

        self.metapaths = list(self.edge_index_dict.keys())

        split_idx = dataset.get_edge_split()
        train_triples, valid_triples, test_triples = split_idx["train"], split_idx["valid"], split_idx["test"]
        self.triples = {}

        # Concat pos edges
        for key in train_triples.keys():
            if isinstance(train_triples[key], torch.Tensor):
                self.triples[("self", key, "self")] = torch.cat(
                    [valid_triples[key], test_triples[key], train_triples[key]],
                    dim=0).permute(1, 0)
            else:
                self.triples[("self", key, "self")] = np.array(
                    valid_triples[key] + test_triples[key] + train_triples[key])

        # Concat neg edges
        for key in valid_triples.keys():
            if is_negative(key):  # edge_neg
                self.triples[("self", "neg", "self")] = torch.cat([valid_triples[key], test_triples[key]],
                                                                  dim=0).permute(1, 0)

        # Create samples index for validation, testing, and training
        self.start_idx = {"valid": 0,
                          "test": valid_triples["edge"].shape[0],
                          "train": valid_triples["edge"].shape[0] + test_triples["edge"].shape[0]}

        self.validation_idx = torch.arange(self.start_idx["valid"],
                                           self.start_idx["valid"] + valid_triples["edge"].shape[0])
        self.testing_idx = torch.arange(self.start_idx["test"],
                                        self.start_idx["test"] + test_triples["edge"].shape[0])
        self.training_idx = torch.arange(self.start_idx["train"],
                                         self.start_idx["train"] + train_triples["edge"].shape[0])

        assert self.validation_idx.max() < self.testing_idx.min()
        assert self.testing_idx.max() < self.training_idx.min()

    @staticmethod
    def get_global_node_index(triples):
        # Gather all unique nodes from sampled triples
        global_node_index = {}

        for metapath in triples:
            head_type, tail_type = metapath[0], metapath[-1]

            global_node_index.setdefault(head_type, []).append(triples[metapath][0])
            global_node_index.setdefault(tail_type, []).append(triples[metapath][1])

        # Find union of nodes from all relations
        global_node_index = {node_type: torch.cat(node_sets, dim=0).unique() \
                             for node_type, node_sets in global_node_index.items()}
        return global_node_index

    @staticmethod
    def get_local_edge_index(triples, global_node_index):
        edges_pos = {}
        edges_neg = {}

        local2batch = {
            node_type: dict(zip(
                global_node_index[node_type].numpy(),
                range(len(global_node_index[node_type])))
            ) for node_type in global_node_index}

        # Get edge_index with batch id
        for metapath in triples:
            head_type, tail_type = metapath[0], metapath[-1]

            if not is_negative(metapath):
                sources = triples[metapath][0].apply_(local2batch[head_type].get)
                targets = triples[metapath][1].apply_(local2batch[tail_type].get)
                edges_pos[metapath] = torch.stack([sources, targets], dim=1)

            elif is_negative(metapath):
                sources = triples[metapath][0].apply_(local2batch[head_type].get)
                targets = triples[metapath][1].apply_(local2batch[tail_type].get)
                edges_neg[metapath] = torch.cat([sources, targets], dim=1)

            else:
                raise Exception(f"something wrong with metapath {metapath}")

        return edges_pos, edges_neg

    def get_collate_fn(self, collate_fn: str, batch_size=None, mode=None):
        return self.sample

    def sample(self, iloc):
        if not isinstance(iloc, torch.Tensor):
            iloc = torch.tensor(iloc)

        X = {"edge_index_dict": {}, "global_node_index": {}, "x_dict": {}}

        edge_index = self.edge_index[iloc]
        edge_reltype = self.edge_reltype[iloc]
        reltype_ids = self.edge_reltype[iloc].unique()

        # Gather all nodes sampled
        X["global_node_index"][self.head_node_type] = edge_index.view(-1).unique()

        local2batch = {
            node_type: dict(zip(X["global_node_index"][node_type].numpy(),
                                range(len(X["global_node_index"][node_type])))
                            ) for node_type in X["global_node_index"]}

        # Get edge_index with batch id
        for relation_id in reltype_ids:
            if relation_id == 1:
                mask = edge_reltype == relation_id
                X["edge_index_dict"][self.metapaths[0]] = edge_index[mask, :].apply_(
                    local2batch[self.head_node_type].get).t()
            elif relation_id == 0:
                mask = edge_reltype == relation_id
                X["edge_index_dict"][tag_negative(self.metapaths[0])] = edge_index[mask, :].apply_(
                    local2batch[self.head_node_type].get).t()

        if self.use_reverse:
            self.add_reverse_edge_index(X["edge_index_dict"])

        # Make x_dict
        if hasattr(self, "x_dict") and len(self.x_dict) > 0:
            X["x_dict"] = {node_type: self.x_dict[node_type][X["global_node_index"][node_type]] \
                           for node_type in self.x_dict}

        return X, None, None


class BidirectionalSampler(EdgeSampler, HeteroNeighborSampler):
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

        # True positive edges
        triples = {k: v[:, e_idx] for k, v in self.triples.items() if not is_negative(k)}

        # Add true neg edges if on valid or test triplet indices
        if e_idx.max() < self.start_idx["train"] and not self.force_neg_sampling:
            triples.update({metapath: edge_index[:, e_idx % edge_index.shape[1]] \
                            for metapath, edge_index in self.triples.items() \
                            if is_negative(metapath)})

        # Set of all nodes from sampled triples
        triplets_node_index = self.get_global_node_index(triples)

        # Get true edges from triples
        edges_pos, edges_neg = self.get_local_edge_index(triples=triples,
                                                         global_node_index=triplets_node_index,
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

        # Build X input dict
        X = {"edge_index_dict": edge_index_dict,
             "edge_pos": edges_pos,
             "global_node_index": global_node_index,
             "x_dict": node_feats}

        if not edges_neg:
            X.update({"head-batch": head_batch, "tail-batch": tail_batch, })
        else:
            X.update({"edge_neg": edges_neg})

        return X, None, None

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
