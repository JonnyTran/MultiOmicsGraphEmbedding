import numpy as np
import torch
from ogb.linkproppred import PygLinkPropPredDataset

from ..datasets import HeteroNetDataset


class LinkSampler(HeteroNetDataset):
    def __init__(self, dataset, node_types, metapaths=None, head_node_type=None, directed=True, train_ratio=0.7,
                 add_reverse_metapaths=True, process_graphs=False):
        super(LinkSampler, self).__init__(dataset, node_types, metapaths, head_node_type, directed, train_ratio,
                                          add_reverse_metapaths,
                                          process_graphs)

    def process_PygLinkDataset(self, dataset: PygLinkPropPredDataset, train_ratio):
        data = dataset[0]
        self._name = dataset.name
        self.edge_index_dict = data.edge_index_dict
        self.num_nodes_dict = data.num_nodes_dict
        if self.node_types is None:
            self.node_types = list(data.num_nodes_dict.keys())
        self.node_attr_shape = {}
        self.multilabel = False

        self.metapaths = list(self.edge_index_dict.keys())

        split_idx = dataset.get_edge_split()
        train_triples, valid_triples, test_triples = split_idx["train"], split_idx["valid"], split_idx["test"]
        self.triples = {}
        for key in train_triples.keys():
            if isinstance(train_triples[key], torch.Tensor):
                self.triples[key] = torch.cat([train_triples[key], valid_triples[key], test_triples[key]], dim=0)
            else:
                self.triples[key] = np.array(train_triples[key] + valid_triples[key] + test_triples[key])

        self.training_idx = torch.arange(0, len(train_triples["relation"]))
        self.validation_idx = torch.arange(self.training_idx.size(0),
                                           self.training_idx.size(0) + len(valid_triples["relation"]))
        self.testing_idx = torch.arange(self.training_idx.size(0) + self.validation_idx.size(0),
                                        self.training_idx.size(0) + self.validation_idx.size(0) + len(
                                            test_triples["relation"]))

        self.train_ratio = self.training_idx.numel() / \
                           sum([self.training_idx.numel(), self.validation_idx.numel(), self.testing_idx.numel()])
        node_indices = torch.cat([self.training_idx, self.validation_idx, self.testing_idx])
        # self.training_idx, self.validation_idx, self.testing_idx = \
        #     self.split_train_val_test(train_ratio=train_ratio, sample_indices=node_indices)

    def get_collate_fn(self, collate_fn: str, batch_size=None, mode=None):
        if "triples_batch" in collate_fn:
            return self.collate_triples_batch
        else:
            raise Exception(f"Correct collate function {collate_fn} not found.")

    def collate_triples_batch(self, iloc):
        if not isinstance(iloc, torch.Tensor):
            iloc = torch.tensor(iloc)

        X = {"edge_index_dict": {}, "global_node_index": {}, "x_dict": {}}

        triples = {k: v[iloc] for k, v in self.triples.items()}

        for metapath_id in triples["relation"].unique():
            metapath = self.metapaths[metapath_id]
            head_type, tail_type = metapath[0], metapath[-1]
            X["edge_index_dict"][metapath] = torch.stack([triples["head"], triples["tail"]], dim=1).t()
            X["global_node_index"].setdefault(head_type, []).append(triples["head"])
            X["global_node_index"].setdefault(tail_type, []).append(triples["tail"])

        X["global_node_index"] = {k: torch.cat(v, dim=0).unique() for k, v in X["global_node_index"].items()}

        if self.use_reverse:
            self.add_reverse_edge_index(X["edge_index_dict"])

        return X, None, None
