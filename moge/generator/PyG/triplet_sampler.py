import logging

import numpy as np
import torch
from ogb.linkproppred import PygLinkPropPredDataset

from moge.generator.PyG.node_sampler import HeteroNeighborSampler
from moge.generator.network import HeteroNetDataset
from moge.module.PyG.latte import is_negative
from moge.module.utils import tensor_sizes


class TripletSampler(HeteroNetDataset):
    def __init__(self, dataset: PygLinkPropPredDataset, node_types=None, metapaths=None, head_node_type=None,
                 directed=True,
                 resample_train=None, add_reverse_metapaths=True, **kwargs):
        super(TripletSampler, self).__init__(dataset, node_types=node_types, metapaths=metapaths,
                                             head_node_type=head_node_type,
                                             directed=directed, resample_train=resample_train,
                                             add_reverse_metapaths=add_reverse_metapaths, **kwargs)
        self.n_classes = None
        self.classes = None
        assert hasattr(self, "validation_idx") and hasattr(self, "triples")

    def process_PygLinkDataset_hetero(self, dataset: PygLinkPropPredDataset):
        data = dataset[0]
        self._name = dataset.name
        self.edge_index_dict = data.edge_index_dict

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
        for key in train_triples.keys():
            if isinstance(train_triples[key], torch.Tensor):
                self.triples[key] = torch.cat([valid_triples[key], test_triples[key], train_triples[key]], dim=0)
            else:
                self.triples[key] = np.array(valid_triples[key] + test_triples[key] + train_triples[key])

        for key in valid_triples.keys():
            if is_negative(key):  # either head_neg or tail_neg
                self.triples[key] = torch.cat([valid_triples[key], test_triples[key]], dim=0)

        self.start_idx = {"valid": 0,
                          "test": len(valid_triples["relation"]),
                          "train": len(valid_triples["relation"]) + len(test_triples["relation"])}

        self.validation_idx = torch.arange(self.start_idx["valid"],
                                           self.start_idx["valid"] + len(valid_triples["relation"]))
        self.testing_idx = torch.arange(self.start_idx["test"], self.start_idx["test"] + len(test_triples["relation"]))
        self.training_idx = torch.arange(self.start_idx["train"],
                                         self.start_idx["train"] + len(train_triples["relation"]))

        assert self.validation_idx.max() < self.testing_idx.min()
        assert self.testing_idx.max() < self.training_idx.min()

    def process_edge_reltype_dataset(self, dataset: PygLinkPropPredDataset):
        data = dataset[0]
        self._name = dataset.name
        self.edge_reltype = data.edge_reltype

        if hasattr(data, "num_nodes_dict"):
            self.num_nodes_dict = data.num_nodes_dict
        elif not hasattr(data, "edge_index_dict"):
            self.head_node_type = "entity"
            self.num_nodes_dict = {self.head_node_type: data.edge_index.max().item() + 1}

        if self.node_types is None:
            self.node_types = list(self.num_nodes_dict.keys())

        if hasattr(data, "x") and data.x is not None:
            self.x_dict = {self.head_node_type: data.x}
        elif hasattr(data, "x_dict") and data.x_dict is not None:
            self.x_dict = data.x_dict
        else:
            self.x_dict = {}

        self.metapaths = [(self.head_node_type, str(k.item()), self.head_node_type) for k in self.edge_reltype.unique()]
        self.edge_index_dict = {k: None for k in self.metapaths}

        split_idx = dataset.get_edge_split()
        train_triples, valid_triples, test_triples = split_idx["train"], split_idx["valid"], split_idx["test"]
        self.triples = {}
        for key in train_triples.keys():
            if isinstance(train_triples[key], torch.Tensor):
                self.triples[key] = torch.cat([valid_triples[key], test_triples[key], train_triples[key]], dim=0)
            else:
                self.triples[key] = np.array(valid_triples[key] + test_triples[key] + train_triples[key])

        for key in valid_triples.keys():
            if is_negative(key):  # either head_neg or tail_neg
                self.triples[key] = torch.cat([valid_triples[key], test_triples[key]], dim=0)

        # Edge index starts with valid_idx, test_idx, then train_idx
        self.start_idx = {"valid": 0,
                          "test": len(valid_triples["relation"]),
                          "train": len(valid_triples["relation"]) + len(test_triples["relation"])}

        self.validation_idx = torch.arange(self.start_idx["valid"],
                                           self.start_idx["valid"] + len(valid_triples["relation"]))
        self.testing_idx = torch.arange(self.start_idx["test"],
                                        self.start_idx["test"] + len(test_triples["relation"]))
        self.training_idx = torch.arange(self.start_idx["train"],
                                         self.start_idx["train"] + len(train_triples["relation"]))

        assert self.validation_idx.max() < self.testing_idx.min()
        assert self.testing_idx.max() < self.training_idx.min()

    def get_collate_fn(self, collate_fn: str, mode=None):
        assert mode is not None, "Must pass arg `mode` at get_collate_fn(). {'train', 'valid', 'test'}"

        def collate_wrapper(iloc):
            return self.sample(iloc, mode=mode)

        if "triples_batch" in collate_fn:
            return collate_wrapper
        else:
            raise Exception(f"Correct collate function {collate_fn} not found.")

    def sample(self, e_idx, mode=None):
        if not isinstance(e_idx, torch.Tensor):
            e_idx = torch.tensor(e_idx)

        # Add neg edges if valid or test
        if e_idx.max() < self.start_idx["train"]:
            has_neg_edges = True
        else:
            has_neg_edges = False

        triples = {k: v[e_idx] for k, v in self.triples.items() if not is_negative(k)}
        if has_neg_edges:
            triples.update({k: v[e_idx] for k, v in self.triples.items() if is_negative(k)})

        relation_ids_all = triples["relation"].unique()

        global_node_index = self.get_nodes(triples, relation_ids_all, metapaths=self.metapaths)

        edge_index_dict = self.get_relabled_edge_index(triples=triples,
                                                       global_node_index=global_node_index,
                                                       relation_ids_all=relation_ids_all,
                                                       metapaths=self.metapaths)
        if self.use_reverse:
            self.add_reverse_edge_index(edge_index_dict)

        # Make x_dict
        if hasattr(self, "x_dict") and len(self.x_dict) > 0:
            node_feats = {node_type: self.x_dict[node_type][global_node_index[node_type]] \
                          for node_type in self.x_dict}
        else:
            node_feats = None

        X = {"edge_index_dict": edge_index_dict,
             "edge_pred_dict": edge_index_dict,
             "global_node_index": global_node_index,
             "x_dict": node_feats}

        return X, None, None

    @staticmethod
    def get_relabled_edge_index(triples, global_node_index, relation_ids_all, metapaths, local2batch=None):
        edges_pos = {}
        edges_neg = {}

        if local2batch is None:
            local2batch = {
                node_type: dict(zip(
                    global_node_index[node_type].numpy(),
                    range(len(global_node_index[node_type])))
                ) for node_type in global_node_index}

        # Get edge_index with batch id
        for relation_id in relation_ids_all:
            metapath = metapaths[relation_id]
            head_type, tail_type = metapath[0], metapath[-1]

            mask = triples["relation"] == relation_id
            sources = triples["head"][mask].apply_(local2batch[head_type].get)
            targets = triples["tail"][mask].apply_(local2batch[tail_type].get)
            edges_pos[metapath] = torch.stack([sources, targets], dim=1).t()

            if any(["neg" in k for k in triples.keys()]):
                head_neg = triples["head_neg"][mask].apply_(local2batch[head_type].get)
                tail_neg = triples["tail_neg"][mask].apply_(local2batch[tail_type].get)
                head_batch = torch.stack([head_neg.view(-1),
                                          targets.repeat(head_neg.size(1))])
                tail_batch = torch.stack([sources.repeat(tail_neg.size(1)),
                                          tail_neg.view(-1)])
                edges_neg[metapath] = torch.cat([head_batch, tail_batch], dim=1)

        return edges_pos, edges_neg

    @staticmethod
    def get_nodes(triples, relation_ids, metapaths):
        # Gather all nodes sampled
        node_index_dict = {}

        for relation_id in relation_ids:
            metapath = metapaths[relation_id]
            head_type, tail_type = metapath[0], metapath[-1]

            mask = triples["relation"] == relation_id
            node_index_dict.setdefault(head_type, []).append(triples["head"][mask])
            node_index_dict.setdefault(tail_type, []).append(triples["tail"][mask])

            if any(["neg" in k for k in triples.keys()]):
                node_index_dict.setdefault(head_type, []).append(triples["head_neg"][mask].view(-1))
                node_index_dict.setdefault(tail_type, []).append(triples["tail_neg"][mask].view(-1))

        # Find union of nodes from all relations
        node_index_dict = {node_type: torch.cat(node_sets, dim=0).unique() \
                           for node_type, node_sets in node_index_dict.items()}
        return node_index_dict


class TripletNeighborSampler(HeteroNeighborSampler):
    def __init__(self, dataset, neighbor_sizes, node_types=None, metapaths=None, head_node_type=None, directed=True,
                 resample_train=None, add_reverse_metapaths=True, inductive=False):
        super(TripletNeighborSampler, self).__init__(dataset, neighbor_sizes, node_types, metapaths, head_node_type,
                                                     directed, resample_train,
                                                     add_reverse_metapaths, inductive)

    def process_PygLinkDataset_hetero(self, dataset: PygLinkPropPredDataset):
        data = dataset[0]
        self._name = dataset.name
        self.edge_index_dict = data.edge_index_dict

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
        for key in train_triples.keys():
            if isinstance(train_triples[key], torch.Tensor):
                self.triples[key] = torch.cat([valid_triples[key], test_triples[key], train_triples[key]], dim=0)
            else:
                self.triples[key] = np.array(valid_triples[key] + test_triples[key] + train_triples[key])

        for key in valid_triples.keys():
            if is_negative(key):  # either head_neg or tail_neg
                self.triples[key] = torch.cat([valid_triples[key], test_triples[key]], dim=0)

        self.start_idx = {"valid": 0,
                          "test": len(valid_triples["relation"]),
                          "train": len(valid_triples["relation"]) + len(test_triples["relation"])}

        self.validation_idx = torch.arange(self.start_idx["valid"],
                                           self.start_idx["valid"] + len(valid_triples["relation"]))
        self.testing_idx = torch.arange(self.start_idx["test"], self.start_idx["test"] + len(test_triples["relation"]))
        self.training_idx = torch.arange(self.start_idx["train"],
                                         self.start_idx["train"] + len(train_triples["relation"]))

        assert self.validation_idx.max() < self.testing_idx.min()
        assert self.testing_idx.max() < self.training_idx.min()

    def get_collate_fn(self, collate_fn: str, mode=None):
        assert mode is not None, "Must pass arg `mode` at get_collate_fn(). {'train', 'valid', 'test'}"

        def collate_wrapper(iloc):
            return self.sample(iloc, mode=mode)

        return collate_wrapper

    def sample(self, e_idx, mode):
        if not isinstance(e_idx, torch.Tensor):
            e_idx = torch.tensor(e_idx)

        triples = {k: v[e_idx] for k, v in self.triples.items() if not is_negative(k)}
        if "train" not in mode:
            triples.update({k: v[e_idx] for k, v in self.triples.items() if is_negative(k)})

        relation_ids_all = triples["relation"].unique()

        # Get all nodes from sampled triplets
        batch_nodes = TripletSampler.get_nodes(triples, relation_ids_all, metapaths=self.metapaths)
        batch_nodes_local = torch.cat([self.local2global[ntype][nid] for ntype, nid in batch_nodes.items()], 0)

        # Get full subgraph from n_id's
        batch_size, n_id, adjs = self.graph_sampler.sample(batch_nodes_local)
        if not isinstance(adjs, list):
            adjs = [adjs]

        # Sample neighbors and return `sampled_local_nodes` as the set of all nodes traversed (in local index)
        sampled_local_nodes = self.get_local_node_index(adjs, n_id)
        X = {"edge_index_dict": {},
             "global_node_index": sampled_local_nodes,
             "x_dict": {}}

        X["edge_index_dict"] = self.get_local_edge_index_dict(adjs=adjs, n_id=n_id,
                                                              sampled_local_nodes=sampled_local_nodes,
                                                              filter_nodes=False)
        logging.info(tensor_sizes(sampled_local_nodes))

        # x_dict attributes
        if hasattr(self, "x_dict") and len(self.x_dict) > 0:
            X["x_dict"] = {node_type: self.x_dict[node_type][X["global_node_index"][node_type]] \
                           for node_type in self.x_dict}

        local2batch = {
            node_type: dict(zip(sampled_local_nodes[node_type].numpy(),
                                range(len(sampled_local_nodes[node_type])))
                            ) for node_type in sampled_local_nodes}

        X["edge_pred_dict"], edges_neg = TripletSampler.get_relabled_edge_index(triples=triples,
                                                                                global_node_index=sampled_local_nodes,
                                                                                relation_ids_all=relation_ids_all,
                                                                                metapaths=self.metapaths,
                                                                                local2batch=local2batch)

        return X, None, None
