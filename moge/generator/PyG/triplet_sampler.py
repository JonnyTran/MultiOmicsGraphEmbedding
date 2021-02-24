import numpy as np
import pandas as pd
import torch
from ogb.linkproppred import PygLinkPropPredDataset

from moge.generator.PyG.node_sampler import HeteroNeighborSampler
from moge.generator.network import HeteroNetDataset
from moge.generator.utils import merge_node_index
from moge.module.PyG.latte import is_negative


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
        triplets_node_index = self.get_nodes(triples, relation_ids_all, metapaths=self.metapaths)

        # Get true edges from triples
        edges_pos, edges_neg = self.get_relabled_edge_index(triples=triples,
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
        batch_size, n_id, adjs = self.graph_sampler.sample(triplets_node_index)
        sampled_local_nodes = self.graph_sampler.get_nodes_dict(adjs, n_id)

        # Merge triplets_node_index + sampled_local_nodes = global_node_index, while ensuring index order in triplets_node_index
        global_node_index = merge_node_index(old_node_index=triplets_node_index,
                                             new_node_index=sampled_local_nodes)

        # Get dict to convert from global node index to batch node index
        edge_index_dict = self.graph_sampler.get_edge_index_dict(adjs=adjs, n_id=n_id,
                                                                 sampled_local_nodes=global_node_index,
                                                                 filter_nodes=2)

        if self.use_reverse:
            self.add_reverse_edge_index(edge_index_dict)

        # Make x_dict
        if hasattr(self, "x_dict") and len(self.x_dict) > 0:
            node_feats = {node_type: self.x_dict[node_type][global_node_index[node_type]] for node_type in self.x_dict}
        else:
            node_feats = {}

        # Calculate subsampling weights on each edge_pos
        edge_weights = {}
        if hasattr(self, "degree_counts") and "train" in mode:
            for metapath, edge_index in edges_pos.items():
                head_type, tail_type = metapath[0], metapath[-1]
                relation_id = self.metapaths.index(metapath)

                head_weights = self.get_degrees(global_node_index[head_type][edge_index[0]],
                                                relation_id=relation_id,
                                                node_type=head_type)
                tail_weights = self.get_degrees(global_node_index[tail_type][edge_index[1]],
                                                relation_id=-relation_id - 1,
                                                node_type=tail_type)

                subsampling_weight = head_weights + tail_weights
                edge_weights[metapath] = torch.sqrt(1.0 / subsampling_weight)

        # Build X input dict
        X = {"edge_index_dict": edge_index_dict,
             "global_node_index": global_node_index,
             "x_dict": node_feats}

        # Build edges_true dict
        y = {"edge_pos": edges_pos, }
        if not edges_neg:
            y.update({"head-batch": head_batch, "tail-batch": tail_batch, })
        else:
            y.update({"edge_neg": edges_neg})

        return X, y, edge_weights

    def get_degrees(self, node_ids: torch.LongTensor, relation_id, node_type):
        return node_ids.apply_(lambda nid: self.degree_counts.get((nid, relation_id, node_type), 1)).type(torch.float)
