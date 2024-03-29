import numpy as np
import pandas as pd
import torch
from ogb.linkproppred import PygLinkPropPredDataset


class EdgeDataset(HeteroGraphDataset):
    DEFAULT_NODE_TYPE = "self"
    DEFAULT_METAPATH = (DEFAULT_NODE_TYPE, "edge", DEFAULT_NODE_TYPE)

    def __init__(self, *args, **kwargs):
        super(EdgeDataset, self).__init__(*args, **kwargs)

    def process_PygLinkDataset_homo(self, dataset: PygLinkPropPredDataset):
        data = dataset[0]
        self._name = dataset.name
        self.edge_index_dict = {EdgeDataset.DEFAULT_METAPATH: data.edge_index}
        self.head_node_type = EdgeDataset.DEFAULT_NODE_TYPE

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
                # True edges
                if valid_triples[key].dim() == 2 and min(valid_triples[key].shape) == 2:
                    if valid_triples[key].shape[1] == 2:
                        self.triples[EdgeDataset.DEFAULT_METAPATH] = torch.cat(
                            [valid_triples[key], test_triples[key], train_triples[key]],
                            dim=0).permute(1, 0)
                    else:
                        self.triples[EdgeDataset.DEFAULT_METAPATH] = torch.cat(
                            [valid_triples[key], test_triples[key], train_triples[key]],
                            dim=0)

                # Edge attributes
                else:
                    self.triples[key] = torch.cat(
                        [valid_triples[key], test_triples[key], train_triples[key]],
                        dim=0)
            else:
                # Edge attribuets
                self.triples[EdgeDataset.DEFAULT_METAPATH] = np.array(
                    valid_triples[key] + test_triples[key] + train_triples[key])

        # Concat neg edges
        for key in valid_triples.keys():
            if is_negative(key):  # edge_neg
                self.triples[("edge_neg")] = torch.cat([valid_triples[key], test_triples[key]],
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
    def gather_node_set(triples):
        # Gather all unique nodes from sampled triples
        global_node_index = {}

        for metapath in triples:
            global_node_index.setdefault(EdgeDataset.DEFAULT_NODE_TYPE, []).append(triples[metapath][0])
            global_node_index.setdefault(EdgeDataset.DEFAULT_NODE_TYPE, []).append(triples[metapath][1])

        # Find union of nodes from all relations
        global_node_index = {node_type: torch.cat(node_sets, dim=0).unique() \
                             for node_type, node_sets in global_node_index.items()}
        return global_node_index

    @staticmethod
    def get_relabled_edge_index(triples, global_node_index):
        edges_pos = {}
        edges_neg = {}

        local2batch = {
            node_type: dict(zip(
                global_node_index[node_type].numpy(),
                range(len(global_node_index[node_type])))
            ) for node_type in global_node_index}

        # Get edge_index with batch id
        for metapath in triples:
            if not is_negative(metapath):  # ("self", "edge", "self")
                sources = triples[metapath][0].apply_(local2batch[EdgeDataset.DEFAULT_NODE_TYPE].get)
                targets = triples[metapath][1].apply_(local2batch[EdgeDataset.DEFAULT_NODE_TYPE].get)
                edges_pos[EdgeDataset.DEFAULT_METAPATH] = torch.stack([sources, targets], dim=0)

            elif is_negative(metapath):  # "edge_neg"
                sources = triples[metapath][0].apply_(local2batch[EdgeDataset.DEFAULT_NODE_TYPE].get)
                targets = triples[metapath][1].apply_(local2batch[EdgeDataset.DEFAULT_NODE_TYPE].get)
                edges_neg[EdgeDataset.DEFAULT_METAPATH] = torch.stack([sources, targets], dim=0)

            else:
                raise Exception(f"something wrong with metapath {metapath}")

        return edges_pos, edges_neg

    def get_collate_fn(self, collate_fn: str, batch_size=None, mode=None):
        return self.sample


class BidirectionalGenerator(EdgeDataset, HeteroNeighborGenerator):
    def __init__(self, dataset: PygLinkPropPredDataset, neighbor_sizes,
                 negative_sampling_size=128, test_negative_sampling_size=500,
                 force_negative_sampling=False,
                 node_types=None, metapaths=None, head_node_type=None, edge_dir=True,
                 resample_train=None, add_reverse_metapaths=True, **kwargs):
        super(BidirectionalGenerator, self).__init__(dataset, neighbor_sizes=neighbor_sizes, node_types=node_types,
                                                     metapaths=metapaths,
                                                     head_node_type=head_node_type, edge_dir=edge_dir,
                                                     resample_train=resample_train,
                                                     add_reverse_metapaths=add_reverse_metapaths, **kwargs)
        self.neg_sampling_size = negative_sampling_size
        self.test_neg_sampling_size = test_negative_sampling_size
        self.force_neg_sampling = force_negative_sampling

        df = pd.DataFrame(self.triples[EdgeDataset.DEFAULT_METAPATH].numpy().T)
        self.degree_counts = df.groupby(0)[1].count().add(df.groupby(1)[0].count(),
                                                          fill_value=0)  # <node_id: in_degree + out_degree>

    def get_collate_fn(self, collate_fn: str, mode=None):
        assert mode is not None, "Must pass arg `mode` at get_collate_fn(). {'train', 'valid', 'test'}"

        def collate_wrapper(iloc):
            return self.sample(iloc, mode=mode)

        return collate_wrapper

    def sample(self, e_idx, mode):
        if not isinstance(e_idx, torch.Tensor):
            e_idx = torch.tensor(e_idx)

        # Select negative sampling size
        if "test" in mode:
            negative_sampling_size = self.test_neg_sampling_size
        elif "valid" in mode:
            negative_sampling_size = self.test_neg_sampling_size
        else:
            negative_sampling_size = self.neg_sampling_size
        negative_sampling_size = int(negative_sampling_size)

        # True positive edges
        triples = {k: v[:, e_idx] for k, v in self.triples.items() if not is_negative(k) and "edge" in k}

        # Add true neg edges if on valid or test triplet indices
        if e_idx.max() < self.start_idx["train"] and not self.force_neg_sampling:
            triples.update({key: edge_index[:, e_idx % edge_index.shape[1]] \
                            for key, edge_index in self.triples.items() \
                            if is_negative(key)})

        # Set of all nodes from sampled triples
        triplets_node_index = EdgeDataset.gather_node_set(triples)

        # Get true edges from triples
        edges_pos, edges_neg = EdgeDataset.get_relabled_edge_index(triples=triples,
                                                                   global_node_index=triplets_node_index)

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

        sampled_local_nodes = self.graph_sampler.get_local_nodes(n_id)

        # Merge triplets_node_index + sampled_local_nodes = global_node_index, while ensuring index order in triplets_node_index
        global_node_index = merge_node_index(old_node_index=triplets_node_index,
                                             new_node_index=sampled_local_nodes)

        # Get dict to convert from global node index to batch node index
        edge_index_dict = self.graph_sampler.get_edge_index_dict(adjs=adjs, n_id=n_id,
                                                                 sampled_local_nodes=global_node_index,
                                                                 filter_nodes=False)

        if self.use_reverse:
            self.add_reverse_edge_index(edge_index_dict)

        # Make x_dict
        if hasattr(self, "x_dict") and len(self.x_dict) > 0:
            node_feats = {node_type: self.x_dict[node_type][global_node_index[node_type]] \
                          for node_type in self.x_dict}
        else:
            node_feats = {}

        edge_weights = {}
        if hasattr(self, "degree_counts"):
            for metapath, edge_index in edges_pos.items():
                head_type, tail_type = metapath[0], metapath[-1]

                head_weights = self.get_degrees(global_node_index[head_type][edge_index[0]])
                tail_weights = self.get_degrees(global_node_index[tail_type][edge_index[1]])

                subsampling_weight = head_weights + tail_weights
                edge_weights[metapath] = torch.sqrt(1.0 / subsampling_weight)

        # Build X input dict
        X = {"edge_index_dict": edge_index_dict,
             "global_node_index": global_node_index,
             "x_dict": node_feats}

        # Build edge_true dict
        y = {"edge_pos": edges_pos, }
        if not edges_neg:
            y.update({"head-batch": head_batch, "tail-batch": tail_batch, })
        else:
            y.update({"edge_neg": edges_neg})

        return X, y, edge_weights

    def get_degrees(self, node_ids: torch.LongTensor):
        return node_ids.apply_(lambda nid: self.degree_counts.get((nid), 1)).type(torch.float)
