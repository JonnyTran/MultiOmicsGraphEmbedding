import os
from typing import Type

import pandas as pd
import networkx as nx
import numpy as np
import tensorflow as tf

import torch
from torch.utils import data

from torch_geometric.datasets import AMiner
from torch_geometric.data import InMemoryDataset

from .sampled_generator import SampledDataGenerator
from openomics.database.interaction import Interactions

from stellargraph.datasets import FB15k_237, WN18RR, BlogCatalog3, AIFB, MovieLens, DatasetLoader
from cogdl.datasets.han_data import ACM_HANDataset, DBLP_HANDataset, IMDB_HANDataset, HANDataset


class HeterogeneousNetworkDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, node_types, metapath=None, head_node_type=None, train_ratio=0.3):
        self.dataset = dataset
        self.metapath = metapath
        self.train_ratio = train_ratio

        if head_node_type is None:
            self.head_node_type = node_types[0]
        else:
            self.head_node_type = head_node_type

        # PyTorchGeometric Dataset
        if isinstance(dataset, InMemoryDataset):
            self.process_inmemorydataset(dataset, train_ratio)

        # StellarGraph Dataset
        elif isinstance(dataset, DatasetLoader):
            self.process_stellargraph(dataset, metapath, node_types, train_ratio)

        # HANDataset Dataset
        elif isinstance(dataset, HANDataset):
            self.process_handataset(dataset, metapath, node_types)
        else:
            raise Exception(f"Unsupported dataset {dataset}")

        self.name = dataset.__class__.__name__

    def process_handataset(self, dataset: HANDataset, metapath, node_types):
        data = dataset.data
        self.edge_index_dict = {metapath: data["adj"][i][0] for i, metapath in enumerate(metapath)}
        self.node_types = node_types
        self.edge_types = list(range(dataset.num_edge))
        self.x = {self.head_node_type: data["x"]}

        self.training_idx = data["train_node"]
        self.training_target = data["train_target"]
        self.validation_idx = data["valid_node"]
        self.validation_target = data["valid_target"]
        self.testing_idx = data["test_node"]
        self.testing_target = data["test_target"]

        self.y_index_dict = {self.head_node_type: torch.cat([self.training_idx, self.validation_idx, self.testing_idx])}
        self.y_dict = {
            self.head_node_type: torch.cat([self.training_target, self.validation_target, self.testing_target])}

        # Sort
        _, indices = torch.sort(self.y_index_dict[self.head_node_type])
        self.y_index_dict[self.head_node_type] = self.y_index_dict[self.head_node_type][indices]
        self.y_dict[self.head_node_type] = self.y_dict[self.head_node_type][indices]

    def process_stellargraph(self, dataset: DatasetLoader, metapath, node_types, train_ratio):
        graph = dataset.load()
        self.node_types = graph.node_types if node_types is None else node_types
        self.edge_types = graph.edge_types
        self.y_index_dict = {k: torch.tensor(graph.nodes(k, use_ilocs=True)) for k in graph.node_types}
        edgelist = graph.edges(include_edge_type=True, use_ilocs=True)
        edge_index_dict = {path: [] for path in metapath}
        for u, v, t in edgelist:
            edge_index_dict[metapath[t]].append([u, v])
        self.edge_index_dict = {metapath: torch.tensor(edges, dtype=torch.long).T for metapath, edges in
                                edge_index_dict.items()}
        perm = torch.randperm(self.y_index_dict[self.head_node_type].size(0))
        self.training_idx = perm[:int(self.y_index_dict[self.head_node_type].size(0) * train_ratio)]
        self.validation_idx = perm[int(self.y_index_dict[self.head_node_type].size(0) * train_ratio):]
        self.testing_idx = perm[int(self.y_index_dict[self.head_node_type].size(0) * train_ratio):]

    def process_inmemorydataset(self, dataset: InMemoryDataset, train_ratio):
        data = dataset[0]
        self.edge_index_dict = data.edge_index_dict
        self.num_nodes_dict = data.num_nodes_dict
        self.node_types = list(data.y_index_dict.keys())
        self.y_dict = data.y_dict
        self.y_index_dict = data.y_index_dict
        # {k: v.unsqueeze(1) for k, v in data.y_index_dict.items()}
        self.metapath = list(self.edge_index_dict.keys())
        perm = torch.randperm(self.y_index_dict[self.head_node_type].size(0))
        self.training_idx = perm[:int(self.y_index_dict[self.head_node_type].size(0) * train_ratio)]
        self.validation_idx = perm[int(self.y_index_dict[self.head_node_type].size(0) * train_ratio):]
        self.testing_idx = perm[int(self.y_index_dict[self.head_node_type].size(0) * train_ratio):]

    def train_dataloader(self, batch_size=128, collate_fn=None):
        loader = data.DataLoader(self.training_idx, batch_size=batch_size,
                                 shuffle=True, num_workers=12,
                                 collate_fn=collate_fn if collate_fn is not None else self.get_collate_fn(collate_fn))
        return loader

    def val_dataloader(self, batch_size=128, collate_fn=None):
        loader = data.DataLoader(self.validation_idx, batch_size=batch_size,
                                 shuffle=False, num_workers=4,
                                 collate_fn=collate_fn if collate_fn is not None else self.get_collate_fn(collate_fn))
        return loader

    def test_dataloader(self, batch_size=128, collate_fn=None):
        loader = data.DataLoader(self.testing_idx, batch_size=batch_size,
                                 shuffle=False, num_workers=4,
                                 collate_fn=collate_fn if collate_fn is not None else self.get_collate_fn(collate_fn))
        return loader

    def get_collate_fn(self, collate_fn: str):
        if collate_fn == "index_cls":
            return self.collate_index_cls
        elif collate_fn == "node_attr_cls":
            return self.collate_node_attr_cls
        else:
            raise Exception(f"Collate function {collate_fn} not found.")

    def collate_node_attr_cls(self, iloc):
        if not isinstance(iloc, torch.Tensor):
            iloc = torch.tensor(iloc)

        X = {}
        X[self.head_node_type] = self.x[self.head_node_type][iloc]

        return X, self.y_dict[self.head_node_type][iloc]

    def collate_index_cls(self, iloc):
        if not isinstance(iloc, torch.Tensor):
            iloc = torch.tensor(iloc)

        X = {}
        X[self.head_node_type] = self.y_index_dict[self.head_node_type][iloc]

        return X, self.y_dict[self.head_node_type][iloc]

class GeneratorDataset(torch.utils.data.Dataset):
    def __init__(self, generator: SampledDataGenerator):
        self._generator = generator
        self.node_list = self._generator.get_connected_nodelist()
        self.n_steps = self._generator.n_steps

    def __len__(self):
        if self.n_steps is not None:
            return self.n_steps
        else:
            return len(self.node_list)

    def __getitem__(self, item=None):
        # seed_node = self.node_list[item]
        sampled_nodes = self._generator.traverse_network(batch_size=self._generator.batch_size, seed_node=None)
        X, y, sample_weights = self._generator.__getdata__(sampled_nodes, variable_length=False)
        X = {k: np.expand_dims(v, 0) for k, v in X.items()}
        y = np.expand_dims(y, 0)
        sample_weights = np.expand_dims(sample_weights, 0)
        return X, y, sample_weights


class TFDataset(tf.data.Dataset):
    def __new__(cls, generator: SampledDataGenerator, output_types=None, output_shapes=None):
        """
        A tf.data wrapper for keras.utils.Sequence generator
        >>> generator = DataGenerator()
        >>> dataset = GeneratorDataset(generator)
        >>> strategy = tf.distribute.MirroredStrategy()
        >>> train_dist_dataset = strategy.experimental_distribute_dataset(dataset)

        :param generator: a keras.utils.Sequence generator.
        """

        def generate():
            while True:
                batch_xs, batch_ys, dset_index = generator.__getitem__(0)
                yield batch_xs, batch_ys, dset_index

        queue = tf.keras.utils.GeneratorEnqueuer(generate, use_multiprocessing=True)

        return tf.data.Dataset.from_generator(
            queue.sequence,
            output_types=generator.get_output_types() if output_types is None else output_types,
            output_shapes=generator.get_output_shapes() if output_shapes is None else output_shapes,
        )