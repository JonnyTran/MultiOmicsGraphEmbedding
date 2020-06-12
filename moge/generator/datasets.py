import os
import pandas as pd
import networkx as nx
import numpy as np
import tensorflow as tf
import torch
from torch.utils import data

from .sampled_generator import SampledDataGenerator
from openomics.database.interaction import Interactions


class AminerDataset(Interactions):
    def __init__(self, path, file_resources=None, source_col_name="source", target_col_name="target", source_index=None,
                 target_index=None, edge_attr=["type"], filters=None, directed=False, relabel_nodes=None,
                 verbose=False):
        if file_resources is None:
            file_resources = {}
            file_resources["id_author.txt"] = os.path.join(path, "id_author.txt")
            file_resources["id_conf.txt"] = os.path.join(path, "id_conf.txt")
            file_resources["paper.txt"] = os.path.join(path, "paper.txt")
            file_resources["paper_author.txt"] = os.path.join(path, "paper_author.txt")
            file_resources["paper_conf.txt"] = os.path.join(path, "paper_conf.txt")
        super(AminerDataset, self).__init__(path, file_resources, source_col_name, target_col_name, source_index,
                                            target_index, edge_attr, filters, directed, relabel_nodes, verbose)

    def load_network(self, file_resources, source_col_name, target_col_name, edge_attr, directed, filters):
        # Nodes
        author = pd.read_table(file_resources["id_author.txt"], names=["id", "name"])
        author["id"] = "a" + author["id"].astype(str)
        author["type"] = "author"
        conf = pd.read_table(file_resources["id_conf.txt"], names=["id", "name"])
        conf["id"] = "c" + conf["id"].astype(str)
        conf["type"] = "conf"
        paper = pd.read_table(file_resources["paper.txt"], names=["id", "name"])
        paper["id"] = "p" + paper["id"].astype(str)
        paper["type"] = "paper"
        nodelist = pd.concat([author, conf, paper])

        # Edgelist
        paper_author = pd.read_table(file_resources["paper_author.txt"], names=["source", "target"])
        paper_author["source"] = "p" + paper_author["source"].astype(str)
        paper_author["target"] = "a" + paper_author["target"].astype(str)
        paper_author["type"] = "writes"

        paper_conf = pd.read_table(file_resources["paper_conf.txt"], names=["source", "target"])
        paper_conf["source"] = "p" + paper_conf["source"].astype(str)
        paper_conf["target"] = "c" + paper_conf["target"].astype(str)
        paper_conf["type"] = "publishes"
        edgelist = pd.concat([paper_author, paper_conf])

        network = nx.from_pandas_edgelist(edgelist, source=source_col_name, target=target_col_name,
                                          edge_attr=edge_attr,
                                          create_using=nx.DiGraph() if directed else nx.Graph())
        print("nodes", network.number_of_nodes(), nodelist.shape)
        nx.set_node_attributes(network, nodelist.set_index("id").to_dict("index"))
        return network


class TorchDataset(torch.utils.data.Dataset):
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