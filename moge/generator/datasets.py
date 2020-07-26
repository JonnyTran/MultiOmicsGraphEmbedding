from collections import OrderedDict
from itertools import islice
import multiprocessing
from scipy.io import loadmat

import networkx as nx
import numpy as np

import tensorflow as tf
import torch
from cogdl.datasets.gtn_data import GTNDataset
from cogdl.datasets.han_data import HANDataset

from ogb.linkproppred import PygLinkPropPredDataset
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.utils.hetero import group_hetero_graph
from torch_geometric.data import NeighborSampler
import deepsnap

from stellargraph.datasets import DatasetLoader
from torch.utils import data
from torch_geometric.data import InMemoryDataset


class HeterogeneousNetworkDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, node_types, metapaths=None, head_node_type=None, directed=False, train_ratio=0.7,
                 add_reverse_metapaths=True):
        self.dataset = dataset
        self.directed = directed
        self.train_ratio = train_ratio
        self.use_reverse = add_reverse_metapaths

        if head_node_type is None:
            self.head_node_type = node_types[0]
            if len(node_types) > 1: print(f"INFO: Selected {self.head_node_type} from node_types: {node_types}")
        else:
            self.head_node_type = head_node_type

        # PyTorchGeometric Dataset
        if isinstance(dataset, PygNodePropPredDataset):
            print("PygNodePropPredDataset")
            self.process_PygNodeDataset(dataset, train_ratio)

        elif isinstance(dataset, InMemoryDataset):
            print("InMemoryDataset")
            self.process_inmemorydataset(dataset, train_ratio)

        # StellarGraph Dataset
        elif isinstance(dataset, DatasetLoader):
            print("InMemoryDataset")
            self.process_stellargraph(dataset, metapaths, node_types, train_ratio)

        # HANDataset Dataset
        elif isinstance(dataset, HANDataset) or isinstance(dataset, GTNDataset):
            print("InMemoryDataset")
            self.process_HANdataset(dataset, metapaths, node_types, train_ratio)

        elif "blogcatalog6k" in dataset:
            self.process_BlogCatalog6k(dataset, train_ratio)
        else:
            raise Exception(f"Unsupported dataset {dataset}")

        if hasattr(self, "y_dict"):
            if self.y_dict[self.head_node_type].dim() > 1:
                self.classes = self.y_dict[self.head_node_type].unique()
                self.n_classes = self.y_dict[self.head_node_type].size(1)
                self.multilabel = True
            else:
                self.classes = self.y_dict[self.head_node_type].unique()
                self.n_classes = self.classes.size(0)
                self.multilabel = False
        else:
            print("WARNING: Dataset doesn't have node label (y_dict attribute).")

        try:
            cpus = multiprocessing.cpu_count()
        except NotImplementedError:
            cpus = 2  # arbitrary default

        self.graphs = {}
        pool = multiprocessing.Pool(processes=cpus)
        output = pool.map(self.create_graph, self.original_metapaths if self.use_reverse else self.metapaths)
        for (metapath, graph) in output:
            self.graphs[metapath] = graph
        pool.close()

        assert hasattr(self, "num_nodes_dict")
        assert hasattr(self, "head_node_type")
        assert hasattr(self, "y_index_dict")
        assert hasattr(self, "y_dict")
        self.name = dataset.__class__.__name__

    def create_graph(self, metapath):
        edgelist = self.edge_index_dict[metapath].t().numpy().astype(str)
        edgelist = np.core.defchararray.add([metapath[0][0], metapath[-1][0]], edgelist)
        graph = nx.from_edgelist(edgelist,
                                 create_using=nx.DiGraph if not self.use_reverse and self.directed else nx.Graph)
        return (metapath, graph)

    @staticmethod
    def add_reverse_edge_index(edge_index_dict) -> None:
        reverse_edge_index_dict = {}
        for metapath in edge_index_dict:
            reverse_metapath = tuple(a + "_by" if i == 1 else a for i, a in enumerate(reversed(metapath)))
            reverse_edge_index_dict[reverse_metapath] = edge_index_dict[metapath][[1, 0], :]
        edge_index_dict.update(reverse_edge_index_dict)

    @staticmethod
    def get_reverse_metapath(metapaths) -> None:
        reverse_metapaths = []
        for metapath in metapaths:
            reverse = tuple(a + "_by" if i == 1 else a for i, a in enumerate(reversed(metapath)))
            reverse_metapaths.append(reverse)
        return reverse_metapaths

    def process_BlogCatalog6k(self, dataset, train_ratio):
        data = loadmat(dataset)  # From http://dmml.asu.edu/users/xufei/Data/blogcatalog6k.mat
        self.y_index_dict = {"user": torch.arange(data["friendship"].shape[0]),
                             "tag": torch.arange(data["tagnetwork"].shape[0])}
        self.node_types = ["user", "tag"]
        self.head_node_type = "user"
        self.y_dict = {self.head_node_type: torch.tensor(data["usercategory"].toarray().astype(int))}
        self.num_nodes_dict = {"user": data["friendship"].shape[0],
                               "tag": data["tagnetwork"].shape[0]}

        self.metapaths = [("user", "usertag", "tag"),
                          ("tag", "tagnetwork", "tag"),
                          ("user", "friendship", "user"), ]
        self.edge_index_dict = {
            ("user", "friendship", "user"): self.adj_to_edgeindex(data["friendship"]),
            ("user", "usertag", "tag"): self.adj_to_edgeindex(data["usertag"]),
            ("tag", "tagnetwork", "tag"): self.adj_to_edgeindex(data["tagnetwork"])}
        self.training_idx, self.validation_idx, self.testing_idx = self.split_train_val_test(train_ratio)

    def process_HANdataset(self, dataset: HANDataset, metapath, node_types, train_ratio):
        data = dataset.data
        self.edge_index_dict = {metapath: data["adj"][i][0] for i, metapath in enumerate(metapath)}
        self.node_types = node_types
        self.edge_types = list(range(dataset.num_edge))
        self.x_dict = {self.head_node_type: data["x"]}
        self.in_features = data["x"].size(1)

        self.training_idx, self.training_target = data["train_node"], data["train_target"]
        self.validation_idx, self.validation_target = data["valid_node"], data["valid_target"]
        self.testing_idx, self.testing_target = data["test_node"], data["test_target"]

        node_indices = torch.cat([self.training_idx, self.validation_idx, self.testing_idx])
        self.y_index_dict = {self.head_node_type: node_indices}
        self.num_nodes_dict = {self.head_node_type: node_indices.size(0)}
        self.y_dict = {
            self.head_node_type: torch.cat([self.training_target, self.validation_target, self.testing_target])}

        # self.y_index_dict = {self.head_node_type: torch.arange(self.x[self.head_node_type].size(0))}
        # self.num_nodes_dict = {self.head_node_type: self.x[self.head_node_type].size(0)}
        #
        # _, indices = torch.sort(node_indices)
        # self.y_dict = {
        #     self.head_node_type: torch.cat([self.training_target, self.validation_target, self.testing_target])[
        #         indices]}

        self.training_idx, self.validation_idx, self.testing_idx = self.split_train_val_test(train_ratio=train_ratio)
        assert self.y_index_dict[self.head_node_type].size(0) == self.y_dict[self.head_node_type].size(0)
        assert torch.max(self.training_idx) < node_indices.shape[0]
        self.data = data

    def process_stellargraph(self, dataset: DatasetLoader, metapath, node_types, train_ratio):
        graph = dataset.load()
        self.node_types = graph.node_types if node_types is None else node_types
        self.edge_types = graph.metapaths
        self.y_index_dict = {k: torch.tensor(graph.nodes(k, use_ilocs=True)) for k in graph.node_types}

        edgelist = graph.edges(include_edge_type=True, use_ilocs=True)
        edge_index_dict = {path: [] for path in metapath}
        for u, v, t in edgelist:
            edge_index_dict[metapath[t]].append([u, v])
        self.edge_index_dict = {metapath: torch.tensor(edges, dtype=torch.long).T for metapath, edges in
                                edge_index_dict.items()}
        self.training_node, self.validation_node, self.testing_node = self.split_train_val_test(train_ratio)

    def process_inmemorydataset(self, dataset: InMemoryDataset, train_ratio):
        data = dataset[0]
        if self.use_reverse:
            self.add_reverse_edge_index(data.edge_index_dict)
        self.edge_index_dict = data.edge_index_dict
        self.num_nodes_dict = data.num_nodes_dict
        self.node_types = list(data.y_index_dict.keys())
        self.y_dict = data.y_dict
        self.y_index_dict = data.y_index_dict
        self.metapaths = list(self.edge_index_dict.keys())
        self.training_idx, self.validation_idx, self.testing_idx = self.split_train_val_test(train_ratio)

    def process_PygNodeDataset(self, dataset: PygNodePropPredDataset, train_ratio):
        data = dataset[0]
        self.edge_index_dict = data.edge_index_dict
        self.num_nodes_dict = data.num_nodes_dict
        self.node_types = list(data.num_nodes_dict.keys())
        self.x_dict = data.x_dict
        self.node_attr_shape = {node_type: x.size(1) for node_type, x in self.x_dict.items()}
        self.y_dict = data.y_dict
        self.y_index_dict = {node_type: torch.arange(data.num_nodes_dict[node_type]) for node_type in
                             data.y_dict.keys()}
        self.multilabel = False

        self.metapaths = list(self.edge_index_dict.keys())
        if self.use_reverse:
            self.original_metapaths = list(self.edge_index_dict.keys())
            self.metapaths = self.metapaths + self.get_reverse_metapath(self.metapaths)

        split_idx = dataset.get_idx_split()
        self.training_idx, self.validation_idx, self.testing_idx = split_idx["train"][self.head_node_type], \
                                                                   split_idx["valid"][self.head_node_type], \
                                                                   split_idx["test"][self.head_node_type]
        self.train_ratio = self.training_idx.numel() / \
                           sum([self.training_idx.numel(), self.validation_idx.numel(), self.testing_idx.numel()])

    def split_train_val_test(self, train_ratio, sample_indices=None):
        perm = torch.randperm(self.num_nodes_dict[self.head_node_type])
        if sample_indices is not None:
            perm = sample_indices[perm]
        training_idx = perm[:int(self.y_index_dict[self.head_node_type].size(0) * train_ratio)]
        validation_idx = perm[int(self.y_index_dict[self.head_node_type].size(0) * train_ratio):]
        testing_idx = perm[int(self.y_index_dict[self.head_node_type].size(0) * train_ratio):]
        return training_idx, validation_idx, testing_idx

    def adj_to_edgeindex(self, adj):
        adj = adj.tocoo(copy=False)
        return torch.tensor(np.vstack((adj.row, adj.col)).astype("long"))

    def train_dataloader(self, collate_fn=None, batch_size=128, num_workers=12):
        loader = data.DataLoader(self.training_idx, batch_size=batch_size,
                                 shuffle=True, num_workers=num_workers,
                                 collate_fn=collate_fn if callable(collate_fn) else self.get_collate_fn(collate_fn,
                                                                                                        batch_size))
        return loader

    def val_dataloader(self, collate_fn=None, batch_size=128, num_workers=4):
        loader = data.DataLoader(self.validation_idx, batch_size=batch_size,
                                 shuffle=False, num_workers=num_workers,
                                 collate_fn=collate_fn if callable(collate_fn) else self.get_collate_fn(collate_fn,
                                                                                                        batch_size))
        return loader

    def test_dataloader(self, collate_fn=None, batch_size=128, num_workers=4):
        loader = data.DataLoader(self.testing_idx, batch_size=batch_size,
                                 shuffle=False, num_workers=num_workers,
                                 collate_fn=collate_fn if callable(collate_fn) else self.get_collate_fn(collate_fn,
                                                                                                        batch_size))
        return loader

    def get_collate_fn(self, collate_fn: str, batch_size=None):
        if batch_size is not None and "PyGNodeDataset_batch" in collate_fn:
            self.batch_size = batch_size * len(self.node_types)
        if "index" in collate_fn:
            return self.collate_index_cls
        elif "attr" in collate_fn:
            return self.collate_node_attr_cls
        elif "HAN_batch" in collate_fn:
            return self.collate_HAN_batch
        elif "HAN" in collate_fn:
            return self.collate_HAN
        elif "PyGNodeDataset_batch" in collate_fn:
            return self.collate_PyGNodeDataset_batch
        else:
            raise Exception(f"Correct collate function {collate_fn} not found.")

    def convert_index2name(self, node_idx: torch.Tensor, node_type):
        return np.core.defchararray.add(node_type[0], node_idx.numpy().astype(str)).tolist()

    def convert_name2index(self, node_names: list):
        """
        Strip letter from node names
        :param node_names:
        :return:
        """
        return torch.tensor([int(name[1:]) for name in node_names], dtype=torch.long)

    def collate_PyGNodeDataset_batch(self, iloc):
        if not isinstance(iloc, torch.Tensor):
            iloc = torch.tensor(iloc)

        seed_nodes = {self.head_node_type: self.convert_index2name(iloc, self.head_node_type)}
        sampled_nodes = self.bfs_traversal(batch_size=self.batch_size, seed_nodes=seed_nodes)
        print("sampled_nodes", {k: len(v) for k, v, in sampled_nodes.items()})
        node_index = self.y_index_dict[self.head_node_type][self.convert_name2index(sampled_nodes[self.head_node_type])]

        assert len(node_index) == len(seed_nodes[self.head_node_type])

        X = {"edge_index_dict": {},
             "x_dict": self.x_dict[self.head_node_type][node_index] if hasattr(self, "x_dict") else None,
             "x_index_dict": {}}

        for metapath in self.original_metapaths if self.use_reverse else self.metapaths:
            if len(sampled_nodes[metapath[0]]) == 0 or sampled_nodes[metapath[-1]] == 0:
                continue
            try:
                X["edge_index_dict"][metapath] = self.get_adj_edgelist(self.graphs[metapath],
                                                                       nodes_A=sampled_nodes[metapath[0]],
                                                                       nodes_B=sampled_nodes[metapath[-1]])
            except:
                X["edge_index_dict"][metapath] = None
        self.add_reverse_edge_index(X["edge_index_dict"])

        for node_type in self.node_types:
            X["x_index_dict"][node_type] = self.convert_name2index(sampled_nodes[node_type])

        y = self.y_dict[self.head_node_type][iloc]
        return X, y, None

    def bfs_traversal(self, batch_size: int, seed_nodes: {str: list}, max_iter=5):
        num_nodes = sum([len(nodes) for node_type, nodes in seed_nodes.items()])
        i = 0
        while num_nodes < batch_size and i < max_iter:
            for metapath, G in self.graphs.items():
                head_type, tail_type = metapath[0], metapath[-1]
                print("metapath", metapath)
                if head_type in seed_nodes:
                    source_nodes = seed_nodes[head_type]
                    neighbor_type = tail_type
                elif tail_type in seed_nodes:
                    source_nodes = seed_nodes[tail_type]
                    neighbor_type = head_type
                else:
                    continue
                # if not any([node in G for node in source_nodes]): continue

                neighbors = [node for source, successors in
                             islice(nx.traversal.bfs_successors(G, source=set(source_nodes)), 1) for node in successors]
                print("neighbors", len(neighbors))

                # Ensure that no node_type becomes the majority of the batch_size
                if len(neighbors) > (batch_size / (len(self.node_types) - 1)):
                    neighbors = neighbors[: int(batch_size / (len(self.node_types) - 1))]

                if neighbor_type in seed_nodes:
                    seed_nodes[neighbor_type].extend(neighbors)
                else:
                    seed_nodes[neighbor_type] = neighbors

            # Remove duplicate
            for node_type in seed_nodes.keys():
                seed_nodes[node_type] = list(OrderedDict.fromkeys(seed_nodes[node_type]))

            # Check whether to gather more nodes to fill batch_size
            num_nodes = sum([len(nodes) for node_type, nodes in seed_nodes.items()])
            i += 1

        # Remove excess node if exceeds batch_size
        if num_nodes > batch_size:
            largest_node_type = max({k: v for k, v in seed_nodes.items() if k != self.head_node_type}, key=len)
            np.random.shuffle(seed_nodes[largest_node_type])
            num_node_remove = num_nodes - batch_size
            seed_nodes[largest_node_type] = seed_nodes[largest_node_type][:-num_node_remove]

        return seed_nodes

    def collate_HAN(self, iloc):
        if not isinstance(iloc, torch.Tensor):
            iloc = torch.tensor(iloc)

        if isinstance(self.dataset, HANDataset):
            X = {"adj": self.data["adj"][:len(self.metapaths)],
                 "x": self.data["x"] if hasattr(self.data, "x") else None,
                 "idx": self.y_index_dict[self.head_node_type][iloc]}
        else:
            X = {
                "adj": [(self.edge_index_dict[i], torch.ones(self.edge_index_dict[i].size(1))) for i in self.metapaths],
                "x": None,
                "idx": self.y_index_dict[self.head_node_type][iloc]}

        y = self.y_dict[self.head_node_type][iloc]
        return X, y, None

    def collate_HAN_batch(self, iloc):
        if not isinstance(iloc, torch.Tensor):
            iloc = torch.tensor(iloc)

        node_index = self.y_index_dict[self.head_node_type][iloc]

        X = {"adj": [(self.get_adj_edgelist(self.graphs[i], node_index),
                      torch.ones(self.get_adj_edgelist(self.graphs[i], node_index).size(1))) for i in self.metapaths],
             "x": self.data["x"][node_index] if hasattr(self.data, "x") else None,
             "idx": node_index}

        y = self.y_dict[self.head_node_type][iloc]
        return X, y, None

    def get_adj_edgelist(self, graph, nodes_A, nodes_B=None):
        if nodes_B == None:
            adj = nx.adj_matrix(graph, nodelist=nodes_A.numpy() if isinstance(nodes_A, torch.Tensor) else nodes_A)
        else:
            adj = nx.algorithms.bipartite.biadjacency_matrix(graph, row_order=nodes_A, column_order=nodes_B)

        adj = adj.tocoo()
        edge_index = torch.tensor(np.vstack([adj.row, adj.col]), dtype=torch.long)
        return edge_index

    def collate_node_attr_cls(self, iloc):
        if not isinstance(iloc, torch.Tensor):
            iloc = torch.tensor(iloc)

        X = {}
        X[self.head_node_type] = self.x_dict[self.head_node_type][iloc]
        X.update(self.edge_index_dict)

        return X, self.y_dict[self.head_node_type][iloc], None

    def collate_index_cls(self, iloc):
        if not isinstance(iloc, torch.Tensor):
            iloc = torch.tensor(iloc)

        X = {}
        X[self.head_node_type] = self.y_index_dict[self.head_node_type][iloc]
        X.update(self.edge_index_dict)

        return X, self.y_dict[self.head_node_type][iloc], None

class GeneratorDataset(torch.utils.data.Dataset):
    def __init__(self, generator):
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
    def __new__(cls, generator, output_types=None, output_shapes=None):
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