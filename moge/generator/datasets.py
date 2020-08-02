from scipy.io import loadmat

import numpy as np

import tensorflow as tf
import torch
from cogdl.datasets.gtn_data import GTNDataset
from cogdl.datasets.han_data import HANDataset

from ogb.linkproppred import PygLinkPropPredDataset
from ogb.nodeproppred import PygNodePropPredDataset

from torch.utils import data
from torch_geometric.data import InMemoryDataset

class HeteroNetDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, node_types, metapaths=None, head_node_type=None, directed=True, train_ratio=0.7,
                 add_reverse_metapaths=True, process_graphs=True):
        """
        This class handles processing of the data & train/test spliting.
        :param process_graphs:
        :param dataset:
        :param node_types:
        :param metapaths:
        :param head_node_type:
        :param directed:
        :param train_ratio:
        :param add_reverse_metapaths:
        """
        self.dataset = dataset
        self.directed = directed
        self.train_ratio = train_ratio
        self.use_reverse = add_reverse_metapaths
        self.node_types = node_types
        self.head_node_type = head_node_type

        if self.node_types is not None:
            assert self.node_types[0] == self.head_node_type

        # PyTorchGeometric Dataset
        if isinstance(dataset, PygNodePropPredDataset):
            print("PygNodePropPredDataset")
            self.process_PygNodeDataset(dataset, train_ratio)
        elif isinstance(dataset, PygLinkPropPredDataset):
            print("PygLinkPropPredDataset")
            self.process_PygLinkDataset(dataset, train_ratio)
        elif isinstance(dataset, InMemoryDataset):
            print("InMemoryDataset")
            self.process_inmemorydataset(dataset, train_ratio)
        # StellarGraph Dataset
        # elif isinstance(dataset, DatasetLoader):
        #     print("StellarGraph Dataset")
        #     self.process_stellargraph(dataset, metapaths, node_types, train_ratio)
        elif isinstance(dataset, HANDataset) or isinstance(dataset, GTNDataset):
            print("HANDataset/GTNDataset")
            self.process_HANdataset(dataset, metapaths, node_types, train_ratio)
        elif "blogcatalog6k" in dataset:
            self.process_BlogCatalog6k(dataset, train_ratio)
        else:
            raise Exception(f"Unsupported dataset {dataset}")

        if hasattr(self, "y_dict"):
            if self.y_dict[self.head_node_type].dim() > 1 and self.y_dict[self.head_node_type].size(-1) != 1:
                self.multilabel = True
                self.classes = self.y_dict[self.head_node_type].unique()
                self.n_classes = self.y_dict[self.head_node_type].size(1)
            else:
                self.multilabel = False
                self.classes = self.y_dict[self.head_node_type].unique()
                self.n_classes = self.classes.size(0)
        else:
            print("WARNING: Dataset doesn't have node label (y_dict attribute).")

        if process_graphs:
            self.process_graph_sampler()

        assert hasattr(self, "num_nodes_dict")
        if not hasattr(self, "node_attr_shape"):
            self.node_attr_shape = {}
        if not hasattr(self, "x_dict"):
            self.x_dict = {}

    def process_graph_sampler(self):
        raise NotImplementedError

    def name(self):
        if not hasattr(self, "_name"):
            return self.dataset.__class__.__name__
        else:
            return self._name

    def split_train_val_test(self, train_ratio, sample_indices=None):
        if sample_indices is not None:
            indices = sample_indices[torch.randperm(sample_indices.size(0))]
        else:
            indices = torch.randperm(self.num_nodes_dict[self.head_node_type])

        num_indices = indices.size(0)
        training_idx = indices[:int(num_indices * train_ratio)]
        validation_idx = indices[int(num_indices * train_ratio):]
        testing_idx = indices[int(num_indices * train_ratio):]
        return training_idx, validation_idx, testing_idx

    def get_metapaths(self):
        if self.use_reverse:
            return self.metapaths + self.get_reverse_metapath(self.metapaths)
        else:
            return self.metapaths

    @staticmethod
    def add_reverse_edge_index(edge_index_dict) -> None:
        reverse_edge_index_dict = {}
        for metapath in edge_index_dict:
            if edge_index_dict[metapath] == None: continue
            reverse_metapath = tuple(a + "_by" if i == 1 else a for i, a in enumerate(reversed(metapath)))
            reverse_edge_index_dict[reverse_metapath] = edge_index_dict[metapath][[1, 0], :]
        edge_index_dict.update(reverse_edge_index_dict)

    @staticmethod
    def get_reverse_metapath(metapaths) -> list:
        reverse_metapaths = []
        for metapath in metapaths:
            reverse = tuple(a + "_by" if i == 1 else a for i, a in enumerate(reversed(metapath)))
            reverse_metapaths.append(reverse)
        return reverse_metapaths

    @staticmethod
    def adj_to_edgeindex(adj):
        adj = adj.tocoo(copy=False)
        return torch.tensor(np.vstack((adj.row, adj.col)).astype("long"))

    def process_BlogCatalog6k(self, dataset, train_ratio):
        data = loadmat(dataset)  # From http://dmml.asu.edu/users/xufei/Data/blogcatalog6k.mat
        self._name = dataset.name
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

    def process_stellargraph(self, dataset, metapath, node_types, train_ratio):
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
        self.edge_index_dict = data.edge_index_dict
        self.num_nodes_dict = data.num_nodes_dict
        self.node_attr_shape = {}
        if self.node_types is None:
            self.node_types = list(data.num_nodes_dict.keys())
        self.y_dict = data.y_dict
        self.y_index_dict = data.y_index_dict

        new_y_dict = {nodetype: torch.zeros(self.y_index_dict[nodetype].max() + 1).type_as(self.y_dict[nodetype]) for
                      nodetype in self.y_dict}
        for node_type in self.y_dict:
            new_y_dict[node_type][self.y_index_dict[node_type]] = self.y_dict[node_type]
        self.y_dict = new_y_dict

        self.metapaths = list(self.edge_index_dict.keys())
        self.training_idx, self.validation_idx, self.testing_idx = \
            self.split_train_val_test(train_ratio,
                                      sample_indices=self.y_index_dict[self.head_node_type])

    def process_PygNodeDataset(self, dataset: PygNodePropPredDataset, train_ratio):
        data = dataset[0]
        self._name = dataset.name
        self.edge_index_dict = data.edge_index_dict
        self.num_nodes_dict = data.num_nodes_dict
        if self.node_types is None:
            self.node_types = list(data.num_nodes_dict.keys())
        self.x_dict = data.x_dict
        self.node_attr_shape = {node_type: x.size(1) for node_type, x in self.x_dict.items()}
        self.y_dict = data.y_dict
        self.y_index_dict = {node_type: torch.arange(data.num_nodes_dict[node_type]) for node_type in
                             data.y_dict.keys()}
        self.multilabel = False

        self.metapaths = list(self.edge_index_dict.keys())

        split_idx = dataset.get_idx_split()
        self.training_idx, self.validation_idx, self.testing_idx = split_idx["train"][self.head_node_type], \
                                                                   split_idx["valid"][self.head_node_type], \
                                                                   split_idx["test"][self.head_node_type]
        self.train_ratio = self.training_idx.numel() / \
                           sum([self.training_idx.numel(), self.validation_idx.numel(), self.testing_idx.numel()])

    def process_PygLinkDataset(self, dataset: PygLinkPropPredDataset, train_ratio):
        data = dataset[0]
        self._name = dataset.name
        self.edge_index_dict = data.edge_index_dict
        self.num_nodes_dict = data.num_nodes_dict
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

    def train_dataloader(self, collate_fn=None, batch_size=128, num_workers=12):
        loader = data.DataLoader(self.training_idx, batch_size=batch_size,
                                 shuffle=True, num_workers=num_workers,
                                 collate_fn=collate_fn if callable(collate_fn) else self.get_collate_fn(collate_fn,
                                                                                                        batch_size,
                                                                                                        mode="training"))
        return loader

    def val_dataloader(self, collate_fn=None, batch_size=128, num_workers=4):
        loader = data.DataLoader(self.validation_idx, batch_size=batch_size,
                                 shuffle=False, num_workers=num_workers,
                                 collate_fn=collate_fn if callable(collate_fn) else self.get_collate_fn(collate_fn,
                                                                                                        batch_size,
                                                                                                        mode="validation"))
        return loader

    def test_dataloader(self, collate_fn=None, batch_size=128, num_workers=4):
        loader = data.DataLoader(self.testing_idx, batch_size=batch_size,
                                 shuffle=False, num_workers=num_workers,
                                 collate_fn=collate_fn if callable(collate_fn) else self.get_collate_fn(collate_fn,
                                                                                                        batch_size,
                                                                                                        mode="testing"))
        return loader

    def get_collate_fn(self, collate_fn: str, batch_size=None, mode=None):
        if batch_size is not None:
            self.batch_size = batch_size * len(self.node_types)

        if "HAN" in collate_fn:
            return self.collate_HAN
        else:
            raise Exception(f"Correct collate function {collate_fn} not found.")

    def collate_LATTELink_batch(self, iloc):
        if not isinstance(iloc, torch.Tensor):
            iloc = torch.tensor(iloc)

        X = {"edge_index_dict": {}, "x_index_dict": {}, "x_dict": {}}

        triples = {k: v[iloc] for k, v in self.triples.items()}

        for metapath_id in triples["relation"].unique():
            metapath = self.metapaths[metapath_id]
            head_type, tail_type = metapath[0], metapath[-1]
            X["edge_index_dict"][metapath] = torch.stack([triples["head"], triples["tail"]], dim=1).t()
            X["x_index_dict"].setdefault(head_type, []).append(triples["head"])
            X["x_index_dict"].setdefault(tail_type, []).append(triples["tail"])

        X["x_index_dict"] = {k: torch.cat(v, dim=0).unique() for k, v in X["x_index_dict"].items()}

        if self.use_reverse:
            self.add_reverse_edge_index(X["edge_index_dict"])

        return X, None, None



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
