from scipy.io import loadmat
import networkx as nx
import pandas as pd
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

        if self.node_types is not None and self.head_node_type is not None:
            assert self.node_types[0] == self.head_node_type
        elif self.head_node_type is None:
            self.head_node_type = self.node_types[0]
            print(f"Selected head_node_type to be {self.head_node_type}")
        else:
            raise Exception("Must pass in node_types")

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
                self.classes = torch.arange(self.y_dict[self.head_node_type].size(1))
                self.class_counts = pd.Series(self.y_dict[self.head_node_type].sum(1).numpy()).value_counts(sort=False)
            else:
                self.multilabel = False
                self.classes = self.y_dict[self.head_node_type].unique()
                if self.y_dict[self.head_node_type].dim() > 1:
                    self.class_counts = pd.Series(self.y_dict[self.head_node_type].squeeze(-1).numpy()).value_counts(
                        sort=False)
                else:
                    self.class_counts = pd.Series(self.y_dict[self.head_node_type].numpy()).value_counts(sort=False)

            self.n_classes = self.classes.size(0)
            self.class_weight = torch.tensor(1 / self.class_counts, dtype=torch.float)
        else:
            print("WARNING: Dataset doesn't have node label (y_dict attribute).")

        if process_graphs:
            self.process_graph_sampler()

        assert hasattr(self, "num_nodes_dict")
        if not hasattr(self, "node_attr_shape"):
            self.node_attr_shape = {}
        if not hasattr(self, "x_dict"):
            self.x_dict = {}
        else:
            self.node_attr_shape = {k: v.size(1) for k, v in self.x_dict.items()}

    def process_graph_sampler(self):
        raise NotImplementedError()

    def get_networkx(self):
        if not hasattr(self, "G"):
            G = nx.Graph()
            for metapath in self.edge_index_dict:
                edgelist = self.edge_index_dict[metapath].t().numpy().astype(str)
                edgelist = np.core.defchararray.add([metapath[0][0], metapath[-1][0]], edgelist)
                edge_type = "".join([n for i, n in enumerate(metapath) if i % 2 == 1])
                G.add_edges_from(edgelist, edge_type=edge_type)

            self.G = G

        return self.G

    def get_embedding_dfs(self, embeddings_dict, global_node_index):
        embeddings = []
        for node_type in self.node_types:
            nodes = global_node_index[node_type].numpy().astype(str)
            nodes = np.core.defchararray.add(node_type[0], nodes)
            if isinstance(embeddings_dict[node_type], torch.Tensor):
                df = pd.DataFrame(embeddings_dict[node_type].detach().cpu().numpy(), index=nodes)
            else:
                df = pd.DataFrame(embeddings_dict[node_type], index=nodes)
            embeddings.append(df)

        return embeddings

    def get_embeddings_types_labels(self, embeddings, global_node_index):
        embeddings_all = pd.concat(embeddings, axis=0)

        types_all = embeddings_all.index.to_series().str.slice(0, 1)
        labels = pd.Series(
            self.y_dict[self.head_node_type][global_node_index[self.head_node_type]].squeeze(-1).numpy(),
            index=embeddings[0].index,
            dtype=str)

        return embeddings_all, types_all, labels

    def get_projection_pos(self, embeddings_all, UMAP: classmethod, n_components=2):
        pos = UMAP(n_components=n_components).fit_transform(embeddings_all)
        pos = {embeddings_all.index[i]: pair for i, pair in enumerate(pos)}
        return pos

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
            return self.metapaths + self.get_reverse_metapath(self.metapaths, self.edge_index_dict)
        else:
            return self.metapaths

    def get_num_nodes_dict(self, edge_index_dict):
        num_nodes_dict = {}
        for keys, edge_index in edge_index_dict.items():
            key = keys[0]
            N = int(edge_index[0].max() + 1)
            num_nodes_dict[key] = max(N, num_nodes_dict.get(key, N))

            key = keys[-1]
            N = int(edge_index[1].max() + 1)
            num_nodes_dict[key] = max(N, num_nodes_dict.get(key, N))
        return num_nodes_dict

    @staticmethod
    def add_reverse_edge_index(edge_index_dict) -> None:
        reverse_edge_index_dict = {}
        for metapath in edge_index_dict:
            if edge_index_dict[metapath] == None: continue
            reverse_metapath = HeteroNetDataset.get_reverse_metapath_name(metapath, edge_index_dict)

            reverse_edge_index_dict[reverse_metapath] = edge_index_dict[metapath][[1, 0], :]
        edge_index_dict.update(reverse_edge_index_dict)

    @staticmethod
    def get_reverse_metapath_name(metapath, edge_index_dict):
        if isinstance(metapath, tuple):
            reverse_metapath = tuple(a + "_by" if i == 1 else a for i, a in enumerate(reversed(metapath)))
        elif isinstance(metapath, str):
            reverse_metapath = "".join(reversed(metapath))
            if reverse_metapath in edge_index_dict:
                reverse_metapath = reverse_metapath[:2] + "_" + reverse_metapath[2:]
        elif isinstance(metapath, int):
            reverse_metapath = str(metapath) + "_"
        else:
            raise NotImplementedError(f"{metapath} not supported")
        return reverse_metapath

    @staticmethod
    def get_reverse_metapath(metapaths, edge_index_dict) -> list:
        reverse_metapaths = []
        for metapath in metapaths:
            reverse = HeteroNetDataset.get_reverse_metapath_name(metapath, edge_index_dict)
            reverse_metapaths.append(reverse)
        return reverse_metapaths

    @staticmethod
    def sps_adj_to_edgeindex(adj):
        adj = adj.tocoo(copy=False)
        return torch.tensor(np.vstack((adj.row, adj.col)).astype("long"))

    def process_BlogCatalog6k(self, dataset, train_ratio):
        data = loadmat(dataset)  # From http://dmml.asu.edu/users/xufei/Data/blogcatalog6k.mat
        self._name = "BlogCatalog3"
        self.y_index_dict = {"user": torch.arange(data["friendship"].shape[0]),
                             "tag": torch.arange(data["tagnetwork"].shape[0])}
        self.node_types = ["user", "tag"]
        self.head_node_type = "user"
        self.y_dict = {self.head_node_type: torch.tensor(data["usercategory"].toarray().astype(int))}
        print("self.y_dict", {k: v.shape for k, v in self.y_dict.items()})

        self.metapaths = [("user", "usertag", "tag"),
                          ("tag", "tagnetwork", "tag"),
                          ("user", "friendship", "user"), ]
        self.edge_index_dict = {
            ("user", "friendship", "user"): self.sps_adj_to_edgeindex(data["friendship"]),
            ("user", "usertag", "tag"): self.sps_adj_to_edgeindex(data["usertag"]),
            ("tag", "tagnetwork", "tag"): self.sps_adj_to_edgeindex(data["tagnetwork"])}
        print("got here", {k: v.shape for k, v in self.edge_index_dict.items()})
        self.num_nodes_dict = self.get_num_nodes_dict(self.edge_index_dict)
        print("got here", self.num_nodes_dict)
        self.training_idx, self.validation_idx, self.testing_idx = self.split_train_val_test(train_ratio)

    def process_HANdataset(self, dataset: HANDataset, metapath, node_types, train_ratio):
        data = dataset.data
        self.edge_index_dict = {metapath: data["adj"][i][0] for i, metapath in enumerate(metapath)}
        self.node_types = node_types
        self.edge_types = list(range(dataset.num_edge))
        self.metapaths = list(self.edge_index_dict.keys())
        self.x_dict = {self.head_node_type: data["x"]}
        self.in_features = data["x"].size(1)

        self.training_idx, self.training_target = data["train_node"], data["train_target"]
        self.validation_idx, self.validation_target = data["valid_node"], data["valid_target"]
        self.testing_idx, self.testing_target = data["test_node"], data["test_target"]

        node_indices = torch.cat([self.training_idx, self.validation_idx, self.testing_idx])
        self.y_index_dict = {self.head_node_type: node_indices}
        self.num_nodes_dict = self.get_num_nodes_dict(self.edge_index_dict)

        self.y_dict = {
            self.head_node_type: torch.cat([self.training_target, self.validation_target, self.testing_target])}

        new_y_dict = {nodetype: -torch.ones(self.num_nodes_dict[nodetype] + 1).type_as(self.y_dict[nodetype]) for
                      nodetype in self.y_dict}
        for node_type in self.y_dict:
            new_y_dict[node_type][self.y_index_dict[node_type]] = self.y_dict[node_type]
        self.y_dict = new_y_dict

        self.training_idx, self.validation_idx, self.testing_idx = \
            self.split_train_val_test(train_ratio=train_ratio, sample_indices=self.y_index_dict[self.head_node_type])
        self.data = data

    def process_stellargraph(self, dataset, metapath, node_types, train_ratio):
        graph = dataset.load()
        self.node_types = graph.node_types if node_types is None else node_types
        self.metapaths = graph.metapaths
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

        new_y_dict = {nodetype: -torch.ones(self.num_nodes_dict[nodetype] + 1).type_as(self.y_dict[nodetype]) for
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
        if "HAN" in collate_fn:
            return self.collate_HAN
        else:
            raise Exception(f"Correct collate function {collate_fn} not found.")


    def collate_HAN(self, iloc):
        if not isinstance(iloc, torch.Tensor):
            iloc = torch.tensor(iloc)

        if isinstance(self.dataset, HANDataset):
            X = {"adj": self.data["adj"][:len(self.metapaths)],
                 "x": self.data["x"] if hasattr(self.data, "x") else None,
                 "idx": iloc}
        else:
            X = {
                "adj": [(self.edge_index_dict[i], torch.ones(self.edge_index_dict[i].size(1))) for i in self.metapaths],
                "x": None,
                "idx": iloc}

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
