from collections import OrderedDict
import networkx as nx
import pandas as pd
import numpy as np

import torch
from cogdl.datasets.gtn_data import GTNDataset, ACM_GTNDataset, DBLP_GTNDataset, IMDB_GTNDataset
from cogdl.datasets.han_data import HANDataset, ACM_HANDataset, DBLP_HANDataset, IMDB_HANDataset

from ogb.nodeproppred import PygNodePropPredDataset

from torch.utils import data

from torch_geometric.data import NeighborSampler
from torch_geometric.utils.hetero import group_hetero_graph
import torch_sparse

from conv import is_negative


def load_node_dataset(dataset, method, hparams, train_ratio=None, dir_path="~/Bioinformatics_ExternalData/OGB/"):
    if dataset == "ACM":
        if method == "HAN" or method == "MetaPath2Vec":
            dataset = HeteroNeighborSampler(ACM_HANDataset(), [25, 20], node_types=["P"],
                                            metapaths=["PAP", "PSP"] if "LATTE" in method else None,
                                            add_reverse_metapaths=True,
                                            head_node_type="P", resample_train=train_ratio, inductive=hparams.inductive)
        else:
            dataset = HeteroNeighborSampler(ACM_GTNDataset(), [25, 20], node_types=["P"],
                                            metapaths=["PAP", "PA_P", "PSP", "PS_P"] if "LATTE" in method else None,
                                            add_reverse_metapaths=False,
                                            head_node_type="P", resample_train=train_ratio, inductive=hparams.inductive)

    elif dataset == "DBLP":
        if method == "HAN":
            dataset = HeteroNeighborSampler(DBLP_HANDataset(), [25, 20],
                                            node_types=["A"], head_node_type="A", metapaths=None,
                                            add_reverse_metapaths=True,
                                            resample_train=train_ratio, inductive=hparams.inductive)
        elif "LATTE" in method:
            dataset = HeteroNeighborSampler(DBLP_HANDataset(), [25, 20],
                                            node_types=["A", "P", "C", "T"], head_node_type="A",
                                            metapaths=["AC", "AP", "AT"],
                                            add_reverse_metapaths=True,
                                            resample_train=train_ratio, inductive=hparams.inductive)
        else:
            dataset = HeteroNeighborSampler(DBLP_GTNDataset(), [25, 20], node_types=["A"], head_node_type="A",
                                            metapaths=["APA", "AP_A", "ACA", "AC_A"] if "LATTE" in method else None,
                                            add_reverse_metapaths=False,
                                            resample_train=train_ratio, inductive=hparams.inductive)

    elif dataset == "IMDB":
        if method == "HAN" or method == "MetaPath2Vec":
            dataset = HeteroNeighborSampler(IMDB_HANDataset(), [25, 20], node_types=["M"],
                                            metapaths=["MAM", "MDM", "MWM"] if "LATTE" in method else None,
                                            add_reverse_metapaths=True,
                                            head_node_type="M",
                                            resample_train=train_ratio, inductive=hparams.inductive)
        else:
            dataset = HeteroNeighborSampler(IMDB_GTNDataset(), neighbor_sizes=[25, 20], node_types=["M"],
                                            metapaths=["MDM", "MD_M", "MAM", "MA_M"] if "LATTE" in method else None,
                                            add_reverse_metapaths=False,
                                            head_node_type="M", inductive=hparams.inductive)
    else:
        raise Exception(f"dataset {dataset} not found")
    return dataset

class Network:
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

    def get_projection_pos(self, embeddings_all, UMAP: classmethod, n_components=2):
        pos = UMAP(n_components=n_components).fit_transform(embeddings_all)
        pos = {embeddings_all.index[i]: pair for i, pair in enumerate(pos)}
        return pos

    def get_node_degrees(self, directed=True):
        index = pd.concat([pd.DataFrame(range(v), [k, ] * v) for k, v in self.num_nodes_dict.items()],
                          axis=0).reset_index()
        multi_index = pd.MultiIndex.from_frame(index, names=["node_type", "node"])

        metapaths = list(self.edge_index_dict.keys())
        metapath_names = [".".join(metapath) if isinstance(metapath, tuple) else metapath for metapath in
                          metapaths]
        self.node_degrees = pd.DataFrame(data=0, index=multi_index,
                                         columns=metapath_names)

        for metapath, name in zip(metapaths, metapath_names):
            edge_index = self.edge_index_dict[metapath]

            head, tail = metapath[0], metapath[-1]
            D = torch_sparse.SparseTensor(row=edge_index[0], col=edge_index[1],
                                          sparse_sizes=(self.num_nodes_dict[head],
                                                        self.num_nodes_dict[tail]))

            self.node_degrees.loc[(head, name)] = (
                    self.node_degrees.loc[(head, name)] + D.storage.rowcount().numpy()).values
            if not directed:
                self.node_degrees.loc[(tail, name)] = (
                        self.node_degrees.loc[(tail, name)] + D.storage.colcount().numpy()).values

        return self.node_degrees

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
        if hasattr(self, "y_dict") and len(self.y_dict) > 0:
            labels = pd.Series(
                self.y_dict[self.head_node_type][global_node_index[self.head_node_type]].squeeze(-1).numpy(),
                index=embeddings[0].index,
                dtype=str)
        else:
            labels = None

        return embeddings_all, types_all, labels


class HeteroNetDataset(torch.utils.data.Dataset, Network):
    def __init__(self, dataset, node_types=None, metapaths=None, head_node_type=None, directed=True,
                 resample_train: float = None, add_reverse_metapaths=True, inductive=True):
        """
        This class handles processing of the data & train/test spliting.
        :param dataset:
        :param node_types:
        :param metapaths:
        :param head_node_type:
        :param directed:
        :param resample_train:
        :param add_reverse_metapaths:
        """
        self.dataset = dataset
        self.directed = directed
        self.use_reverse = add_reverse_metapaths
        self.node_types = node_types
        self.head_node_type = head_node_type
        self.inductive = inductive

        # PyTorchGeometric Dataset
        if isinstance(dataset, HANDataset) or isinstance(dataset, GTNDataset):
            print(f"{dataset.__class__.__name__}")
            self.process_COGDLdataset(dataset, metapaths, node_types, resample_train)
        else:
            raise Exception(f"Unsupported dataset {dataset}")

        if hasattr(self, "y_dict"):
            if self.y_dict[self.head_node_type].dim() > 1 and self.y_dict[self.head_node_type].size(-1) != 1:
                self.multilabel = True
                self.classes = torch.arange(self.y_dict[self.head_node_type].size(1))
                self.class_counts = self.y_dict[self.head_node_type].sum(0)
            else:
                self.multilabel = False

                mask = self.y_dict[self.head_node_type] != -1
                labels = self.y_dict[self.head_node_type][mask]
                self.classes = labels.unique()

                if self.y_dict[self.head_node_type].dim() > 1:
                    labels = labels.squeeze(-1).numpy()
                else:
                    labels = labels.numpy()
                self.class_counts = pd.Series(labels).value_counts(sort=False)

            self.n_classes = self.classes.size(0)
            self.class_weight = torch.true_divide(1, torch.tensor(self.class_counts, dtype=torch.float))

            assert -1 not in self.classes
            assert self.class_weight.numel() == self.n_classes, f"self.class_weight {self.class_weight.numel()}, n_classes {self.n_classes}"
        else:
            self.multilabel = False
            self.n_classes = None
            print("WARNING: Dataset doesn't have node label (y_dict attribute).")

        assert hasattr(self, "num_nodes_dict")

        if not hasattr(self, "x_dict") or len(self.x_dict) == 0:
            self.x_dict = {}

        if resample_train is not None and resample_train > 0:
            self.resample_training_idx(resample_train)
        else:
            print("train_ratio", self.get_train_ratio())
        self.train_ratio = self.get_train_ratio()

    def name(self):
        if not hasattr(self, "_name"):
            return self.dataset.__class__.__name__
        else:
            return self._name

    @property
    def node_attr_shape(self):
        if not hasattr(self, "x_dict") or len(self.x_dict) == 0:
            node_attr_shape = {}
        else:
            node_attr_shape = {k: v.size(1) for k, v in self.x_dict.items()}
        return node_attr_shape

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

    def resample_training_idx(self, train_ratio):
        all_idx = torch.cat([self.training_idx, self.validation_idx, self.testing_idx])
        self.training_idx, self.validation_idx, self.testing_idx = \
            self.split_train_val_test(train_ratio=train_ratio, sample_indices=all_idx)
        print(f"Resampled training set at {self.get_train_ratio()}%")

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
            if is_negative(metapath) or edge_index_dict[metapath] == None: continue
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
        elif isinstance(metapath, (int, np.int)):
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

    def process_COGDLdataset(self, dataset: HANDataset, metapath, node_types, train_ratio):
        data = dataset.data
        assert self.head_node_type is not None
        assert node_types is not None
        print(f"Edge_types: {len(data['adj'])}")
        self.node_types = node_types
        if metapath is not None:
            self.edge_index_dict = {metapath: data["adj"][i][0] for i, metapath in enumerate(metapath)}
        else:
            self.edge_index_dict = {f"{self.head_node_type}{i}{self.head_node_type}": data["adj"][i][0] \
                                    for i in range(len(data["adj"]))}
        self.edge_types = list(range(dataset.num_edge))
        self.metapaths = list(self.edge_index_dict.keys())
        self.x_dict = {self.head_node_type: data["x"]}
        self.in_features = data["x"].size(1)

        self.training_idx, self.training_target = data["train_node"], data["train_target"]
        self.validation_idx, self.validation_target = data["valid_node"], data["valid_target"]
        self.testing_idx, self.testing_target = data["test_node"], data["test_target"]

        self.y_index_dict = {self.head_node_type: torch.cat([self.training_idx, self.validation_idx, self.testing_idx])}
        self.num_nodes_dict = self.get_num_nodes_dict(self.edge_index_dict)

        # Create new labels vector for all nodes, with -1 for nodes without label
        self.y_dict = {
            self.head_node_type: torch.cat([self.training_target, self.validation_target, self.testing_target])}

        new_y_dict = {nodetype: -torch.ones(self.num_nodes_dict[nodetype] + 1).type_as(self.y_dict[nodetype]) \
                      for nodetype in self.y_dict}
        for node_type in self.y_dict:
            new_y_dict[node_type][self.y_index_dict[node_type]] = self.y_dict[node_type]
        self.y_dict = new_y_dict

        if self.inductive:
            other_nodes = torch.arange(self.num_nodes_dict[self.head_node_type])
            idx = ~np.isin(other_nodes, self.training_idx) & \
                  ~np.isin(other_nodes, self.validation_idx) & \
                  ~np.isin(other_nodes, self.testing_idx)
            other_nodes = other_nodes[idx]
            self.training_subgraph_idx = torch.cat(
                [self.training_idx, torch.tensor(other_nodes, dtype=self.training_idx.dtype)],
                dim=0).unique()

        self.data = data

    def train_dataloader(self, collate_fn=None, batch_size=128, num_workers=12, **kwargs):
        loader = data.DataLoader(self.training_idx, batch_size=batch_size,
                                 shuffle=True, num_workers=num_workers,
                                 collate_fn=collate_fn if callable(collate_fn) \
                                     else self.get_collate_fn(collate_fn, mode="train", **kwargs))
        return loader

    def valtrain_dataloader(self, collate_fn=None, batch_size=128, num_workers=12, **kwargs):
        loader = data.DataLoader(torch.cat([self.training_idx, self.validation_idx]), batch_size=batch_size,
                                 shuffle=True, num_workers=num_workers,
                                 collate_fn=collate_fn if callable(collate_fn) \
                                     else self.get_collate_fn(collate_fn, mode="validation", **kwargs))
        return loader

    def valid_dataloader(self, collate_fn=None, batch_size=128, num_workers=4, **kwargs):
        loader = data.DataLoader(self.validation_idx, batch_size=batch_size,
                                 shuffle=True, num_workers=num_workers,
                                 collate_fn=collate_fn if callable(collate_fn) \
                                     else self.get_collate_fn(collate_fn, mode="validation", **kwargs))
        return loader

    def test_dataloader(self, collate_fn=None, batch_size=128, num_workers=4, **kwargs):
        loader = data.DataLoader(self.testing_idx, batch_size=batch_size,
                                 shuffle=True, num_workers=num_workers,
                                 collate_fn=collate_fn if callable(collate_fn) \
                                     else self.get_collate_fn(collate_fn, mode="testing", **kwargs))
        return loader

    def get_collate_fn(self, collate_fn: str, mode=None, **kwargs):

        def collate_wrapper(iloc):
            if "HAN_batch" in collate_fn:
                return self.collate_HAN_batch(iloc, mode=mode)
            elif "HAN" in collate_fn:
                return self.collate_HAN(iloc, mode=mode)
            else:
                raise Exception(f"Correct collate function {collate_fn} not found.")

        return collate_wrapper

    def filter_edge_index(self, input, allowed_nodes):
        if isinstance(input, tuple):
            edge_index = input[0]
            values = edge_index[1]
        else:
            edge_index = input
            values = None

        mask = np.isin(edge_index[0], allowed_nodes) & np.isin(edge_index[1], allowed_nodes)
        edge_index = edge_index[:, mask]

        if values == None:
            values = torch.ones(edge_index.size(1))
        else:
            values = values[mask]

        return (edge_index, values)

    def collate_HAN(self, iloc, mode=None):
        if not isinstance(iloc, torch.Tensor):
            iloc = torch.tensor(iloc)

        if "train" in mode:
            filter = True if self.inductive else False
            if self.inductive and hasattr(self, "training_subgraph_idx"):
                allowed_nodes = self.training_subgraph_idx
            else:
                allowed_nodes = self.training_idx
        elif "valid" in mode:
            filter = True if self.inductive else False
            if self.inductive and hasattr(self, "training_subgraph_idx"):
                allowed_nodes = torch.cat([self.validation_idx, self.training_subgraph_idx])
            else:
                allowed_nodes = self.validation_idx
        elif "test" in mode:
            filter = False
            allowed_nodes = self.testing_idx
        else:
            filter = False
            print("WARNING: should pass a value in `mode` in collate_HAN()")

        if isinstance(self.dataset, HANDataset):
            X = {"adj": [(edge_index, values) \
                             if not filter else self.filter_edge_index((edge_index, values), allowed_nodes) \
                         for edge_index, values in self.data["adj"][:len(self.metapaths)]],
                 "x": self.data["x"] if hasattr(self.data, "x") else None,
                 "idx": iloc}
        else:
            X = {
                "adj": [(self.edge_index_dict[i], torch.ones(self.edge_index_dict[i].size(1))) \
                            if not filter else self.filter_edge_index(self.edge_index_dict[i], allowed_nodes) \
                        for i in self.metapaths],
                "x": self.data["x"] if hasattr(self.data, "x") else None,
                "idx": iloc}

        X["adj"] = [edge for edge in X["adj"] if edge[0].size(1) > 0]

        y = self.y_dict[self.head_node_type][iloc]
        return X, y, None

    def collate_HAN_batch(self, iloc, mode=None):
        if not isinstance(iloc, torch.Tensor):
            iloc = torch.tensor(iloc)

        X_batch, y, weights = self.sample(iloc, mode=mode)  # uses HeteroNetSampler PyG sampler method

        X = {}
        X["adj"] = [(X_batch["edge_index_dict"][metapath], torch.ones(X_batch["edge_index_dict"][metapath].size(1))) \
                    for metapath in self.metapaths if metapath in X_batch["edge_index_dict"]]
        X["x"] = self.data["x"][X_batch["global_node_index"][self.head_node_type]]
        X["idx"] = X_batch["global_node_index"][self.head_node_type]

        return X, y, weights

    def get_train_ratio(self):
        if self.validation_idx.size() != self.testing_idx.size() or not (self.validation_idx == self.testing_idx).all():
            train_ratio = self.training_idx.numel() / \
                          sum([self.training_idx.numel(), self.validation_idx.numel(), self.testing_idx.numel()])
        else:
            train_ratio = self.training_idx.numel() / sum([self.training_idx.numel(), self.validation_idx.numel()])
        return train_ratio


class HeteroNeighborSampler(HeteroNetDataset):
    def __init__(self, dataset, neighbor_sizes, node_types=None, metapaths=None, head_node_type=None, directed=True,
                 resample_train=None, add_reverse_metapaths=True, inductive=False):
        self.neighbor_sizes = neighbor_sizes
        super(HeteroNeighborSampler, self).__init__(dataset, node_types, metapaths, head_node_type, directed,
                                                    resample_train, add_reverse_metapaths, inductive)

        if self.use_reverse:
            self.add_reverse_edge_index(self.edge_index_dict)

        # Ensure head_node_type is first item in num_nodes_dict, since NeighborSampler.sample() function takes in index only the first
        num_nodes_dict = OrderedDict([(node_type, self.num_nodes_dict[node_type]) for node_type in self.node_types])

        self.edge_index, self.edge_type, self.node_type, self.local_node_idx, self.local2global, self.key2int = \
            group_hetero_graph(self.edge_index_dict, num_nodes_dict)

        self.int2node_type = {type_int: node_type for node_type, type_int in self.key2int.items() if
                              node_type in self.node_types}
        self.int2edge_type = {type_int: edge_type for edge_type, type_int in self.key2int.items() if
                              edge_type in self.edge_index_dict}

        self.neighbor_sampler = NeighborSampler(self.edge_index, node_idx=self.training_idx,
                                                sizes=self.neighbor_sizes, batch_size=128, shuffle=True)

    def process_PygNodeDataset_hetero(self, dataset: PygNodePropPredDataset, ):
        data = dataset[0]
        self._name = dataset.name
        self.edge_index_dict = data.edge_index_dict
        self.num_nodes_dict = data.num_nodes_dict if hasattr(data, "num_nodes_dict") else self.get_num_nodes_dict(
            self.edge_index_dict)

        if self.node_types is None:
            self.node_types = list(self.num_nodes_dict.keys())

        if hasattr(data, "x_dict"):
            self.x_dict = data.x_dict
        elif hasattr(data, "x"):
            self.x_dict = {self.head_node_type: data.x}
        else:
            self.x_dict = {}

        if hasattr(data, "y_dict"):
            self.y_dict = data.y_dict
        elif hasattr(data, "y"):
            self.y_dict = {self.head_node_type: data.y}
        else:
            self.y_dict = {}

        self.y_index_dict = {node_type: torch.arange(self.num_nodes_dict[node_type]) for node_type in
                             self.y_dict.keys()}

        if self.head_node_type is None:
            if hasattr(self, "y_dict"):
                self.head_node_type = list(self.y_dict.keys())[0]
            else:
                self.head_node_type = self.node_types[0]

        self.metapaths = list(self.edge_index_dict.keys())

        split_idx = dataset.get_idx_split()
        self.training_idx, self.validation_idx, self.testing_idx = split_idx["train"][self.head_node_type], \
                                                                   split_idx["valid"][self.head_node_type], \
                                                                   split_idx["test"][self.head_node_type]

    def process_PygNodeDataset_homo(self, dataset: PygNodePropPredDataset, ):
        data = dataset[0]
        self._name = dataset.name
        self.head_node_type = "entity"

        if not hasattr(data, "edge_reltype") and not hasattr(data, "edge_attr"):
            self.metapaths = [(self.head_node_type, "default", self.head_node_type)]
            self.edge_index_dict = {self.metapaths[0]: data.edge_index}
            self.num_nodes_dict = self.get_num_nodes_dict(self.edge_index_dict)


        elif False and hasattr(data, "edge_attr") and hasattr(data, "node_species"):  # for ogbn-proteins
            self.edge_index_dict = {}
            edge_reltype = data.edge_attr.argmax(1)

            for node_type in data.node_species.unique():
                for edge_type in range(data.edge_attr.size(1)):
                    edge_mask = edge_reltype == edge_type
                    node_mask = (data.node_species[data.edge_index[0]].squeeze(-1) == node_type).logical_and(
                        data.node_species[data.edge_index[1]].squeeze(-1) == node_type)

                    edge_index = data.edge_index[:, node_mask.logical_and(edge_mask)]

                    if edge_index.size(1) == 0: continue
                    self.edge_index_dict[(str(node_type.item()), str(edge_type), str(node_type.item()))] = edge_index

            self.num_nodes_dict = {str(node_type.item()): data.node_species.size(0) for node_type in
                                   data.node_species.unique()}
            self.metapaths = list(self.edge_index_dict.keys())
            self.head_node_type = self.metapaths[0][0]
            self.y_dict = {node_type: data.y for node_type in self.num_nodes_dict}
            # TODO need to convert global node_index to local index

        elif hasattr(data, "edge_attr"):  # for ogbn-proteins
            self.edge_index_dict = {}
            edge_reltype = data.edge_attr.argmax(1)

            for edge_type in range(data.edge_attr.size(1)):
                mask = edge_reltype == edge_type
                edge_index = data.edge_index[:, mask]

                if edge_index.size(1) == 0: continue
                self.edge_index_dict[(self.head_node_type, str(edge_type), self.head_node_type)] = edge_index

            self.metapaths = list(self.edge_index_dict.keys())
            self.num_nodes_dict = self.get_num_nodes_dict(self.edge_index_dict)

        else:
            raise Exception("Something wrong here")

        if self.node_types is None:
            self.node_types = list(self.num_nodes_dict.keys())

        self.x_dict = {self.head_node_type: data.x} if hasattr(data, "x") and data.x is not None else {}
        if not hasattr(self, "y_dict"):
            self.y_dict = {self.head_node_type: data.y} if hasattr(data, "y") else {}

        self.metapaths = list(self.edge_index_dict.keys())

        split_idx = dataset.get_idx_split()
        self.training_idx, self.validation_idx, self.testing_idx = split_idx["train"], split_idx["valid"], split_idx[
            "test"]

    def get_collate_fn(self, collate_fn: str, mode=None):
        assert mode is not None, "Must pass arg `mode` at get_collate_fn(). {'train', 'valid', 'test'}"

        def collate_wrapper(iloc):
            return self.sample(iloc, mode=mode)

        if "neighbor_sampler" in collate_fn:
            return collate_wrapper
        else:
            return super().get_collate_fn(collate_fn, mode=mode)

    def get_local_nodes_dict(self, adjs, n_id):
        """

        :param iloc: A tensor of indices for nodes of `head_node_type`
        :return sampled_nodes, n_id, adjs:
        """
        sampled_nodes = {}
        for adj in adjs:
            for row_col in [0, 1]:
                node_ids = n_id[adj.edge_index[row_col]]
                node_types = self.node_type[node_ids]

                for node_type_id in node_types.unique():
                    mask = node_types == node_type_id
                    local_node_ids = self.local_node_idx[node_ids[mask]]
                    sampled_nodes.setdefault(self.int2node_type[node_type_id.item()], []).append(local_node_ids)

        # Concatenate & remove duplicate nodes
        sampled_nodes = {k: torch.cat(v, dim=0).unique() for k, v in sampled_nodes.items()}
        return sampled_nodes

    def sample(self, iloc, mode):
        """

        :param iloc: A tensor of a batch of indices in training_idx, validation_idx, or testing_idx
        :return:
        """
        if not isinstance(iloc, torch.Tensor):
            iloc = torch.tensor(iloc)

        batch_size, n_id, adjs = self.neighbor_sampler.sample(self.local2global[self.head_node_type][iloc])
        if not isinstance(adjs, list):
            adjs = [adjs]
        # Sample neighbors and return `sampled_local_nodes` as the set of all nodes traversed (in local index)
        sampled_local_nodes = self.get_local_nodes_dict(adjs, n_id)

        # Ensure the sampled nodes only either belongs to training, validation, or testing set
        if "train" in mode:
            filter = True if self.inductive else False
            if self.inductive and hasattr(self, "training_subgraph_idx"):
                allowed_nodes = self.training_subgraph_idx
            else:
                allowed_nodes = self.training_idx
        elif "valid" in mode:
            filter = True if self.inductive else False
            if self.inductive and hasattr(self, "training_subgraph_idx"):
                allowed_nodes = torch.cat([self.validation_idx, self.training_subgraph_idx])
            else:
                allowed_nodes = self.validation_idx
        elif "test" in mode:
            filter = False
            allowed_nodes = self.testing_idx
        else:
            raise Exception(f"Must set `mode` to either 'training', 'validation', or 'testing'. mode={mode}")

        if filter:
            node_mask = np.isin(sampled_local_nodes[self.head_node_type], allowed_nodes)
            sampled_local_nodes[self.head_node_type] = sampled_local_nodes[self.head_node_type][node_mask]

        # `global_node_index` here actually refers to the 'local' type-specific index of the original graph
        X = {"edge_index_dict": {},
             "global_node_index": sampled_local_nodes,
             "x_dict": {}}

        local2batch = {
            node_type: dict(zip(sampled_local_nodes[node_type].numpy(),
                                range(len(sampled_local_nodes[node_type])))
                            ) for node_type in sampled_local_nodes}

        X["edge_index_dict"] = self.get_local_edge_index_dict(adjs=adjs, n_id=n_id,
                                                              sampled_local_nodes=sampled_local_nodes,
                                                              local2batch=local2batch,
                                                              filter_nodes=filter)

        # x_dict attributes
        if hasattr(self, "x_dict") and len(self.x_dict) > 0:
            X["x_dict"] = {node_type: self.x_dict[node_type][X["global_node_index"][node_type]] \
                           for node_type in self.x_dict}

        # y_dict
        if len(self.y_dict) > 1:
            y = {node_type: y_true[X["global_node_index"][node_type]] for node_type, y_true in self.y_dict.items()}
        else:
            y = self.y_dict[self.head_node_type][X["global_node_index"][self.head_node_type]].squeeze(-1)

        weights = (y != -1) & np.isin(X["global_node_index"][self.head_node_type], allowed_nodes)
        weights = torch.tensor(weights, dtype=torch.float)

        if hasattr(self, "x_dict") and len(self.x_dict) > 0:
            assert X["global_node_index"][self.head_node_type].size(0) == X["x_dict"][self.head_node_type].size(0)

        # assert y.size(0) == X["global_node_index"][self.head_node_type].size(0)
        # assert y.size(0) == weights.size(0)
        return X, y, weights

    def get_local_edge_index_dict(self, adjs, n_id, sampled_local_nodes: dict, local2batch: dict,
                                  filter_nodes: bool):
        """
        # Conbine all edge_index's and convert local node id to "batch node index" that aligns with `x_dict` and `global_node_index`
        :param adjs:
        :param n_id:
        :param sampled_local_nodes:
        :param local2batch:
        :param filter_nodes:
        :return:
        """
        edge_index_dict = {}
        for adj in adjs:
            for edge_type_id in self.edge_type[adj.e_id].unique():
                metapath = self.int2edge_type[edge_type_id.item()]
                head_type, tail_type = metapath[0], metapath[-1]

                # Filter edges to correct edge_type_id
                edge_mask = self.edge_type[adj.e_id] == edge_type_id
                edge_index = adj.edge_index[:, edge_mask]

                # convert from "sampled_edge_index" to global index
                edge_index[0] = n_id[edge_index[0]]
                edge_index[1] = n_id[edge_index[1]]

                if filter_nodes:
                    # If node_type==self.head_node_type, then remove edge_index with nodes not in allowed_nodes_idx
                    allowed_nodes_idx = self.local2global[self.head_node_type][sampled_local_nodes[self.head_node_type]]
                    if head_type == self.head_node_type and tail_type == self.head_node_type:
                        mask = np.isin(edge_index[0], allowed_nodes_idx) & np.isin(edge_index[1], allowed_nodes_idx)
                        edge_index = edge_index[:, mask]
                    elif head_type == self.head_node_type:
                        mask = np.isin(edge_index[0], allowed_nodes_idx)
                        edge_index = edge_index[:, mask]
                    elif tail_type == self.head_node_type:
                        mask = np.isin(edge_index[1], allowed_nodes_idx)
                        edge_index = edge_index[:, mask]

                # Convert node global index -> local index -> batch index
                edge_index[0] = self.local_node_idx[edge_index[0]].apply_(local2batch[head_type].get)
                edge_index[1] = self.local_node_idx[edge_index[1]].apply_(local2batch[tail_type].get)

                edge_index_dict.setdefault(metapath, []).append(edge_index)
        # Join edges from the adjs
        edge_index_dict = {metapath: torch.cat(edge_index, dim=1) \
                           for metapath, edge_index in edge_index_dict.items()}
        # Ensure no duplicate edge from adjs[0] to adjs[1]...
        edge_index_dict = {metapath: edge_index[:, self.nonduplicate_indices(edge_index)] \
                           for metapath, edge_index in edge_index_dict.items()}
        return edge_index_dict

    def nonduplicate_indices(self, edge_index):
        edge_df = pd.DataFrame(edge_index.t().numpy())  # shape: (n_edges, 2)
        return ~edge_df.duplicated(subset=[0, 1])
