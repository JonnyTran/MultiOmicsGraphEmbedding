from abc import abstractmethod
from typing import Union, List, Tuple

import dgl
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch_sparse
from ogb.graphproppred import DglGraphPropPredDataset
from ogb.linkproppred import PygLinkPropPredDataset, DglLinkPropPredDataset
from ogb.nodeproppred import PygNodePropPredDataset, DglNodePropPredDataset
from torch import Tensor
from torch.utils import data
from torch_geometric.data import HeteroData
from torch_geometric.data import InMemoryDataset as PyGInMemoryDataset
from torch_geometric.utils import is_undirected

import moge.model.PyG.utils
from moge.dataset.utils import get_reverse_metapaths
from moge.model.PyG.utils import get_edge_index_values


class Graph:
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

    def get_node_degrees(self, undirected=True):
        if hasattr(self, "node_degrees") and self.node_degrees is not None:
            return self.node_degrees

        if any("rev_" in metapath for metapath in self.edge_index_dict):
            undirected = False

        index = pd.concat([pd.DataFrame(range(num_nodes), [ntype, ] * num_nodes) \
                           for ntype, num_nodes in self.num_nodes_dict.items()],
                          axis=0).reset_index()
        node_type_nid = pd.MultiIndex.from_frame(index, names=["ntype", "nid"])

        metapaths = list(self.edge_index_dict.keys())
        metapath_names = [".".join(metapath) if isinstance(metapath, tuple) else metapath for metapath in
                          metapaths]
        df = pd.DataFrame(data=0, index=node_type_nid, columns=metapath_names)

        for metapath, name in zip(metapaths, metapath_names):
            edge_index = self.edge_index_dict[metapath]
            head, tail = metapath[0], metapath[-1]

            adj = torch_sparse.SparseTensor(row=edge_index[0], col=edge_index[1],
                                            sparse_sizes=(self.num_nodes_dict[head], self.num_nodes_dict[tail]))

            df.loc[(head, name)] = (df.loc[(head, name)] + adj.storage.rowcount().numpy()).values
            if undirected:
                df.loc[(tail, name)] = (df.loc[(tail, name)] + adj.storage.colcount().numpy()).values

        self.node_degrees = df
        return df

    def get_projection_pos(self, node_embs, UMAP: classmethod, n_components=2):
        pos = UMAP(n_components=n_components).fit_transform(node_embs)
        pos = {node_embs.index[i]: pair for i, pair in enumerate(pos)}
        return pos


class HeteroGraphDataset(torch.utils.data.Dataset, Graph):
    def __init__(self,
                 dataset: Union[PyGInMemoryDataset, PygNodePropPredDataset, PygLinkPropPredDataset,
                                DglNodePropPredDataset, DglLinkPropPredDataset],
                 node_types: List[str] = None,
                 metapaths: List[Tuple[str, str, str]] = None,
                 head_node_type: str = None,
                 edge_dir: str = "in",
                 reshuffle_train: float = None,
                 add_reverse_metapaths: bool = True,
                 inductive: bool = False, ):
        self.dataset = dataset
        self.edge_dir = edge_dir
        self.use_reverse = add_reverse_metapaths
        self.node_types = node_types
        self.head_node_type = head_node_type
        self.inductive = inductive

        # OGB Datasets
        if isinstance(dataset, PygNodePropPredDataset) and not hasattr(dataset[0], "edge_index_dict"):
            print("PygNodePropPredDataset Homogenous (use HeteroNeighborSampler class)")
            self.process_PygNodeDataset_homo(dataset)
        elif isinstance(dataset, PygNodePropPredDataset) and hasattr(dataset[0], "edge_index_dict"):
            print("PygNodePropPredDataset Hetero (use HeteroNeighborSampler class)")
            self.process_PygNodeDataset_hetero(dataset)

        elif isinstance(dataset, HeteroData):
            self.process_pyg_heterodata(dataset)

        # DGL Datasets
        elif isinstance(dataset, DglNodePropPredDataset):
            if len(dataset[0][0].ntypes) + len(dataset[0][0].etypes) > 2:
                print("DGLNodePropPredDataset Hetero")
                self.process_DglNodeDataset_hetero(dataset)
            else:
                print("DGLNodePropPredDataset Homo")
                self.process_DglNodeDataset_homo(dataset)
        elif isinstance(dataset, DglLinkPropPredDataset):
            print("DGLLinkPropPredDataset Hetero")
            self.process_DglLinkDataset_hetero(dataset)
        elif isinstance(dataset, DglGraphPropPredDataset):
            print("DGLGraphPropPredDataset Homo")
            if len(dataset[0][0].ntypes) + len(dataset[0][0].etypes) > 2:
                self.process_DglGraphDataset_hetero(dataset)
            else:
                self.process_DglGraphDataset_homo(dataset)

        elif isinstance(dataset, dgl.DGLGraph):
            if isinstance(dataset, dgl.DGLHeteroGraph):
                self.process_dgl_heterodata(dataset)
            elif isinstance(dataset, dgl.DGLGraph):
                raise NotImplementedError(type(dataset))

        # PyG datasets
        elif isinstance(dataset, PygLinkPropPredDataset) and hasattr(dataset[0], "edge_reltype") and \
                not hasattr(dataset[0], "edge_index_dict"):
            print("PygLink_edge_reltype_dataset Hetero (use TripletSampler class)")
            self.process_edge_reltype_dataset(dataset)
        elif isinstance(dataset, PygLinkPropPredDataset) and hasattr(dataset[0], "edge_index_dict"):
            print("PygLinkDataset Hetero (use TripletSampler class)")
            self.process_PygLinkDataset_hetero(dataset)
        elif isinstance(dataset, PygLinkPropPredDataset) and not hasattr(dataset[0], "edge_index_dict") \
                and not hasattr(dataset[0], "edge_reltype"):
            print("PygLinkDataset Homo (use EdgeSampler class)")
            self.process_PygLinkDataset_homo(dataset)

        # Pytorch Geometric Datasets
        elif isinstance(dataset, PyGInMemoryDataset):
            print("InMemoryDataset")
            self.process_inmemorydataset(dataset, train_ratio=0.5)
        elif dataset.__class__.__name__ in ["HANDataset", "GTNDataset"]:
            print(f"{dataset.__class__.__name__}")
            self.process_COGDLdataset(dataset, metapaths, node_types, reshuffle_train)

        else:
            raise Exception(f"Unsupported dataset {dataset}")

        # Node classifications
        if hasattr(self, "y_dict") and self.y_dict:
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

            assert -1 not in self.classes

        # Graph classification
        elif hasattr(self, "labels") and self.labels is not None:
            if self.labels.dim() > 1 and self.labels.size(1) != 1:
                self.multilabel = True
                self.n_classes = self.labels.size(1)
                self.classes = torch.arange(self.labels.size(1))
                self.class_counts = self.labels.sum(0)
            else:
                self.multilabel = False
                self.classes = self.labels.unique()
                self.n_classes = self.classes.size(0)

                self.class_counts = pd.Series(self.labels.numpy()).value_counts(sort=False)

        else:
            self.multilabel = False
            self.n_classes = None

        if hasattr(self, "class_counts"):
            self.class_weight = torch.sqrt(torch.true_divide(1, torch.tensor(self.class_counts, dtype=torch.float)))
            assert self.class_weight.numel() == self.n_classes, \
                f"self.class_weight {self.class_weight.numel()}, n_classes {self.n_classes}"

        assert hasattr(self, "num_nodes_dict")

        if not hasattr(self, "x_dict") or self.x_dict is None or len(self.x_dict) == 0:
            self.x_dict = {}

        if reshuffle_train is not None and reshuffle_train:
            self.resample_training_idx(reshuffle_train)
        else:
            if hasattr(self, "training_idx"):
                self.train_ratio = self.get_train_ratio()
                print("train_ratio", self.train_ratio)

    def is_undirected(self):
        if hasattr(self, "edge_index_dict") and self.edge_index_dict:
            for m, edge_index in self.edge_index_dict.items():
                e_index, e_values = get_edge_index_values(edge_index)
                print(m, "is_undirected:", is_undirected(edge_index=e_index))

    def name(self):
        if not hasattr(self, "_name"):
            return self.dataset.__class__.__name__
        else:
            return self._name

    @abstractmethod
    def compute_node_degrees(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def node_attr_shape(self):
        if not hasattr(self, "x_dict") or len(self.x_dict) == 0:
            node_attr_shape = {}
        else:
            node_attr_shape = {k: v.size(1) for k, v in self.x_dict.items()}
        return node_attr_shape

    @property
    def node_attr_size(self):
        node_feat_sizes = np.unique(list(self.node_attr_shape.values()))
        if len(node_feat_sizes) == 1:
            in_features = node_feat_sizes[0]
        elif len(node_feat_sizes) == 0:
            return None
        else:
            raise Exception(
                f"Must use self.node_attr_shape as node types have different feature sizes. {node_feat_sizes}")

        return in_features

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

    def resample_training_idx(self, train_ratio: float):
        if train_ratio == 1.0:
            ratio = self.get_train_ratio()
        else:
            ratio = train_ratio

            all_idx = torch.cat([self.training_idx, self.validation_idx, self.testing_idx])
            self.training_idx, self.validation_idx, self.testing_idx = \
                self.split_train_val_test(train_ratio=ratio, sample_indices=all_idx)
        print(f"Resampled training set at {self.get_train_ratio()}%")

    def get_metapaths(self, khop=False):
        """
        Returns original metapaths including reverse metapaths if use_reverse
        :return:
        """
        metapaths = self.metapaths
        if self.use_reverse and not any("rev_" in metapath[1] for metapath in self.metapaths):
            metapaths = metapaths + get_reverse_metapaths(self.metapaths)

        if khop:
            t_order_metapaths = metapaths
            for k in range(len(self.neighbor_sizes) - 1):
                t_order_metapaths = moge.module.PyG.utils.join_metapaths(t_order_metapaths, metapaths)
                metapaths = metapaths + t_order_metapaths

        return metapaths

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
    def get_unique_nodes(edge_index_dict, source=True, target=True):
        node_ids_dict = {}
        for metapath, edge_index in edge_index_dict.items():
            if source:
                node_ids_dict.setdefault(metapath[0], []).append(edge_index[0])
            if target:
                node_ids_dict.setdefault(metapath[-1], []).append(edge_index[1])

        for node_type in node_ids_dict:
            node_ids_dict[node_type] = torch.cat(node_ids_dict[node_type], 0).unique()

        return node_ids_dict

    def process_inmemorydataset(self, dataset: PyGInMemoryDataset, train_ratio):
        data = dataset[0]
        self.edge_index_dict = data.edge_index_dict
        self.num_nodes_dict = data.num_nodes_dict
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
        assert train_ratio is not None
        self.training_idx, self.validation_idx, self.testing_idx = \
            self.split_train_val_test(train_ratio,
                                      sample_indices=self.y_index_dict[self.head_node_type])

    def train_dataloader(self, collate_fn=None, batch_size=128, num_workers=0, **kwargs):
        loader = data.DataLoader(self.training_idx, batch_size=batch_size,
                                 shuffle=True, num_workers=num_workers,
                                 collate_fn=collate_fn if callable(collate_fn) \
                                     else self.get_collate_fn(collate_fn, mode="train"),
                                 **kwargs)
        return loader


    def valid_dataloader(self, collate_fn=None, batch_size=128, num_workers=0, **kwargs):
        loader = data.DataLoader(self.validation_idx, batch_size=batch_size,
                                 shuffle=False, num_workers=num_workers,
                                 collate_fn=collate_fn if callable(collate_fn) \
                                     else self.get_collate_fn(collate_fn, mode="validation", ),
                                 **kwargs)
        return loader

    def test_dataloader(self, collate_fn=None, batch_size=128, num_workers=0, **kwargs):
        loader = data.DataLoader(self.testing_idx, batch_size=batch_size,
                                 shuffle=False, num_workers=num_workers,
                                 collate_fn=collate_fn if callable(collate_fn) \
                                     else self.get_collate_fn(collate_fn, mode="testing", ),
                                 **kwargs)
        return loader

    def filter_edge_index(self, edge_index: Union[Tensor, Tuple[Tensor, Tensor]], allowed_nodes: np.ndarray) -> Tuple[
        Tensor, Tensor]:
        if isinstance(edge_index, tuple):
            edge_index = edge_index[0]
            values = edge_index[1]
        else:
            edge_index = edge_index
            values = torch.ones(edge_index.size(1))

        mask = np.isin(edge_index[0], allowed_nodes) & np.isin(edge_index[1], allowed_nodes)
        edge_index = edge_index[:, mask]
        values = values[mask]

        return (edge_index, values)

    def get_train_ratio(self):
        if self.validation_idx.size() != self.testing_idx.size() or not (self.validation_idx == self.testing_idx).all():
            train_ratio = self.training_idx.numel() / \
                          sum([self.training_idx.numel(), self.validation_idx.numel(), self.testing_idx.numel()])
        else:
            train_ratio = self.training_idx.numel() / sum([self.training_idx.numel(), self.validation_idx.numel()])
        return train_ratio


@DeprecationWarning
class CogDLDataset(HeteroGraphDataset):
    def process_COGDLdataset(self, dataset, metapath, node_types, train_ratio):
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


@DeprecationWarning
class HANDataset(HeteroGraphDataset):
    def get_collate_fn(self, collate_fn: str, mode=None, **kwargs):
        def collate_wrapper(iloc):
            if "HAN_batch" in collate_fn:
                return self.collate_HAN_batch(iloc, mode=mode)
            elif "HAN" in collate_fn:
                return self.collate_HAN(iloc, mode=mode)
            elif "collate_HGT_batch" in collate_fn:
                return self.collate_HGT_batch(iloc, mode=mode)
            else:
                raise Exception(f"Correct collate function {collate_fn} not found.")

        return collate_wrapper

    def collate_HAN(self, iloc, mode=None):
        if not isinstance(iloc, Tensor):
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

        X["global_node_index"] = torch.arange(X["x"].shape[0])

        y = self.y_dict[self.head_node_type][iloc]
        return X, y, None

    def collate_HAN_batch(self, iloc, mode=None):
        if not isinstance(iloc, Tensor):
            iloc = torch.tensor(iloc)

        X_batch, y, weights = self.sample(iloc, mode=mode)  # uses HeteroNetSampler PyG sampler method

        X = {}
        X["adj"] = [(X_batch["edge_index_dict"][metapath], torch.ones(X_batch["edge_index_dict"][metapath].size(1))) \
                    for metapath in self.metapaths if metapath in X_batch["edge_index_dict"]]
        X["x"] = self.data["x"][X_batch["global_node_index"][self.head_node_type]]
        X["idx"] = X_batch["global_node_index"][self.head_node_type]

        X["global_node_index"] = X_batch["global_node_index"]  # Debugging purposes

        return X, y, weights

    def collate_HGT_batch(self, iloc, mode=None):
        X_batch, y, weights = self.sample(iloc, mode=mode)  # uses HeteroNetSampler PyG sampler method

        X = {}
        X["node_inp"] = torch.vstack([X_batch["x_dict"][ntype] for ntype in self.node_types])
        X["node_type"] = torch.hstack([nid * torch.ones((X_batch["x_dict"][ntype].shape[0],), dtype=int) \
                                       for nid, ntype in enumerate(self.node_types)])
        # assert X["node_inp"].shape[0] == X["node_type"].shape[0]

        X["edge_index"] = torch.hstack([X_batch["edge_index_dict"][metapath] \
                                        for metapath in self.metapaths if metapath in X_batch["edge_index_dict"]])
        X["edge_type"] = torch.hstack([eid * torch.ones(X_batch["edge_index_dict"][metapath].shape[1], dtype=int) \
                                       for eid, metapath in enumerate(self.metapaths) if
                                       metapath in X_batch["edge_index_dict"]])
        # assert X["edge_index"].shape[1] == X["edge_type"].shape[0]

        X["global_node_index"] = X_batch["global_node_index"]  # Debugging purposes
        X["edge_time"] = None

        return X, y, weights
