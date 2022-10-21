import os
import pickle
from argparse import Namespace
from os.path import join
from pathlib import Path
from pprint import pprint
from typing import List, Tuple, Union, Dict, Optional, Callable

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from pandas import DataFrame, Series, Index
from torch import Tensor
from torch.utils.data import DataLoader
from torch_geometric.data import HeteroData
from torch_sparse.tensor import SparseTensor

from moge.dataset.PyG.neighbor_sampler import NeighborLoaderX, HGTLoaderX
from moge.dataset.graph import HeteroGraphDataset
from moge.dataset.io import get_attrs
from moge.dataset.sequences import SequenceTokenizers
from moge.dataset.utils import edge_index_to_adjs, gather_node_dict, get_relabled_edge_index, is_negative
from moge.model.PyG.utils import num_edges, convert_to_nx_edgelist
from moge.model.utils import to_device, tensor_sizes
from moge.network.hetero import HeteroNetwork


def reverse_metapath_name(metapath: Tuple[str, str, str]) -> Tuple[str, str, str]:
    rev_metapath = tuple(reversed(["rev_" + type if i % 2 == 1 else type for i, type in enumerate(metapath)]))
    return rev_metapath


class HeteroNodeClfDataset(HeteroGraphDataset):
    nodes: Dict[str, Index]
    nodes_namespace: Dict[str, Series]

    def __init__(
            self,
            dataset: HeteroData,
            seq_tokenizer: SequenceTokenizers = None,
            neighbor_loader: str = "NeighborLoader",
            neighbor_sizes: Union[List[int], Dict[str, List[int]]] = [128, 128],
            node_types: List[str] = None,
            metapaths: List[Tuple[str, str, str]] = None,
            head_node_type: str = None,
            pred_ntypes: List[str] = None,
            edge_dir: str = "in",
            add_reverse_metapaths: bool = False,
            inductive: bool = False,
            **kwargs,
    ):
        super().__init__(
            dataset,
            node_types=node_types,
            metapaths=metapaths,
            head_node_type=head_node_type,
            pred_ntypes=pred_ntypes,
            edge_dir=edge_dir,
            add_reverse_metapaths=add_reverse_metapaths,
            inductive=inductive,
            **kwargs,
        )
        if seq_tokenizer:
            self.seq_tokenizer = seq_tokenizer

        self.neighbor_loader = neighbor_loader
        self.neighbor_sizes = neighbor_sizes

    def process_pyg_heterodata(self, hetero: HeteroData):
        self.x_dict = hetero.x_dict
        self.node_types = hetero.node_types
        self.num_nodes_dict = {ntype: hetero[ntype].num_nodes for ntype in hetero.node_types}
        self.global_node_index = {
            ntype: torch.arange(num_nodes) for ntype, num_nodes in self.num_nodes_dict.items() if num_nodes
        }

        self.y_dict = {ntype: hetero[ntype].y for ntype in hetero.node_types if hasattr(hetero[ntype], "y")}

        # Add reverse metapaths to allow reverse message passing for directed edges
        if self.use_reverse and not any("rev_" in metapath[1] for metapath in hetero.edge_index_dict):
            transform = T.ToUndirected(merge=False)
            hetero: HeteroData = transform(hetero)

        self.metapaths = hetero.edge_types
        self.edge_index_dict = hetero.edge_index_dict

        self.G = hetero

    @classmethod
    def from_heteronetwork(
            cls,
            network: HeteroNetwork,
            node_attr_cols: List[str] = None,
            target: str = None,
            min_count: int = None,
            expression=False,
            sequence=False,
            labels_subset: Optional[Union[Index, np.ndarray]] = None,
            head_node_type: Optional[str] = None,
            ntype_subset: Optional[List[str]] = None,
            add_reverse_metapaths=True,
            split_namespace=False,
            go_ntype=None,
            exclude_etypes: List[Union[str, Tuple]] = None,
            pred_ntypes: List[str] = None,
            train_test_split="node_mask",
            **kwargs,
    ):
        hetero, classes, nodes, training_idx, validation_idx, testing_idx = network.to_pyg_heterodata(
            node_attr_cols=node_attr_cols,
            target=target,
            min_count=min_count,
            labels_subset=labels_subset,
            head_node_type=head_node_type,
            ntype_subset=ntype_subset,
            exclude_etypes=exclude_etypes,
            sequence=sequence,
            expression=expression,
            train_test_split=train_test_split,
            **kwargs,
        )

        self = cls(
            hetero,
            metapaths=hetero.edge_types,
            add_reverse_metapaths=add_reverse_metapaths,
            edge_dir="in",
            head_node_type=head_node_type,
            pred_ntypes=pred_ntypes,
            **kwargs,
        )
        self.classes = classes
        self.nodes = {ntype: nids for ntype, nids in nodes.items() if ntype in self.node_types}
        self._name = network._name if hasattr(network, "_name") else ""
        self.network = network

        self.pred_metapaths = network.pred_metapaths if hasattr(network, "pred_metapaths") else []
        self.neg_pred_metapaths = network.neg_pred_metapaths if hasattr(network, "neg_pred_metapaths") else []

        self.go_ntype = go_ntype
        self.split_namespace = split_namespace
        if split_namespace:
            assert self.go_ntype is not None
            self.nodes_namespace = {}
            self.ntype_mapping = {}
            for ntype, df in network.annotations.items():
                if "namespace" in df.columns:
                    node2namespace = network.annotations[ntype]["namespace"]

                    if ntype in self.pred_ntypes:
    node2namespace = node2namespace.replace(
        {"biological_process": "BPO", "cellular_component": "CCO", "molecular_function": "MFO"}
    )

                    self.nodes_namespace[ntype] = node2namespace.loc[~node2namespace.index.duplicated()]
                    self.ntype_mapping.update({namespace: ntype for namespace in np.unique(self.nodes_namespace[ntype])})return self

    @classmethod
    def load(cls, path: Path, **kwargs):
        if isinstance(path, str) and "~" in path:
    path = os.path.expanduser(path)

        hetero: HeteroData = torch.load(join(path, "heterodata.pt"))

        with open(join(path, "attrs.pickle"), "rb") as f:
    attrs: Dict = pickle.load(f)
        if kwargs:
            attrs.update(kwargs)

        self = cls(hetero, **attrs)

        self._name = os.path.basename(path)

        self.network = Namespace()
        self.network.annotations = {}
        for ntype in hetero.node_types:
    if os.path.exists(join(path, f"{ntype}.pickle")):
        self.network.annotations[ntype] = pd.read_pickle(join(path, f"{ntype}.pickle"))
    elif os.path.exists(join(path, f"{ntype}.parquet")):
        self.network.annotations[ntype] = pd.read_parquet(join(path, f"{ntype}.parquet"))

        nodes = pd.read_pickle(join(path, "nodes.pickle"))
        self.nodes = {ntype: nids for ntype, nids in nodes.items() if ntype in self.node_types}
        return self

    def save(self, path):
        if isinstance(path, str) and "~" in path:
    path = os.path.expanduser(path)

        if not os.path.exists(path):
            os.makedirs(path)

        if isinstance(self.network.nodes, (pd.Index, pd.Series)):
            self.network.nodes.to_pickle(join(path, "nodes.pickle"))
        elif isinstance(self.network.nodes, dict):
            with open(join(path, "nodes.pickle"), "wb") as f:
                pickle.dump(self.network.nodes, f)

        for ntype, df in self.network.annotations.items():
            df.to_pickle(join(path, f"{ntype}.pickle"))

        torch.save(self.G, join(path, "heterodata.pt"))

        attrs = get_attrs(
            self,
            exclude={
                "x_dict",
                "y_dict",
                "edge_index_dict",
                "global_node_index",
                "nodes",
                "node_attr_shape",
                "node_attr_sparse",
                "num_nodes_dict",
            },
        )
        with open(join(path, "attrs.pickle"), "wb") as f:
            pickle.dump(attrs, f)

    @property
    def class_indices(self) -> Optional[Dict[str, Tensor]]:
        if self.pred_ntypes is None or set(self.pred_ntypes).difference(self.nodes.keys()) or self.classes is None:
            return None
        class_indices = {}
        for pred_ntype in self.pred_ntypes:
            class_indices[pred_ntype] = torch.from_numpy(self.nodes[pred_ntype].get_indexer_for(self.classes))
            class_indices[pred_ntype] = class_indices[pred_ntype][class_indices[pred_ntype] >= 0]
        return class_indices

    @property
    def node_mask_counts(self) -> DataFrame:
        ntypes = self.G.node_types
        return pd.DataFrame(
            tensor_sizes(
                dict(
                    train={
                        ntype: self.G[ntype].train_mask.sum()
                        for ntype in ntypes
                        if hasattr(self.G[ntype], "train_mask")
                    },
                    valid={
                        ntype: self.G[ntype].valid_mask.sum()
                        for ntype in ntypes
                        if hasattr(self.G[ntype], "valid_mask")
                    },
                    test={
                        ntype: self.G[ntype].test_mask.sum() for ntype in ntypes if hasattr(self.G[ntype], "test_mask")
                    },
                )
            )
        )

    @property
    def edge_mask_counts(self) -> DataFrame:
        etypes = self.G.edge_types
        df = pd.DataFrame(
            tensor_sizes(
                dict(
                    train={
                        etype: self.G[etype].train_mask.sum()
                        for etype in etypes
                        if hasattr(self.G[etype], "train_mask")
                    },
                    valid={
                        etype: self.G[etype].valid_mask.sum()
                        for etype in etypes
                        if hasattr(self.G[etype], "valid_mask")
                    },
                    test={
                        etype: self.G[etype].test_mask.sum() for etype in etypes if hasattr(self.G[etype], "test_mask")
                    },
                )
            )
        )
        if df.empty:
            return None
        df.index.names = ["src_ntype", "etype", "dst_ntype"]
        return df.sort_index()

    def create_graph_sampler(
            self,
            graph: HeteroData,
            batch_size: int,
            node_type: str,
            node_mask: Tensor,
            transform_fn: Callable = None,
            num_workers=10,
            verbose=False,
            shuffle=True,
            **kwargs,
    ):
        min_expansion_size = min(self.neighbor_sizes)
        max_expansion_size = 100

        if self.neighbor_loader == "NeighborLoader":
            self.num_neighbors = {metapath: self.neighbor_sizes for metapath in self.metapaths}

        elif self.neighbor_loader == "HGTLoader":
            self.num_neighbors = {ntype: self.neighbor_sizes for ntype in self.node_types}

        print(f"{self.neighbor_loader} neighbor_sizes:") if verbose else None
        pprint(self.num_neighbors, width=300) if verbose else None

        args = dict(
            data=graph,
            num_neighbors=self.num_neighbors,
            batch_size=batch_size,
            transform=transform_fn,
            input_nodes=(node_type, node_mask),
            shuffle=shuffle,
            num_workers=num_workers,
            **kwargs,
        )

        if self.class_indices is not None and set(self.class_indices.keys()).intersection(self.node_types):
            args["class_indices"] = self.class_indices

        if self.neighbor_loader == "NeighborLoader":
            dataset = NeighborLoaderX(**args)
        elif self.neighbor_loader == "HGTLoader":
            dataset = HGTLoaderX(**args)

        return dataset

    def transform(self, hetero: HeteroData):
        X = {}
        X["edge_index_dict"] = {metapath: edge_index for metapath, edge_index in hetero.edge_index_dict.items()}
        X["global_node_index"] = {ntype: nid for ntype, nid in hetero.nid_dict.items() if nid.numel()}
        X["sizes"] = {ntype: size for ntype, size in hetero.num_nodes_dict.items() if size}
        X["batch_size"] = hetero.batch_size_dict

        # Node featuers
        X["x_dict"] = {ntype: x for ntype, x in hetero.x_dict.items() if x.numel()}
        for ntype, feat in X["x_dict"].items():
            nids = X["global_node_index"][ntype]
            if isinstance(feat, SparseTensor):
                X["x_dict"][ntype] = feat[nids]
            elif isinstance(feat, pd.Series):
                X["x_dict"][ntype] = feat.iloc[nids.numpy()]

        if hasattr(hetero, "sequence_dict") and hasattr(self, "seq_tokenizer"):
            X["sequences"] = {}
            for ntype in X["global_node_index"]:
    if not hasattr(hetero[ntype], "sequence") or ntype not in self.seq_tokenizer.tokenizers:
        continue
    X["sequences"][ntype] = self.seq_tokenizer.encode_sequences(hetero, ntype=ntype, max_length=None)

        # Node labels
        y_dict = {}
        for ntype, y in hetero.y_dict.items():
    if y.size(0) == 0 or ntype not in X["global_node_index"]:
        continue
    nids = X["global_node_index"][ntype]
    if isinstance(y, SparseTensor):
        if y.density():
            y_dict[ntype] = y[nids].to_dense()
    elif isinstance(y, Tensor):
        y_dict[ntype] = y

        if len(y_dict) == 1:
            y = y_dict[list(y_dict.keys()).pop()]
            if y.dim() == 2 and y.size(1) == 1:
                y = y.squeeze(-1)

            if y.dim() == 1:
                weights = (y >= 0).to(torch.float)
            elif y.dim() == 2:
                weights = (y.sum(1) > 0).to(torch.float)
            else:
                weights = None

        elif len(y_dict) > 1:
            weights = {}
            for ntype, y in y_dict.items():
                if y.dim() == 2 and y.size(1) == 1:
                    y_dict[ntype] = y.squeeze(-1)

                if y.dim() == 1:
                    weights[ntype] = (y >= 0).to(torch.float)
                elif y.dim() == 2:
                    weights[ntype] = (y.sum(1) > 0).to(torch.float)
                else:
                    weights = None
        else:
            weights = None

        assert weights is None or isinstance(weights, dict) or (isinstance(weights, Tensor) and weights.dim() == 1)

        return X, y_dict, weights

    def full_batch(self):
        return self.transform(self.G)

    def get_node_metadata(
            self,
            global_node_index: Dict[str, Tensor],
            embeddings: Dict[str, Tensor],
            weights: Optional[Dict[str, Series]] = None,
            losses: Dict[str, Tensor] = None,
    ) -> DataFrame:
        """
        Collect node metadata for all nodes in X["global_node_index"]
        Args:
            batch (): a batch's dict of data
            embeddings (): Embeddings of nodes in the `X` batch
            weights (Optional[Dict[str, Series]]): A Dict of ntype and a Pandas Series same same as number of nodes
                where entries > 0 are returned.

        Returns:
            node_metadata (DataFrame)
        """
        # if not hasattr(self, "node_metadata"):
        self.create_node_metadata(self.network.annotations, nodes=self.nodes)

        global_node_index = {
            ntype: nids.numpy() if isinstance(nids, Tensor) else nids
            for ntype, nids in global_node_index.items()
            if ntype in embeddings
        }

        # Concatenated list of node embeddings and other metadata
        nodes_emb = {
            ntype: embeddings[ntype].detach().numpy() if isinstance(embeddings[ntype], Tensor) else embeddings[ntype]
            for ntype in embeddings
        }
        nodes_emb = np.concatenate([nodes_emb[ntype] for ntype in global_node_index])

        node_train_valid_test = np.vstack(
            [
                np.concatenate([self.G[ntype].train_mask[nids].numpy() for ntype, nids in global_node_index.items()]),
                np.concatenate([self.G[ntype].valid_mask[nids].numpy() for ntype, nids in global_node_index.items()]),
                np.concatenate([self.G[ntype].test_mask[nids].numpy() for ntype, nids in global_node_index.items()]),
            ]
        ).T
        node_train_valid_test = np.array(["Train", "Valid", "Test"])[node_train_valid_test.argmax(1)]

        if losses:
            node_losses = np.concatenate(
                [
                    losses[ntype] if ntype in losses else [None for i in range(global_node_index[ntype].size)]
                    for ntype in global_node_index
                ]
            )
        else:
            node_losses = None

        # Metadata
        # Build node metadata dataframe from concatenated lists of node metadata for multiple ntypes
        df = pd.DataFrame(
            {
                "node": np.concatenate([self.nodes[ntype][global_node_index[ntype]] for ntype in global_node_index]),
                "ntype": np.concatenate(
                    [[ntype for i in range(global_node_index[ntype].shape[0])] for ntype in global_node_index]
                ),
                "train_valid_test": node_train_valid_test,
                "loss": node_losses,
            },
            index=pd.Index(np.concatenate([global_node_index[ntype] for ntype in global_node_index]), name="nid"),
        )

        # Get TSNE 2d position from embeddings
        try:
            from umap import UMAP

            tsne = UMAP(n_components=2, n_jobs=-1)
        except Exception as e:
            import MulticoreTSNE

            tsne = MulticoreTSNE.MulticoreTSNE(n_components=2, perplexity=15, learning_rate=10, n_jobs=-1)

        nodes_pos = tsne.fit_transform(nodes_emb)
        nodes_pos = {node_name: pos for node_name, pos in zip(df.index, nodes_pos)}
        df[["pos1", "pos2"]] = np.vstack(df.index.map(nodes_pos))
        df = df.assign(loss=df["loss"].astype(float), pos1=df["pos1"].astype(float), pos2=df["pos2"].astype(float))

        # Set index
        df = df.reset_index()
        df["nx_node"] = df["ntype"] + "-" + df["node"]
        df = df.set_index(["ntype", "nid"])

        # Update all nodes embeddings with self.node_metadata
        for col in set(self.node_metadata.columns) - set(df.columns):
            df[col] = None
        df.update(self.node_metadata, overwrite=False)
        df.dropna(axis=1, how="all", inplace=True)

        # Update self.node_metadata with df
        for col in set(df.columns) - set(self.node_metadata.columns):
            self.node_metadata[col] = None
        self.node_metadata.update(df[~df.index.duplicated()])

        # return only nodes that have > 0 weights (used for visualization of node clf models)
        if weights is not None:
            nodes_weight = {
                ntype: weights[ntype].detach().numpy() if isinstance(weights[ntype], Tensor) else weights[ntype]
                for ntype in weights
            }
            nodes_weight = np.concatenate([nodes_weight[ntype] for ntype in global_node_index]).astype(bool)

            return df.loc[nodes_weight]

        return df

    def to_networkx(
            self,
            nodes: Dict[str, Union[List[str], List[int]]] = None,
            edge_index_dict: Dict[Tuple[str, str, str], Tensor] = [],
            global_node_idx: Dict[str, Tensor] = None,
    ) -> nx.MultiDiGraph:
        G = nx.MultiDiGraph()

        if edge_index_dict is None or len(edge_index_dict) == 0:
            edge_index_dict = self.G.edge_index_dict

        elif isinstance(edge_index_dict, list):
            # edge_index_dict is a list of metapaths
            edge_index_dict = {
                m: eidx for m, eidx in self.G.edge_index_dict.items() if m in edge_index_dict or m[1] in edge_index_dict
            }

        edge_index_dict = {m: eid for m, eid in edge_index_dict.items() if "rev_" not in m[1]}

        edge_list = convert_to_nx_edgelist(edge_index_dict, node_names=self.nodes, global_node_idx=global_node_idx)
        for etype, edges in edge_list.items():
            G.add_edges_from(edges, etype=etype)

        if nodes:
            filter_nodes = []

            for ntype, node_list in nodes.items():
                if isinstance(node_list, Tensor):
                    node_list = node_list.detach().numpy().tolist()

                if all(isinstance(node, str) for node in node_list):
                    select_nodes = ntype + "-" + pd.Index(node_list)
                elif all(isinstance(node, int) for node in node_list):
                    select_nodes = ntype + "-" + self.nodes[ntype][nodes[ntype]]
                else:
                    print([type(node) for node in node_list])
                    select_nodes = []

                if G.number_of_nodes() > 0 and set(select_nodes).issubset(set(G.nodes())):
                    filter_nodes.extend(select_nodes)
                else:
                    G.add_nodes_from(select_nodes)

            if len(filter_nodes):
                H = G.copy()
                H = G.subgraph(filter_nodes)
            else:
                H = G
        else:
            H = G.copy()
            H.remove_nodes_from(list(nx.isolates(H)))

        return H

    def train_dataloader(self, collate_fn=None, batch_size=128, num_workers=0, **kwargs):
        graph = self.G
        node_mask = graph[self.head_node_type].train_mask & graph[self.head_node_type].y.sum(1).type(torch.bool)

        dataset = self.create_graph_sampler(
            graph,
            batch_size,
            node_type=self.head_node_type,
            node_mask=node_mask,
            transform_fn=self.transform,
            num_workers=num_workers,
            shuffle=True,
        )

        return dataset

    def valid_dataloader(self, collate_fn=None, batch_size=128, num_workers=0, **kwargs):
        graph = self.G
        node_mask = graph[self.head_node_type].valid_mask & graph[self.head_node_type].y.sum(1).type(torch.bool)

        dataset = self.create_graph_sampler(
            self.G,
            batch_size,
            node_type=self.head_node_type,
            node_mask=node_mask,
            transform_fn=self.transform,
            num_workers=num_workers,
            shuffle=False,
        )

        return dataset

    def test_dataloader(self, collate_fn=None, batch_size=128, num_workers=0, **kwargs):
        graph = self.G
        node_mask = graph[self.head_node_type].test_mask & graph[self.head_node_type].y.sum(1).type(torch.bool)

        dataset = self.create_graph_sampler(
            self.G,
            batch_size,
            node_type=self.head_node_type,
            node_mask=node_mask,
            transform_fn=self.transform,
            num_workers=num_workers,
            shuffle=False,
        )

        return dataset


class HeteroLinkPredDataset(HeteroNodeClfDataset):
    def __init__(
        self,
        dataset: HeteroData,
        negative_sampling_size=1000,
        seq_tokenizer: SequenceTokenizers = None,
        neighbor_loader: str = "NeighborLoader",
        neighbor_sizes: Union[List[int], Dict[str, List[int]]] = [128, 128],
        node_types: List[str] = None,
        metapaths: List[Tuple[str, str, str]] = None,
        head_node_type: str = None,
        edge_dir: str = "in",
        add_reverse_metapaths: bool = True,
        inductive: bool = False,
        train_test_split="edge_mask",
        **kwargs,
    ):
        super().__init__(
            dataset,
            seq_tokenizer=seq_tokenizer,
            neighbor_loader=neighbor_loader,
            neighbor_sizes=neighbor_sizes,
            node_types=node_types,
            metapaths=metapaths,
            head_node_type=head_node_type,
            edge_dir=edge_dir,
            add_reverse_metapaths=add_reverse_metapaths,
            inductive=inductive,
            train_test_split=train_test_split,
            **kwargs,
        )
        self.negative_sampling_size = negative_sampling_size
        self.eval_negative_sampling_size = 1000

        self.neighbor_sizes = neighbor_sizes
        self.multilabel = False

        self.graph_sampler = self.create_graph_sampler(
            self.G,
            batch_size=1,
            node_type=head_node_type,
            node_mask=torch.ones(self.G[self.head_node_type].num_nodes),
            transform_fn=super().transform,
            num_workers=0,
        )

    @classmethod
    def from_heteronetwork(
            cls,
            network: HeteroNetwork,
            node_attr_cols: List[str] = None,
            target: str = None,
            min_count: int = None,
            expression=False,
            sequence=False,
            label_subset: Optional[Union[Index, np.ndarray]] = None,
            head_node_type: Optional[str] = None,
            ntype_subset: Optional[List[str]] = None,
            exclude_metapaths=None,
            add_reverse_metapaths=True,
            split_namespace=False,
            **kwargs,
    ):
        self = super().from_heteronetwork(
            network,
            node_attr_cols=node_attr_cols,
            target=target,
            min_count=min_count,
            expression=expression,
            sequence=sequence,
            label_subset=label_subset,
            head_node_type=head_node_type,
            ntype_subset=ntype_subset,
            exclude_etypes=network.pred_metapaths,
            add_reverse_metapaths=add_reverse_metapaths,
            split_namespace=split_namespace,
            **kwargs,
        )

        # Train/valid/test positive annotations
        self.triples, self.training_idx, self.validation_idx, self.testing_idx = network.get_triples(
            self.pred_metapaths, positive=True
        )

        # Train/valid/test negative annotations
        self.triples_neg, self.training_idx_neg, self.validation_idx_neg, self.testing_idx_neg = network.get_triples(
            self.neg_pred_metapaths, negative=True
        )

        self.triples.update(self.triples_neg)

        # Adjacency of pos edges (for neg sampling)
        edge_index_dict, edge_neg_dict = get_relabled_edge_index(
            triples=self.triples,
            global_node_index={ntype: torch.arange(len(nodelist)) for ntype, nodelist in self.nodes.items()},
            metapaths=self.pred_metapaths,
            relation_ids_all=self.triples["relation"].unique(),
        )

        self.triples_adj: Dict[Tuple[str, str, str], SparseTensor] = edge_index_to_adjs(
            edge_index_dict, nodes=self.nodes
        )

        return self

    def get_prior(self) -> Tensor:
        return torch.tensor(1) / (1 + self.negative_sampling_size)

    @staticmethod
    def relabel_edge_index_dict(
            edge_index_dict: Dict[Tuple[str, str, str], Tensor],
            global_node_index: Dict[str, Tensor],
            global2batch: Dict[str, Tensor] = None,
            mode=None,
    ) -> Dict[Tuple[str, str, str], Tensor]:
        if global2batch is None:
            global2batch = {
                node_type: dict(zip(global_node_index[node_type].numpy(), range(len(global_node_index[node_type]))))
                for node_type in global_node_index
            }

        renamed_edge_index_dict = {}
        for metapath, edge_index in edge_index_dict.items():
            head_type, edge_type, tail_type = metapath

            if edge_index.size(0) == 2 and mode is None:
                sources = edge_index[0].apply_(global2batch[head_type].get)
                targets = edge_index[-1].apply_(global2batch[tail_type].get)

                renamed_edge_index_dict[metapath] = torch.stack([sources, targets], dim=1).t()

            elif mode == "head_batch":
                renamed_edge_index_dict[metapath] = edge_index.apply_(global2batch[head_type].get)
            elif mode == "tail_batch":
                renamed_edge_index_dict[metapath] = edge_index.apply_(global2batch[tail_type].get)
            else:
                raise Exception(
                    "Must provide an edge_index with shape (2, N) or pass 'head_batch' or 'tail_batch' " "in `mode`."
                )

        return renamed_edge_index_dict

    def transform(self, edge_idx: List[int], mode=None) -> Tuple[Dict[str, Dict], Dict[str, Dict], Optional[Tensor]]:
        if not isinstance(edge_idx, torch.LongTensor):
            edge_idx = torch.LongTensor(edge_idx)

        # Get train/valid/test indices
        if edge_idx[0] in self.training_idx:
            mode = "train" if mode is None else mode
            max_negative_sampling_size = self.negative_sampling_size
            all_neg_edge_idx = self.training_idx_neg
        elif edge_idx[0] in self.validation_idx:
            mode = "valid" if mode is None else mode
            max_negative_sampling_size = self.eval_negative_sampling_size
            all_neg_edge_idx = self.validation_idx_neg
        elif edge_idx[0] in self.testing_idx:
            mode = "test" if mode is None else mode
            max_negative_sampling_size = self.eval_negative_sampling_size
            all_neg_edge_idx = self.testing_idx_neg

        # True pos edges
        edge_pos, edge_neg = self.get_edge_index_dict_from_triples(
            edge_idx, neg_edge_idx=all_neg_edge_idx, metapaths=self.pred_metapaths
        )

        # Get all nodes induced by sampled edges
        if num_edges(edge_neg):
            query_edges = {
                metapath: torch.cat(
                    [
                        edge_pos[metapath] if metapath in edge_pos else torch.tensor([], dtype=torch.long),
                        edge_neg[metapath] if metapath in edge_neg else torch.tensor([], dtype=torch.long),
                    ],
                    dim=1,
                )
                for metapath in set(edge_pos).union(set(edge_neg))
            }
        else:
            query_edges = edge_pos

        query_nodes = gather_node_dict(query_edges)

        # Add random GO term nodes for negative sampling
        go_nodes_proba = 1 - F.one_hot(query_nodes[self.go_ntype], num_classes=self.num_nodes_dict[self.go_ntype]).sum(
            axis=0
        ).to(torch.float)
        go_nids = torch.multinomial(go_nodes_proba, num_samples=self.negative_sampling_size, replacement=False)
        query_nodes[self.go_ntype] = torch.cat([query_nodes[self.go_ntype], go_nids])

        # Get subgraph induced by neighborhood hopping from the query nodes
        X, _, _ = self.graph_sampler.transform_fn(self.graph_sampler.collate_fn(query_nodes))
        # print(tensor_sizes(edge_pos))
        X["batch_size"] = {ntype: query_nodes[ntype].numel() for ntype in query_nodes}

        # Edge_pos must be global index, not batch index
        if self.split_namespace:
            edge_pos = self.split_edge_index_by_nodes_namespace(edge_pos, batch_to_global=None)
        head_batch, tail_batch = self.generate_negative_sampling(
            edge_pos,
            global_node_index=X["global_node_index"],
            max_negative_sampling_size=max_negative_sampling_size,
            mode=mode,
        )

        # Rename node index from global to batch
        global2batch = {
            ntype: dict(zip(X["global_node_index"][ntype].numpy(), range(len(X["global_node_index"][ntype]))))
            for ntype in X["global_node_index"]
        }
        if self.split_namespace:
            for namespace, go_ntype in self.ntype_mapping.items():
                global2batch[namespace] = global2batch[go_ntype]

        edge_true = {}
        edge_pos = self.relabel_edge_index_dict(
            edge_pos, global_node_index=X["global_node_index"], global2batch=global2batch
        )
        if self.split_namespace:
            edge_true["edge_pos"] = self.split_edge_index_by_nodes_namespace(
                edge_pos, batch_to_global=X["global_node_index"]
            )
        else:
            edge_true["edge_pos"] = edge_pos

        if num_edges(edge_neg):
            edge_neg = self.relabel_edge_index_dict(
                edge_neg, global_node_index=X["global_node_index"], global2batch=global2batch
            )
            if self.split_namespace:
                edge_true["edge_neg"] = self.split_edge_index_by_nodes_namespace(
                    edge_neg, batch_to_global=X["global_node_index"]
                )
            else:
                edge_true["edge_neg"] = edge_neg

        # Negative sampling
        edge_true.update({"head_batch": head_batch, "tail_batch": tail_batch})

        edge_weights = None
        return X, edge_true, edge_weights

    # def query(self):

    def get_edge_index_dict_from_triples(self, edge_idx: Tensor, neg_edge_idx: Tensor = None, metapaths=None):
        triples = {k: v[edge_idx] for k, v in self.triples.items() if not is_negative(k)}

        # If ensures same number of true neg edges to true pos edges
        if neg_edge_idx is not None:
            neg_edge_idx = np.random.choice(
                neg_edge_idx, size=min(edge_idx.numel(), neg_edge_idx.numel()), replace=False
            )
            triples_neg = {k: v[neg_edge_idx] for k, v in self.triples.items() if is_negative(k)}
            triples.update(triples_neg)

        # Get edge_index_dict from triplets
        if metapaths is None:
            metapaths = self.metapaths
        edge_pos, edge_neg = get_relabled_edge_index(
            triples=triples, global_node_index=self.global_node_index, metapaths=metapaths
        )
        return edge_pos, edge_neg

    def generate_negative_sampling(
            self,
            edge_pos: Dict[Tuple[str, str, str], Tensor],
            global_node_index: Dict[str, Tensor],
            max_negative_sampling_size: int,
            mode: str = None,
    ) -> Tuple[Dict[Tuple[str, str, str], Tensor], Dict[Tuple[str, str, str], Tensor]]:
        """

        Args:
            edge_pos (): Edge index dict in global index
            global_node_index ():
            max_negative_sampling_size ():
            mode ():

        Returns:

        """
        head_batch = {}
        tail_batch = {}

        sampling_size = self.get_neg_sampling_size(
            edge_pos, global_node_index=global_node_index, max_samples=max_negative_sampling_size, mode=mode
        )

        # Perform negative sampling
        for metapath, edge_index in edge_pos.items():
            head_type, tail_type = metapath[0], metapath[-1]
            adj: SparseTensor = (
                self.triples_adj[metapath]
                if metapath in self.triples_adj
                else self.triples_adj[metapath[:-1] + (self.go_ntype,)]
            )

            # head noise distribution
            head_neg_nodes = global_node_index[head_type]
            head_prob_dist = 1 - adj[head_neg_nodes, edge_index[1]].to_dense().T

            head_batch[metapath] = torch.multinomial(head_prob_dist, num_samples=sampling_size, replacement=True)

            # Tail noise distribution
            tail_neg_nodes = global_node_index[tail_type if tail_type in global_node_index else self.go_ntype]
            tail_prob_dist = 1 - adj[edge_index[0], tail_neg_nodes].to_dense()

            # Only generate negative tail_batch within BPO, CCO, or MFO terms of the positive edge's tail go_type
            # for go_type in ['biological_process', 'cellular_component', 'molecular_function']:
            #     if go_type != tail_type: continue
            #
            #     go_terms_mask = self.nodes_namespace[tail_neg_nodes] != go_type
            #     tail_prob_dist[:, go_terms_mask] = 0

            tail_batch[metapath] = torch.multinomial(tail_prob_dist, num_samples=sampling_size, replacement=True)

        return head_batch, tail_batch

    def get_neg_sampling_size(
            self,
            edge_index_dict: Dict[Tuple[str, str, str], Tensor],
            global_node_index: Dict[str, Tensor],
            max_samples: int,
            mode: str,
    ) -> int:
        # Choose neg sampling size by twice (minimum) the count of head or tail nodes, if it's less than self.negative_sampling_size
        if mode == "train":
            head_tail_ntypes = [ntype for metapath in edge_index_dict for ntype in [metapath[0], metapath[-1]]]
            min_num_node = min(
                global_node_index[ntype if ntype in global_node_index else self.ntype_mapping[ntype]].size(0)
                for ntype in head_tail_ntypes
            )

            sampling_size = min(min_num_node, max_samples // 2)

        else:
            sampling_size = max_samples // 2

        sampling_size = int(max(sampling_size, 1))

        return sampling_size

    def split_edge_index_by_nodes_namespace(
            self, edge_index_dict: Dict[Tuple[str, str, str], Tensor], batch_to_global: Dict[str, Tensor] = None
    ) -> Dict[Tuple[str, str, str], Tensor]:
        if not hasattr(self, "nodes_namespace"):
            return edge_index_dict

        out_edge_index_dict = {}

        for metapath, edge_index in edge_index_dict.items():
            tail_type = metapath[-1] if metapath[-1] in self.node_types else self.ntype_mapping[metapath[-1]]

            # Pos or neg edges
            if edge_index.size(0) == 2:
                if batch_to_global is not None:
                    go_terms = batch_to_global[tail_type][edge_index[1]]
                else:
                    go_terms = edge_index[1]

                namespaces = self.nodes_namespace[tail_type].iloc[go_terms]

                for namespace in np.unique(namespaces):
                    mask = namespaces == namespace
                    new_metapath = metapath[:-1] + (namespace,)

                    if not isinstance(mask, bool):
                        out_edge_index_dict[new_metapath] = edge_index[:, mask]
                    else:
                        out_edge_index_dict[new_metapath] = edge_index

        return out_edge_index_dict

    def full_batch(self, edge_idx: Tensor = None, mode="test", device="cpu"):
        if edge_idx is None:
            edge_idx = torch.cat([self.training_idx, self.validation_idx, self.testing_idx])
        elif not torch.is_tensor(edge_idx):
            edge_idx = torch.tensor(edge_idx)

        X, edge_pred, _ = self.transform(edge_idx=edge_idx, mode=mode)

        if device != "cpu":
            X = to_device(X, device)
            edge_pred = to_device(edge_pred, device)

        return X, edge_pred, _

    def to_networkx(
            self,
            nodes: Dict[str, Union[List[str], List[int]]] = None,
            edge_index_dict: Union[Dict[Tuple[str, str, str], Tensor], List[Tuple[str, str, str]]] = [],
            global_node_idx: Dict[str, Tensor] = None,
            pos_edges: Dict[Tuple[str, str, str], Tensor] = None,
    ) -> nx.MultiDiGraph:
        G = super().to_networkx(nodes, edge_index_dict, global_node_idx=global_node_idx)

        if pos_edges is not None:
            edge_list = convert_to_nx_edgelist(pos_edges, node_names=self.nodes, global_node_idx=global_node_idx)
            for etype, edges in edge_list.items():
                G.add_edges_from(edges, etype=etype)

        return G

    def train_dataloader(self, collate_fn=None, batch_size=128, num_workers=0, shuffle=True, **kwargs):
        dataset = DataLoader(
            self.training_idx,
            batch_size=batch_size,
            collate_fn=self.transform,
            shuffle=shuffle,
            num_workers=num_workers,
            **kwargs,
        )
        return dataset

    def valid_dataloader(self, collate_fn=None, batch_size=128, num_workers=0, shuffle=False, **kwargs):
    dataset = DataLoader(
        self.validation_idx,
        batch_size=batch_size,
        collate_fn=self.transform,
        shuffle=shuffle,
        num_workers=num_workers,
        **kwargs,
    )
    return dataset

    def test_dataloader(self, collate_fn=None, batch_size=128, num_workers=0, shuffle=False, **kwargs):
    dataset = DataLoader(
        self.testing_idx,
        batch_size=batch_size,
        collate_fn=self.transform,
        shuffle=shuffle,
        num_workers=num_workers,
        **kwargs,
    )
    return dataset

    def dataloader(self, edge_idx, batch_size=128, num_workers=0):
    dataset = DataLoader(
        edge_idx, batch_size=batch_size, collate_fn=self.transform, shuffle=False, num_workers=num_workers
    )
    return dataset
