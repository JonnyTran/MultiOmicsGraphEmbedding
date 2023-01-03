import itertools
import os
import pickle
from argparse import Namespace
from collections import OrderedDict
from os.path import join
from pathlib import Path
from pprint import pprint
from typing import List, Tuple, Union, Dict, Optional, Callable

import colorhash
import joblib
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
import tqdm
from logzero import logger
from pandas import DataFrame, Series, Index
from ruamel import yaml
from scipy.sparse import coo_matrix
from six.moves import intern
from sklearn.cluster import KMeans
from torch import Tensor
from torch.utils.data import DataLoader
from torch_geometric.data import HeteroData
from torch_geometric.utils import remove_self_loops, to_undirected
from torch_sparse.tensor import SparseTensor

from moge.dataset.PyG.neighbor_sampler import NeighborLoaderX, HGTLoaderX
from moge.dataset.PyG.utils import AddMetaPaths
from moge.dataset.graph import HeteroGraphDataset
from moge.dataset.io import get_attrs
from moge.dataset.sequences import SequenceTokenizers
from moge.dataset.utils import edge_index_to_adjs, gather_node_dict, \
    get_relabled_edge_index, is_negative
from moge.model.PyG.utils import num_edges, convert_to_nx_edgelist
from moge.model.utils import to_device, tensor_sizes
from moge.network.hetero import HeteroNetwork


def reverse_metapath_name(metapath: Tuple[str, str, str]) -> Tuple[str, str, str]:
    rev_metapath = tuple(reversed(["rev_" + type if i % 2 == 1 else type \
                                   for i, type in enumerate(metapath)]))
    return rev_metapath


class HeteroNodeClfDataset(HeteroGraphDataset):
    nodes: Dict[str, Index]
    nodes_namespace: Dict[str, Series]

    def __init__(self, dataset: HeteroData, seq_tokenizer: SequenceTokenizers = None,
                 neighbor_loader: str = "NeighborLoader",
                 neighbor_sizes: Union[List[int], Dict[str, List[int]]] = [128, 128],
                 node_types: List[str] = None,
                 metapaths: List[Tuple[str, str, str]] = None,
                 head_node_type: str = None,
                 pred_ntypes: List[str] = None,
                 edge_dir: str = "in",
                 add_reverse_metapaths: bool = False,
                 undirected_ntypes: List[str] = None,
                 inductive: bool = False, **kwargs):
        if seq_tokenizer:
            self.seq_tokenizer = seq_tokenizer
        self.undirected_ntypes = undirected_ntypes
        self.neighbor_loader = neighbor_loader
        self.neighbor_sizes = neighbor_sizes

        super().__init__(dataset, node_types=node_types, metapaths=metapaths, head_node_type=head_node_type,
                         pred_ntypes=pred_ntypes,
                         edge_dir=edge_dir, add_reverse_metapaths=add_reverse_metapaths, inductive=inductive, **kwargs)

    def process_pyg_heterodata(self, hetero: HeteroData):
        self.x_dict = hetero.x_dict
        self.node_types = hetero.node_types
        self.num_nodes_dict = {ntype: hetero[ntype].num_nodes \
                               for ntype in hetero.node_types}
        self.global_node_index = {ntype: torch.arange(num_nodes) \
                                  for ntype, num_nodes in self.num_nodes_dict.items() if num_nodes}

        self.y_dict = {ntype: hetero[ntype].y \
                       for ntype in hetero.node_types if hasattr(hetero[ntype], "y")}

        # Add reverse metapaths to allow reverse message passing for directed edges
        if self.use_reverse and not any('rev_' in metapath[1] for metapath in hetero.edge_index_dict):
            transform = T.ToUndirected(merge=False)
            hetero: HeteroData = transform(hetero)

        # Remove the reverse etype for undirected edge type
        if self.use_reverse and self.undirected_ntypes:
            for ntype in self.undirected_ntypes:
                undirected_metapaths = [metapath for metapath in hetero.edge_types \
                                        if metapath[0] == ntype == metapath[-1]]
                for metapath in undirected_metapaths:
                    reversed_metapath = reverse_metapath_name(metapath)
                    if reversed_metapath in hetero.edge_types:
                        del hetero[reversed_metapath]
                        if hasattr(hetero[metapath], 'edge_weight'):
                            hetero[metapath].edge_index, hetero[metapath].edge_weight = \
                                to_undirected(hetero[metapath].edge_index, hetero[metapath].edge_weight)
                        else:
                            hetero[metapath].edge_index = to_undirected(hetero[metapath].edge_index)

        self.metapaths = hetero.edge_types
        self.edge_index_dict = hetero.edge_index_dict

        self.G = hetero

    @classmethod
    def from_heteronetwork(cls, network: HeteroNetwork, node_attr_cols: List[str] = None,
                           target: str = None, min_count: int = None,
                           expression=False, sequence=False,
                           labels_subset: Optional[Union[Index, np.ndarray]] = None,
                           head_node_type: Optional[str] = None,
                           ntype_subset: Optional[List[str]] = None,
                           add_reverse_metapaths=True,
                           split_namespace=False,
                           exclude_etypes: List[Union[str, Tuple]] = None,
                           pred_ntypes: List[str] = None,
                           train_test_split="node_mask", **kwargs):
        hetero, classes, nodes, training_idx, validation_idx, testing_idx = \
            network.to_pyg_heterodata(node_attr_cols=node_attr_cols, target=target, min_count=min_count,
                                      labels_subset=labels_subset, head_node_type=head_node_type,
                                      ntype_subset=ntype_subset, exclude_etypes=exclude_etypes, sequence=sequence,
                                      expression=expression, train_test_split=train_test_split, **kwargs)

        self = cls(hetero, metapaths=hetero.edge_types, add_reverse_metapaths=add_reverse_metapaths,
                   edge_dir="in", head_node_type=head_node_type, pred_ntypes=pred_ntypes, **kwargs)
        self.classes = classes
        self.nodes = {ntype: nids for ntype, nids in nodes.items() if ntype in self.node_types}
        self._name = network._name if hasattr(network, '_name') else ""
        self.network = network

        self.pred_metapaths = network.pred_metapaths if hasattr(network, 'pred_metapaths') else []
        self.neg_pred_metapaths = network.neg_pred_metapaths if hasattr(network, 'neg_pred_metapaths') else []

        self.split_namespace = split_namespace
        if split_namespace:
            self.nodes_namespace = {}
            self.ntype_mapping = {}
            for ntype, df in network.annotations.items():
                if "namespace" in df.columns:
                    node2namespace = network.annotations[ntype]["namespace"]

                    if ntype in self.pred_ntypes:
                        node2namespace = node2namespace.replace({'biological_process': 'BPO',
                                                                 'cellular_component': 'CCO',
                                                                 'molecular_function': 'MFO'})

                    self.nodes_namespace[ntype] = node2namespace.loc[~node2namespace.index.duplicated()]
                    self.ntype_mapping.update(
                        {namespace: ntype for namespace in np.unique(self.nodes_namespace[ntype])})

        return self

    @classmethod
    def load(cls, path: Path, **hparams):
        if isinstance(path, str) and '~' in path:
            path = os.path.expanduser(path)

        hetero: HeteroData = torch.load(join(path, 'heterodata.pt'))

        with open(join(path, 'metadata.pickle'), 'rb') as f:
            attrs: Dict = pickle.load(f)
        if hparams:
            attrs.update({k: v for k, v in hparams.items() if k != 'dataset'})

        self = cls(hetero, **attrs)
        self._name = os.path.basename(path)

        self.network = Namespace()

        # Load node annotations
        self.network.annotations = {}
        for ntype in hetero.node_types:
            ann_df = None
            if os.path.exists(join(path, f'{ntype}.pickle')):
                ann_df = pd.read_pickle(join(path, f'{ntype}.pickle'))
            elif os.path.exists(join(path, f'{ntype}.parquet')):
                ann_df = pd.read_parquet(join(path, f'{ntype}.parquet'))

            if ann_df is not None:
                self.network.annotations[ntype] = ann_df

                if 'sequence' in hparams and hparams['sequence'] and 'sequence' in ann_df.columns:
                    hetero[ntype].sequence = ann_df['sequence']

        # Load sequence tokenizer
        if 'vocabularies' in hparams:
            seq_tokenizer = SequenceTokenizers(
                vocabularies=hparams['vocabularies'],
                max_length=hparams['max_length'] if 'max_length' in hparams else None)
            self.seq_tokenizer = seq_tokenizer

        # Load nodes list
        nodes = pd.read_pickle(join(path, 'nodes.pickle'))
        self.nodes: Dict[str, pd.Index] = {ntype: nids for ntype, nids in nodes.items() if ntype in self.node_types}

        # Post-processing to fix some data inconsistencies
        ## Missing classes not in .obo
        for go_ntype in self.pred_ntypes:
            extra_classes = None
            if go_ntype in self.nodes:
                extra_classes = pd.Index(self.classes).difference(self.nodes[go_ntype])
                if extra_classes.size:
                    logger.info(f"extra nodes {go_ntype}, {self.classes.shape}, {extra_classes.size}")
                    self.nodes[go_ntype] = self.nodes[go_ntype].append(extra_classes)
                assert not self.nodes[go_ntype].duplicated().any()

            # Missing nodes_namespace
            if go_ntype not in self.nodes_namespace:
                self.nodes_namespace[go_ntype] = pd.Series([go_ntype for i in range(self.n_classes)],
                                                           index=self.classes)
            if extra_classes is not None and extra_classes.size:
                logger.info(
                    f"nodes_namespace extra_classes {go_ntype}, {self.nodes_namespace[go_ntype].shape}, {extra_classes.size}")
                self.nodes_namespace[go_ntype] = self.nodes_namespace[go_ntype].append(
                    pd.Series([go_ntype for i in range(len(extra_classes))],
                              index=self.classes))

        # Rename nodes_namespace
        self.nodes_namespace = {ntype: df.replace({'biological_process': 'BPO',
                                                   'cellular_component': 'CCO',
                                                   'molecular_function': 'MFO'}) \
                                for ntype, df in self.nodes_namespace.items()}

        if self.head_node_type in self.network.annotations and \
                'species_id' in self.network.annotations[self.head_node_type].columns:
            self.nodes_namespace[self.head_node_type] = self.network.annotations[self.head_node_type]["species_id"]

        if get_attrs(hparams, 'n_neighbors', None) and hparams.n_neighbors not in self.neighbor_sizes:
            self.neighbor_sizes = [hparams.n_neighbors for _ in self.neighbor_sizes]

        return self

    def save(self, path, add_slug=False):
        if isinstance(path, str) and '~' in path:
            path = os.path.expanduser(path)
            path = path.rstrip('/')

        if add_slug:
            if path.endswith("/"):
                path = path.rstrip('/')
            if not path.endswith('.'):
                path = path + '.'
            # add slug to dataset directory
            path = path + self.name

        logger.info(f"Saving {self.__class__.__name__} to .../{os.path.basename(path)}/")
        if not os.path.exists(path):
            os.makedirs(path)

        # Nodes
        nodes = self.nodes if hasattr(self, "nodes") and self.nodes != None else self.network.nodes
        if isinstance(nodes, (pd.Index, pd.Series)):
            nodes.to_pickle(join(path, 'nodes.pickle'))
        elif isinstance(nodes, dict):
            with open(join(path, 'nodes.pickle'), 'wb') as f:
                pickle.dump(nodes, f)

        # Write ntype annotations
        for ntype, df in self.network.annotations.items():
            if ntype not in self.G.node_types: continue
            node_ann_pickle = join(path, f'{ntype}.pickle')

            if not os.path.exists(node_ann_pickle):
                df.to_pickle(node_ann_pickle)

        # Write ntype feature_transformer
        for target, mlb in self.network.feature_transformer.items():
            mlb_path = join(path, f'{target}.mlb')
            if not os.path.exists(mlb_path):
                joblib.dump(mlb, mlb_path)

        # Write PyG HeteroData
        torch.save(self.G, join(path, 'heterodata.pt'))

        # Write metadata
        attrs = get_attrs(self, exclude={'x_dict', 'y_dict', 'edge_index_dict', 'global_node_index',
                                         'nodes', 'node_attr_shape', 'node_attr_sparse', 'num_nodes_dict',
                                         'node_degrees', 'node_mask_counts', 'node_metadata', 'class_indices'})
        with open(join(path, 'metadata.pickle'), 'wb') as f:
            pickle.dump(attrs, f)

        # Write metadata to JSON so can be readable
        with open(join(path, 'metadata.yml'), 'w') as outfile:
            yaml.dump(tensor_sizes(attrs), outfile, default_flow_style=False)

    def metagraph(self) -> nx.MultiDiGraph:
        G = nx.MultiDiGraph([(u, v, e) for u, e, v in self.G.metadata()[1]])
        return G

    @property
    def name(self) -> str:
        if '-' in self._name and '.' in self._name:
            return self._name

        ntypes = ''.join(s.capitalize()[0] for s in self.node_types)
        pntypes = ''.join(''.join(s[0] for s in ntype.split("_")) for ntype in self.pred_ntypes)
        slug = f'{ntypes}{len(self.metapaths)}-{pntypes}'

        return '.'.join([self._name, slug])

    @property
    def class_indices(self) -> Optional[Dict[str, Tensor]]:
        if self.pred_ntypes is None or set(self.pred_ntypes).difference(self.nodes.keys()) or self.classes is None:
            return None
        class_indices = {}
        for pred_ntype in self.pred_ntypes:
            indices = self.nodes[pred_ntype].get_indexer_for(self.classes)
            class_indices[pred_ntype] = torch.from_numpy(indices)
        return class_indices

    @property
    def node_mask_counts(self) -> DataFrame:
        ntypes = self.G.node_types
        return pd.DataFrame(tensor_sizes(dict(
            train={ntype: self.G[ntype].train_mask.sum() for ntype in ntypes if hasattr(self.G[ntype], 'train_mask')},
            valid={ntype: self.G[ntype].valid_mask.sum() for ntype in ntypes if hasattr(self.G[ntype], 'valid_mask')},
            test={ntype: self.G[ntype].test_mask.sum() for ntype in ntypes if hasattr(self.G[ntype], 'test_mask')})))

    @property
    def edge_mask_counts(self) -> DataFrame:
        etypes = self.G.edge_types
        df = pd.DataFrame(tensor_sizes(dict(
            train={etype: self.G[etype].train_mask.sum() for etype in etypes if hasattr(self.G[etype], 'train_mask')},
            valid={etype: self.G[etype].valid_mask.sum() for etype in etypes if hasattr(self.G[etype], 'valid_mask')},
            test={etype: self.G[etype].test_mask.sum() for etype in etypes if hasattr(self.G[etype], 'test_mask')})))
        if df.empty:
            return None
        df.index.names = ['src_ntype', 'etype', 'dst_ntype']
        return df.sort_index()

    def create_graph_sampler(self,
                             graph: HeteroData,
                             batch_size: int,
                             node_type: str,
                             node_mask: Tensor,
                             transform_fn: Callable = None,
                             neighbor_loader=None,
                             num_neighbors: Dict[Union[str, Tuple], int] = None,
                             add_metapaths: List[List[Tuple[str, str, str]]] = None,
                             max_sample=30,
                             num_workers=0,
                             verbose=False,
                             shuffle=True,
                             **kwargs) -> DataLoader:
        # Num neighbors
        if neighbor_loader is None:
            neighbor_loader = self.neighbor_loader
        if neighbor_loader == "NeighborLoader":
            n_neighbors = {metapath: self.neighbor_sizes for metapath in graph.edge_types}
        elif neighbor_loader == "HGTLoader":
            n_neighbors = {ntype: self.neighbor_sizes for ntype in self.node_types}
        else:
            raise Exception(f"self.neighbor_loader must be one of 'NeighborLoader' or 'HGTLoader'")
        if num_neighbors:
            n_neighbors.update(num_neighbors)

        if verbose:
            pprint(f"{neighbor_loader} neighbor_sizes: {n_neighbors}", width=350)

        # Add metapaths to hetero before
        if add_metapaths and callable(transform_fn):
            op = AddMetaPaths(add_metapaths, max_sample=max_sample, weighted=True)
            logger.info(f"AddMetaPaths {len(add_metapaths)}")
            transform = lambda x: transform_fn(x, op)
        elif callable(transform_fn):
            transform = transform_fn
        else:
            transform = None

        args = dict(
            data=graph,
            num_neighbors=n_neighbors,
            batch_size=batch_size,
            transform=transform,
            input_nodes=(node_type, node_mask),
            shuffle=shuffle,
            num_workers=num_workers,
            **kwargs,
        )

        if self.class_indices is not None and set(self.class_indices.keys()).intersection(self.node_types):
            args["class_indices"] = self.class_indices

        if neighbor_loader == "NeighborLoader":
            dataset = NeighborLoaderX(**args)
        elif neighbor_loader == "HGTLoader":
            dataset = HGTLoaderX(**args)
        else:
            raise Exception()

        return dataset

    def transform(self, hetero: HeteroData, transform: Callable = None) \
            -> Tuple[Dict[str, Dict], Dict, Optional[Dict]]:
        if callable(transform):
            hetero = transform(hetero)

        X = {}
        X["global_node_index"] = {ntype: nid for ntype, nid in hetero.nid_dict.items() if nid.numel()}
        X['sizes'] = {ntype: size for ntype, size in hetero.num_nodes_dict.items() if size}
        X['batch_size'] = hetero.batch_size_dict
        if self.class_indices:
            class_sizes = {ntype: idx.numel() for ntype, idx in self.class_indices.items()}
            X['batch_size'].update(class_sizes)

        # Edge-index dict
        X["edge_index_dict"] = {}
        for metapath, edge_index in hetero.edge_index_dict.items():
            if hasattr(hetero, 'edge_weight_dict') and metapath in hetero.edge_weight_dict:
                edge_attr = hetero.edge_weight_dict[metapath]
            else:
                edge_attr = None

            # Metapath was generated from `AddMetapaths`
            if hasattr(hetero, 'metapath_dict') and metapath in hetero.metapath_dict:
                if metapath[0] == metapath[-1]:
                    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)

                # Concatenate metapaths
                metapath = tuple(itertools.chain.from_iterable(
                    [m[1:] if i > 0 else m for i, m in enumerate(hetero.metapath_dict[metapath])]))
                edge_attr = None

            if edge_attr is not None:
                X["edge_index_dict"][metapath] = (edge_index, edge_attr)
            else:
                X["edge_index_dict"][metapath] = edge_index

        # Node features
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
                if not hasattr(hetero[ntype], "sequence") or ntype not in self.seq_tokenizer.tokenizers: continue
                X["sequences"][ntype] = self.seq_tokenizer.encode_sequences(hetero, ntype=ntype, max_length=None)

        # Node labels
        y_dict = {}
        for ntype, y in hetero.y_dict.items():
            if y.size(0) == 0 or ntype not in X["global_node_index"]: continue
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

    def get_node_metadata(self, ntype_nids: Dict[str, Tensor],
                          embeddings: Dict[str, Tensor],
                          weights: Optional[Dict[str, Series]] = None,
                          losses: Dict[str, np.ndarray] = None,
                          n_clusters=20,
                          update_df: pd.DataFrame = None) -> DataFrame:
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
        self.create_node_metadata(self.network.annotations, nodes=self.nodes)

        ntype_nids: Dict[str, np.ndarray] = OrderedDict(
            {ntype: nids.numpy() if isinstance(nids, Tensor) else nids \
             for ntype, nids in ntype_nids.items() \
             if ntype in embeddings})

        # Concatenated list of node embeddings and other metadata
        nodes_emb = {ntype: emb.detach().cpu().numpy() if isinstance(emb, Tensor) else emb \
                     for ntype, emb in embeddings.items() \
                     if ntype in ntype_nids}
        nodes_emb_concat = np.concatenate([nodes_emb[ntype][:len(nids)] \
                                           for ntype, nids in ntype_nids.items()])

        # train/valid/test
        node_train_valid_test = np.vstack([
            np.concatenate([self.G[ntype].train_mask[nids].numpy() for ntype, nids in ntype_nids.items()]),
            np.concatenate([self.G[ntype].valid_mask[nids].numpy() for ntype, nids in ntype_nids.items()]),
            np.concatenate([self.G[ntype].test_mask[nids].numpy() for ntype, nids in ntype_nids.items()])],
        ).T
        node_train_valid_test = np.array(["Train", "Valid", "Test"])[node_train_valid_test.argmax(1)]

        # Node loss
        ntype_losses = None
        if isinstance(losses, dict) and losses:
            # Add missing nodes for certain ntype in `losses`
            for ntype in (ntype for ntype in losses if losses[ntype].size != ntype_nids[ntype].size):
                num_missing = np.array([None for _ in range(ntype_nids[ntype].size - losses[ntype].size)])
                losses[ntype] = np.concatenate([losses[ntype], num_missing])

            ntype_losses = np.concatenate([losses[ntype] \
                                               if ntype in losses else \
                                               [None for i in range(ntype_nids[ntype].size)] \
                                           for ntype in ntype_nids])

        # Metadata
        # Build node metadata dataframe from concatenated lists of node metadata for multiple ntypes
        if update_df is None:
            data = {"node": np.concatenate([self.nodes[ntype][ntype_nids[ntype]] \
                                            for ntype in ntype_nids]),
                    "ntype": np.concatenate([[ntype for i in range(ntype_nids[ntype].shape[0])] \
                                             for ntype in ntype_nids]),
                    "train_valid_test": node_train_valid_test,
                    "loss": ntype_losses}
            df = pd.DataFrame(
                data,
                index=pd.Index(np.concatenate([ntype_nids[ntype] for ntype in ntype_nids]), name="nid"))
        else:
            df = update_df

        # Get TSNE 2d position from embeddings
        if 'pos1' not in df.columns:
            try:
                from umap import UMAP
                tsne = UMAP(n_components=2, n_jobs=-1, n_epochs=100, verbose=True)
            except ImportError as ie:
                logger.error(ie.__repr__())
                import MulticoreTSNE
                tsne = MulticoreTSNE.MulticoreTSNE(n_components=2, perplexity=15, learning_rate=10, n_jobs=-1)

            try:
                tsne.fit(nodes_emb_concat)
            except KeyboardInterrupt:
                logger.info(f"Stop training {tsne}")
            finally:
                nodes_pos = tsne.transform(nodes_emb_concat)

            nodes_pos = {node_name: pos for node_name, pos in zip(df.index, nodes_pos)}
            df[['pos1', 'pos2']] = np.vstack(df.index.map(nodes_pos))
            df = df.assign(loss=df['loss'].astype(float), pos1=df['pos1'].astype(float), pos2=df['pos2'].astype(float))

        # Predict kmeans clusters
        if n_clusters and 'kmeans_cluster_id' not in df.columns:
            logger.info(f"Kmeans with k={n_clusters}")
            kmeans_pred = []
            for ntype in ntype_nids:
                kmeans = KMeans(n_clusters)
                pred = ntype + pd.Index(kmeans.fit_predict(nodes_emb[ntype])).astype(str)
                kmeans_pred.append(pred)
            df['kmeans_cluster_id'] = np.concatenate(kmeans_pred)

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
        self.node_metadata.update(df[~df.index.duplicated()], )

        # return only nodes that have > 0 weights (used for visualization of node clf models)
        if weights is not None:
            nodes_weight = {ntype: weights[ntype].detach().numpy() \
                if isinstance(weights[ntype], Tensor) else weights[ntype] \
                            for ntype in weights}
            nodes_weight = np.concatenate([nodes_weight[ntype] for ntype in ntype_nids]).astype(bool)

            return df.loc[nodes_weight]

        return df

    def to_networkx(self, edge_index_dict: Dict[Tuple[str, str, str], Tensor] = None,
                    alphas_adjacencies: Dict[str, pd.DataFrame] = None,
                    nodes_subgraph: Dict[str, Union[List[str], List[int]]] = None,
                    num_hops: int = 1,
                    min_value: float = None,
                    global_node_idx: Dict[str, Tensor] = None,
                    node_title: Dict[str, str] = None,
                    sep="-", ) -> nx.MultiDiGraph:
        """

        Args:
            edge_index_dict (Dict[Tuple[str, str, str], Tensor]): default None.

            alphas_adjacencies (Dict[str, pd.DataFrame]): default None.
                A dict of metapaths (joined str) to sparse DataFrames adjancecy matrix with index and columns containing
                node index for self.nodes.
            nodes_subgraph (Dict[str, List[str]]): A dict of ntype and node lists.
                Only select edges that have dst nodes provided in `nodes_subgraph`.
            min_value (float): default None
                If given a number, then only select edges with edge `weight` >= `min_value`.
            global_node_idx ():
            sep (str): default "-".
                If given, then the node names in networkx will be

        Returns:

        """
        G = nx.MultiDiGraph()

        if edge_index_dict:
            if isinstance(edge_index_dict, list):
                # edge_index_dict is a list of metapaths
                edge_index_dict = {m: eidx for m, eidx in self.G.edge_index_dict.items() if
                                   m in edge_index_dict or m[1] in edge_index_dict}

            edge_index_dict = {m: eid for m, eid in edge_index_dict.items() if "rev_" not in m[1]}

            # Need sep in order to differentiate same node names between multiple ntypes
            edgelists = convert_to_nx_edgelist(edge_index_dict, node_names=self.nodes,
                                               global_node_idx=global_node_idx,
                                               sep=sep)
            for etype, edges in edgelists.items():
                G.add_edges_from(edges, etype=etype)

            # Filter by `nodes` to get subgraph
            if nodes_subgraph:
                filter_nodes = []

                for ntype, node_list in nodes_subgraph.items():
                    if isinstance(node_list, Tensor):
                        node_list = node_list.detach().cpu().numpy().tolist()

                    if all(isinstance(node, str) for node in node_list):
                        select_nodes = pd.Index(node_list)
                    elif all(isinstance(node, int) for node in node_list):
                        select_nodes = self.nodes[ntype][nodes_subgraph[ntype]]
                    else:
                        print([type(node) for node in node_list])
                        select_nodes = []

                    if sep and isinstance(select_nodes, pd.Index):
                        select_nodes = ntype + "-" + select_nodes

                    if set(select_nodes).difference(set(G.nodes())):
                        G.add_nodes_from(set(select_nodes).difference(set(G.nodes())))

                    filter_nodes.extend(select_nodes)

                if len(filter_nodes):
                    G = G.copy().subgraph(filter_nodes)

        elif alphas_adjacencies:
            for etype, adj in tqdm.tqdm(alphas_adjacencies.items(), total=len(alphas_adjacencies)):
                metapath = etype.split(".")
                head_type, tail_type = metapath[0], metapath[-1]
                coo:coo_matrix = adj.sparse.to_coo()

                if isinstance(min_value, (int, float)):
                    coo.data *= (coo.data >= min_value)
                    coo.eliminate_zeros()

                src_idx = adj.columns[coo.col]
                dst_idx = adj.index[coo.row]
                weights = coo.data

                if sep:
                    src = head_type + "-" + self.nodes[head_type][src_idx]
                    dst = tail_type + "-" + self.nodes[tail_type][dst_idx]
                else:
                    src = self.nodes[head_type][src_idx]
                    dst = self.nodes[tail_type][dst_idx]

                # Filter by `nodes` to get subgraph
                if nodes_subgraph and tail_type in nodes_subgraph:
                    mask = self.nodes[tail_type][dst_idx].isin(nodes_subgraph[tail_type])
                    for i in range(1, num_hops):
                        mask = mask | np.isin(dst, src[mask])
                    src, dst, weights = src[mask], dst[mask], weights[mask]

                # Edges
                key = intern('.'.join(metapath[1::2]))
                G.add_edges_from(((u, v, key, {intern('weight'): w, intern('value'): w})
                                  for u, v, w in zip(src, dst, weights.tolist())),
                                 title=key,
                                 color=intern(colorhash.ColorHash(key).hex))
                # Nodes
                for ntype, nodelist in zip([head_type, tail_type], [src, dst]):
                    if node_title and ntype in node_title and \
                            node_title[ntype] in self.network.annotations[ntype].columns:
                        node_label = self.network.annotations[ntype][node_title[ntype]]

                        node_attrs = {node: {intern('group'): ntype,
                                             intern('title'): ntype,
                                             intern('label'): node_label[node]} \
                                      for node in nodelist}
                    else:
                        node_attrs = {node: {intern('group'): ntype,
                                             intern('title'): ntype} \
                                      for node in nodelist}

                    nx.set_node_attributes(G, node_attrs)

        else:
            raise Exception('Must provide at least one of `edge_index_dict` or `alphas_adjacencies`')

        H = G.copy()
        H.remove_nodes_from(list(nx.isolates(H)))

        return H

    def remove_nodes(self, G: HeteroData, nodes: Dict[str, Union[torch.BoolTensor, np.ndarray, pd.Index]]) \
            -> HeteroData:
        """
        Given a set of nodes to remove, drop all edges incident to those nodes (but keeping the node set intact)

        Args:
            G (Data or HeteroData):
            nodes (): A dict of ntype to either a boolean mask vector, a node ids integer vector, or a list of node
                name strings.

        Returns:
            HeteroData
        """
        data = G.__copy__()
        # Gather all nodes
        n_id_dict = {ntype: torch.arange(data[ntype].num_nodes) for ntype in data.node_types}

        for ntype, drop_ids in nodes.items():
            element = drop_ids[0]
            if isinstance(element, str):
                drop_ids = self.nodes[ntype].get_indexer_for(drop_ids)
            elif isinstance(element.dtype, (type(torch.bool), type(np.bool))):
                assert len(drop_ids) == len(n_id_dict[ntype]) == data[ntype].num_nodes, \
                    f'{len(drop_ids)} != {len(n_id_dict[ntype])} != {data[ntype].num_nodes}'
                drop_ids = drop_ids.nonzero().ravel()

            element = drop_ids[0]
            assert isinstance(element, int) or isinstance(element.dtype, (type(torch.int), type(np.int))), \
                f"value {element} must be a int"
            mask = np.isin(n_id_dict[ntype], drop_ids, assume_unique=True, invert=True)
            n_id_dict[ntype] = n_id_dict[ntype][mask]

        logger.info(f'Removed {data.num_nodes - sum(len(nids) for nids in n_id_dict.values())} nodes')

        for store in data.edge_stores:
            if 'edge_index' not in store:
                continue

            if store._key is None:
                src = dst = None
            else:
                src, _, dst = store._key

            row_mask = np.isin(store.edge_index[0], n_id_dict[src], assume_unique=False)
            col_mask = np.isin(store.edge_index[1], n_id_dict[dst], assume_unique=False)
            mask = row_mask & col_mask

            for key, value in store.items():
                if key == 'edge_index':
                    store.edge_index = store.edge_index[:, mask]

                elif value.shape[0] == mask.shape[0]:
                    store[key] = value[mask]

        return data

    def remove_edges(self, G: HeteroData, edge_masks: Dict[str, torch.BoolTensor]):
        data = G.__copy__()
        #     assert len(drop_ids) == data[etype].num_edges, f'{len(drop_ids)} != {data[etype].num_edges}'

        for store in data.edge_stores:
            if 'edge_index' not in store:
                continue

            if store._key is None:
                src = dst = None
            else:
                src, _, dst = store._key

            if store._key in edge_masks:
                mask = edge_masks[store._key]
                for key, value in store.items():
                    if key == 'edge_index':
                        store.edge_index = store.edge_index[:, mask]
                    elif value.shape[0] == mask.shape[0]:
                        store[key] = value[mask]

    def train_dataloader(self, collate_fn=None, batch_size=128, num_workers=0, add_metapaths=None, **kwargs):
        graph = self.G
        head_ntype = self.head_node_type

        if self.inductive:
            valid_test_nodes = {head_ntype: graph[head_ntype].valid_mask | graph[head_ntype].test_mask}
            graph = self.remove_nodes(graph, nodes=valid_test_nodes)

        # Select only train nodes with at least 1 label
        node_mask = graph[head_ntype].train_mask & graph[head_ntype].y.sum(1).type(torch.bool)
        logger.info(f'train_dataloader size: {node_mask.sum()}')

        dataset = self.create_graph_sampler(graph, batch_size, node_type=head_ntype, node_mask=node_mask,
                                            transform_fn=self.transform, add_metapaths=add_metapaths,
                                            num_workers=num_workers, shuffle=True)

        return dataset

    def valid_dataloader(self, collate_fn=None, batch_size=128, num_workers=0, add_metapaths=None, **kwargs):
        graph = self.G
        head_ntype = self.head_node_type

        # Select only valid nodes with at least 1 label
        node_mask = graph[head_ntype].valid_mask & graph[head_ntype].y.sum(1).type(torch.bool)
        logger.info(f'valid_dataloader size: {node_mask.sum()}')

        num_train_overlap = (graph[head_ntype].valid_mask & (graph[head_ntype].train_mask == True)).sum()
        assert num_train_overlap == 0, f"num_train_overlap: {num_train_overlap}"

        dataset = self.create_graph_sampler(self.G, batch_size, node_type=head_ntype, node_mask=node_mask,
                                            transform_fn=self.transform, add_metapaths=add_metapaths,
                                            num_workers=num_workers, shuffle=False)

        return dataset

    def test_dataloader(self, collate_fn=None, batch_size=128, num_workers=0, add_metapaths=None, node_mask=None,
                        **kwargs):
        graph = self.G
        head_ntype = self.head_node_type

        # Select only test nodes with at least 1 label
        if node_mask is not None:
            node_mask = node_mask & graph[head_ntype].y.sum(1).type(torch.bool)
        else:
            node_mask = graph[head_ntype].test_mask & graph[head_ntype].y.sum(1).type(torch.bool)
        logger.info(f'test_dataloader size: {node_mask.sum()}')

        num_train_overlap = (graph[head_ntype].test_mask & (graph[head_ntype].train_mask == True)).sum()
        assert num_train_overlap == 0, f"num_train_overlap: {num_train_overlap}"

        dataset = self.create_graph_sampler(self.G, batch_size, node_type=head_ntype, node_mask=node_mask,
                                            transform_fn=self.transform, add_metapaths=add_metapaths,
                                            num_workers=num_workers, shuffle=False)

        return dataset


class HeteroLinkPredDataset(HeteroNodeClfDataset):
    def __init__(self, dataset: HeteroData,
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
                 **kwargs):
        super().__init__(dataset, seq_tokenizer=seq_tokenizer, neighbor_loader=neighbor_loader,
                         neighbor_sizes=neighbor_sizes, node_types=node_types, metapaths=metapaths,
                         head_node_type=head_node_type, edge_dir=edge_dir, add_reverse_metapaths=add_reverse_metapaths,
                         inductive=inductive, train_test_split=train_test_split, **kwargs)
        self.negative_sampling_size = negative_sampling_size
        self.eval_negative_sampling_size = 1000

        self.neighbor_sizes = neighbor_sizes
        self.multilabel = False

        self.graph_sampler = self.create_graph_sampler(self.G, batch_size=1,
                                                       node_type=head_node_type,
                                                       node_mask=torch.ones(self.G[self.head_node_type].num_nodes),
                                                       transform_fn=super().transform,
                                                       num_workers=0)

    @classmethod
    def from_heteronetwork(cls, network: HeteroNetwork, node_attr_cols: List[str] = None, target: str = None,
                           min_count: int = None, expression=False, sequence=False,
                           label_subset: Optional[Union[Index, np.ndarray]] = None,
                           head_node_type: Optional[str] = None,
                           ntype_subset: Optional[List[str]] = None, exclude_metapaths=None,
                           add_reverse_metapaths=True,
                           split_namespace=False,
                           **kwargs):
        self = super().from_heteronetwork(network, node_attr_cols=node_attr_cols,
                                          target=target, min_count=min_count, expression=expression, sequence=sequence,
                                          label_subset=label_subset, head_node_type=head_node_type,
                                          ntype_subset=ntype_subset,
                                          exclude_etypes=network.pred_metapaths,
                                          add_reverse_metapaths=add_reverse_metapaths, split_namespace=split_namespace,
                                          **kwargs)
        assert 'go_ntype' in kwargs

        # Train/valid/test positive annotations
        self.triples, self.training_idx, self.validation_idx, self.testing_idx = \
            network.get_triples(self.pred_metapaths, positive=True)

        # Train/valid/test negative annotations
        self.triples_neg, self.training_idx_neg, self.validation_idx_neg, self.testing_idx_neg = \
            network.get_triples(self.neg_pred_metapaths, negative=True)

        self.triples.update(self.triples_neg)

        # Adjacency of pos edges (for neg sampling)
        edge_index_dict, edge_neg_dict = get_relabled_edge_index(
            triples=self.triples,
            global_node_index={ntype: torch.arange(len(nodelist)) for ntype, nodelist in self.nodes.items()},
            metapaths=self.pred_metapaths,
            relation_ids_all=self.triples["relation"].unique())

        self.triples_adj: Dict[Tuple[str, str, str], SparseTensor] = edge_index_to_adjs(edge_index_dict,
                                                                                        nodes=self.nodes)

        return self


    def get_prior(self) -> Tensor:
        return torch.tensor(1) / (1 + self.negative_sampling_size)

    @staticmethod
    def relabel_edge_index_dict(edge_index_dict: Dict[Tuple[str, str, str], Tensor],
                                global_node_index: Dict[str, Tensor],
                                global2batch: Dict[str, Tensor] = None,
                                mode=None) -> Dict[Tuple[str, str, str], Tensor]:
        if global2batch is None:
            global2batch = {
                node_type: dict(zip(global_node_index[node_type].numpy(), range(len(global_node_index[node_type])))) \
                for node_type in global_node_index}

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
                raise Exception("Must provide an edge_index with shape (2, N) or pass 'head_batch' or 'tail_batch' "
                                "in `mode`.")

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
        edge_pos, edge_neg = self.get_edge_index_dict_from_triples(edge_idx, neg_edge_idx=all_neg_edge_idx,
                                                                   metapaths=self.pred_metapaths)

        # Get all nodes induced by sampled edges
        if num_edges(edge_neg):
            query_edges = {
                metapath: torch.cat(
                    [edge_pos[metapath] if metapath in edge_pos else torch.tensor([], dtype=torch.long),
                     edge_neg[metapath] if metapath in edge_neg else torch.tensor([], dtype=torch.long)], dim=1) \
                for metapath in set(edge_pos).union(set(edge_neg))}
        else:
            query_edges = edge_pos

        query_nodes = gather_node_dict(query_edges)

        # Add random GO term nodes for negative sampling
        go_ntype = self.pred_ntypes[0]
        go_nodes_proba = 1 - F.one_hot(query_nodes[go_ntype], num_classes=self.num_nodes_dict[go_ntype]) \
            .sum(axis=0).to(torch.float)
        go_nids = torch.multinomial(go_nodes_proba, num_samples=self.negative_sampling_size, replacement=False)
        query_nodes[go_ntype] = torch.cat([query_nodes[go_ntype], go_nids])

        # Get subgraph induced by neighborhood hopping from the query nodes
        X, _, _ = self.graph_sampler.transform_fn(self.graph_sampler.collate_fn(query_nodes))
        # print(tensor_sizes(edge_pos))
        X['batch_size'] = {ntype: query_nodes[ntype].numel() for ntype in query_nodes}

        # Edge_pos must be global index, not batch index
        if self.split_namespace:
            edge_pos = self.split_edge_index_by_nodes_namespace(edge_pos, batch_to_global=None)
        head_batch, tail_batch = self.generate_negative_sampling(edge_pos,
                                                                 global_node_index=X["global_node_index"],
                                                                 max_negative_sampling_size=max_negative_sampling_size,
                                                                 mode=mode)

        # Rename node index from global to batch
        global2batch = {ntype: dict(zip(
            X["global_node_index"][ntype].numpy(),
            range(len(X["global_node_index"][ntype])))
        ) for ntype in X["global_node_index"]}
        if self.split_namespace:
            for namespace, go_ntype in self.ntype_mapping.items():
                global2batch[namespace] = global2batch[go_ntype]

        edge_true = {}
        edge_pos = self.relabel_edge_index_dict(edge_pos, global_node_index=X["global_node_index"],
                                                global2batch=global2batch)
        if self.split_namespace:
            edge_true['edge_pos'] = self.split_edge_index_by_nodes_namespace(edge_pos,
                                                                             batch_to_global=X["global_node_index"])
        else:
            edge_true['edge_pos'] = edge_pos

        if num_edges(edge_neg):
            edge_neg = self.relabel_edge_index_dict(edge_neg, global_node_index=X["global_node_index"],
                                                    global2batch=global2batch, )
            if self.split_namespace:
                edge_true['edge_neg'] = self.split_edge_index_by_nodes_namespace(edge_neg,
                                                                                 batch_to_global=X["global_node_index"])
            else:
                edge_true['edge_neg'] = edge_neg

        # Negative sampling
        edge_true.update({"head_batch": head_batch, "tail_batch": tail_batch, })

        edge_weights = None
        return X, edge_true, edge_weights

    # def query(self):

    def get_edge_index_dict_from_triples(self, edge_idx: Tensor, neg_edge_idx: Tensor = None, metapaths=None):
        triples = {k: v[edge_idx] for k, v in self.triples.items() if not is_negative(k)}

        # If ensures same number of true neg edges to true pos edges
        if neg_edge_idx is not None:
            neg_edge_idx = np.random.choice(neg_edge_idx, size=min(edge_idx.numel(), neg_edge_idx.numel()),
                                            replace=False)
            triples_neg = {k: v[neg_edge_idx] for k, v in self.triples.items() if is_negative(k)}
            triples.update(triples_neg)

        # Get edge_index_dict from triplets
        if metapaths is None:
            metapaths = self.metapaths
        edge_pos, edge_neg = get_relabled_edge_index(triples=triples,
                                                     global_node_index=self.global_node_index,
                                                     metapaths=metapaths)
        return edge_pos, edge_neg

    def generate_negative_sampling(self, edge_pos: Dict[Tuple[str, str, str], Tensor],
                                   global_node_index: Dict[str, Tensor],
                                   max_negative_sampling_size: int,
                                   mode: str = None) \
            -> Tuple[Dict[Tuple[str, str, str], Tensor], Dict[Tuple[str, str, str], Tensor]]:
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

        sampling_size = self.get_neg_sampling_size(edge_pos, global_node_index=global_node_index,
                                                   max_samples=max_negative_sampling_size, mode=mode)
        go_ntype = self.pred_ntypes[0]

        # Perform negative sampling
        for metapath, edge_index in edge_pos.items():
            head_type, tail_type = metapath[0], metapath[-1]
            adj: SparseTensor = self.triples_adj[metapath] \
                if metapath in self.triples_adj \
                else self.triples_adj[metapath[:-1] + (go_ntype,)]

            # head noise distribution
            head_neg_nodes = global_node_index[head_type]
            head_prob_dist = 1 - adj[head_neg_nodes, edge_index[1]].to_dense().T

            head_batch[metapath] = torch.multinomial(head_prob_dist, num_samples=sampling_size, replacement=True)

            # Tail noise distribution
            tail_neg_nodes = global_node_index[tail_type if tail_type in global_node_index else go_ntype]
            tail_prob_dist = 1 - adj[edge_index[0], tail_neg_nodes].to_dense()

            # Only generate negative tail_batch within BPO, CCO, or MFO terms of the positive edge's tail go_type
            # for go_type in ['biological_process', 'cellular_component', 'molecular_function']:
            #     if go_type != tail_type: continue
            #
            #     go_terms_mask = self.nodes_namespace[tail_neg_nodes] != go_type
            #     tail_prob_dist[:, go_terms_mask] = 0

            tail_batch[metapath] = torch.multinomial(tail_prob_dist, num_samples=sampling_size, replacement=True)

        return head_batch, tail_batch

    def get_neg_sampling_size(self, edge_index_dict: Dict[Tuple[str, str, str], Tensor],
                              global_node_index: Dict[str, Tensor],
                              max_samples: int, mode: str) -> int:
        # Choose neg sampling size by twice (minimum) the count of head or tail nodes, if it's less than self.negative_sampling_size
        if mode == "train":
            head_tail_ntypes = [ntype for metapath in edge_index_dict for ntype in [metapath[0], metapath[-1]]]
            min_num_node = min(
                global_node_index[ntype if ntype in global_node_index else self.ntype_mapping[ntype]].size(0) \
                for ntype in head_tail_ntypes)

            sampling_size = min(min_num_node, max_samples // 2)

        else:
            sampling_size = max_samples // 2

        sampling_size = int(max(sampling_size, 1))

        return sampling_size

    def split_edge_index_by_nodes_namespace(self, edge_index_dict: Dict[Tuple[str, str, str], Tensor],
                                            batch_to_global: Dict[str, Tensor] = None) -> Dict[
        Tuple[str, str, str], Tensor]:
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

    def to_networkx(self, edge_index_dict: Dict[Tuple[str, str, str], Tensor] = None,
                    pos_edges=None,
                    nodes_subgraph: Dict[str, Union[List[str], List[int]]] = None,
                    global_node_idx: Dict[str, Tensor] = None) \
            -> nx.MultiDiGraph:
        G = super().to_networkx(edge_index_dict, nodes_subgraph=nodes_subgraph, global_node_idx=global_node_idx)

        if pos_edges is not None:
            edge_list = convert_to_nx_edgelist(pos_edges, node_names=self.nodes,
                                               global_node_idx=global_node_idx)
            for etype, edges in edge_list.items():
                G.add_edges_from(edges, etype=etype)

        return G

    def train_dataloader(self, collate_fn=None, batch_size=128, num_workers=0, shuffle=True, **kwargs):
        dataset = DataLoader(self.training_idx, batch_size=batch_size,
                             collate_fn=self.transform,
                             shuffle=shuffle,
                             num_workers=num_workers, **kwargs)
        return dataset

    def valid_dataloader(self, collate_fn=None, batch_size=128, num_workers=0, shuffle=False, **kwargs):
        dataset = DataLoader(self.validation_idx, batch_size=batch_size,
                             collate_fn=self.transform,
                             shuffle=shuffle,
                             num_workers=num_workers, **kwargs)
        return dataset

    def test_dataloader(self, collate_fn=None, batch_size=128, num_workers=0, shuffle=False, **kwargs):
        dataset = DataLoader(self.testing_idx, batch_size=batch_size,
                             collate_fn=self.transform,
                             shuffle=shuffle,
                             num_workers=num_workers, **kwargs)
        return dataset

    def dataloader(self, edge_idx, batch_size=128, num_workers=0):
        dataset = DataLoader(edge_idx, batch_size=batch_size,
                             collate_fn=self.transform,
                             shuffle=False,
                             num_workers=num_workers)
        return dataset
