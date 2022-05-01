from pprint import pprint
from typing import List, Tuple, Union, Dict

import networkx as nx
import numpy as np
import pandas as pd
import torch
from openomics.database.ontology import Ontology
from torch_geometric.data import HeteroData

from moge.dataset.PyG.neighbor_sampler import NeighborLoader, HGTLoader
from moge.dataset.graph import HeteroGraphDataset
from moge.dataset.sequences import SequenceTokenizer
# from torch_geometric.loader import HGTLoader, NeighborLoader
from moge.dataset.utils import get_edge_index


class HeteroNodeClfDataset(HeteroGraphDataset):
    def __init__(self, dataset: HeteroData,
                 seq_tokenizer: SequenceTokenizer = None,
                 neighbor_loader: str = "NeighborLoader",
                 neighbor_sizes: Union[List[int], Dict[str, List[int]]] = [128, 128],
                 node_types: List[str] = None, metapaths: List[Tuple[str, str, str]] = None, head_node_type: str = None,
                 edge_dir: str = "in", reshuffle_train: float = None, add_reverse_metapaths: bool = False,
                 inductive: bool = False, **kwargs):
        super().__init__(dataset, node_types, metapaths, head_node_type, edge_dir, reshuffle_train,
                         add_reverse_metapaths, inductive, **kwargs)
        if seq_tokenizer:
            self.seq_tokenizer = seq_tokenizer

        self.neighbor_loader = neighbor_loader
        self.neighbor_sizes = neighbor_sizes

        if self.neighbor_loader == "NeighborLoader":
            self.num_neighbors = {
                etype: neighbor_sizes if etype[1] != 'associated' else [-1, ] * len(neighbor_sizes)
                for etype in self.metapaths}
        elif self.neighbor_loader == "HGTLoader":
            self.num_neighbors = {
                ntype: neighbor_sizes if ntype != 'GO_term' else [self.num_nodes_dict["GO_term"], ] * len(
                    neighbor_sizes)
                for ntype in self.node_types}

        print(f"{self.neighbor_loader} neighbor_sizes:")
        pprint(self.num_neighbors)

    @classmethod
    def from_pyg_heterodata(cls, hetero: HeteroData, classes: List[str], **kwargs):
        self = cls(dataset=hetero, metapaths=hetero.edge_types, add_reverse_metapaths=False,
                   edge_dir="in", **kwargs)
        self.classes = classes
        self._name = ""

        return self

    def process_pyg_heterodata(self, hetero: HeteroData):
        self.G = hetero
        self.x_dict = hetero.x_dict
        self.node_types = hetero.node_types
        self.num_nodes_dict = {ntype: hetero[ntype].num_nodes for ntype in hetero.node_types}
        self.y_dict = {ntype: hetero[ntype].y for ntype in hetero.node_types if hasattr(hetero[ntype], "y")}

        self.metapaths = hetero.edge_types
        self.edge_index_dict = {etype: edge_index for etype, edge_index in zip(hetero.edge_types, hetero.edge_stores)}

    def add_ontology_edges(self, ontology, train_date='2017-06-15', valid_date='2017-11-15', ):
        all_go = set(ontology.network.nodes).intersection(ontology.data.index)

        # Order nodes with classes nodes first
        if self.classes is not None:
            go_nodes = np.concatenate([self.classes, np.array(list(set(all_go) - set(self.classes)))])
        else:
            go_nodes = np.array(list(all_go))

        # Edges between GO terms
        edge_types = {e for u, v, e in ontology.network.edges}
        go_ntype = "GO_term"
        edge_index_dict = ontology.to_scipy_adjacency(nodes=go_nodes, edge_types=edge_types,
                                                      format="pyg", d_ntype=go_ntype)
        for metapath, edge_index in edge_index_dict.items():
            if edge_index.size(1) < 200: continue
            self.G[metapath].edge_index = edge_index

        # Cls node attrs
        for attr, values in ontology.data.loc[go_nodes][["name", "namespace", "def"]].iteritems():
            self.G[go_ntype][attr] = values.to_numpy()

        self.G[go_ntype]['nid'] = torch.arange(len(go_nodes), dtype=torch.long)
        self.G[go_ntype].num_nodes = len(go_nodes)

        # Edges between RNA nodes and GO terms
        train_go_ann, valid_go_ann, test_go_ann = ontology.annotation_train_val_test_split(
            train_date=train_date, valid_date=valid_date, groupby=["gene_name"])
        go_ann = pd.concat([train_go_ann, valid_go_ann, test_go_ann], axis=0)
        nx_graph = nx.from_pandas_edgelist(go_ann["go_id"].explode().to_frame().reset_index(),
                                           source="gene_name", target="go_id", create_using=nx.DiGraph)
        metapath = (self.head_node_type, "associated", go_ntype)
        self.G[metapath].edge_index = get_edge_index(nx_graph,
                                                     nodes_A=self.nodes[metapath[0]], nodes_B=go_nodes)

        # Set test nodes as new nodes in annotations
        train_node_list = train_go_ann.index
        valid_node_list = valid_go_ann.index.drop(train_go_ann.index, errors="ignore")
        test_node_list = test_go_ann.index.drop(train_go_ann.index, errors="ignore") \
            .drop(valid_go_ann.index, errors="ignore")

        # Set train test split on hetero graph
        train_idx = {ntype: ntype_nids.get_indexer_for(ntype_nids.intersection(train_node_list)) \
                     for ntype, ntype_nids in self.nodes.items()}
        valid_idx = {ntype: ntype_nids.get_indexer_for(ntype_nids.intersection(valid_node_list)) \
                     for ntype, ntype_nids in self.nodes.items()}
        test_idx = {ntype: ntype_nids.get_indexer_for(ntype_nids.intersection(test_node_list)) \
                    for ntype, ntype_nids in self.nodes.items()}

        for ntype in self.node_types:
            if self.G[ntype].num_nodes is None:
                self.G[ntype].num_nodes = len(self.nodes[ntype])

            if ntype in train_idx:
                mask = torch.zeros(self.G[ntype].num_nodes, dtype=torch.bool)
                mask[train_idx[ntype]] = 1
                self.G[ntype].train_mask = mask

            if ntype in valid_idx:
                mask = torch.zeros(self.G[ntype].num_nodes, dtype=torch.bool)
                mask[valid_idx[ntype]] = 1
                self.G[ntype].valid_mask = mask

            if ntype in test_idx:
                mask = torch.zeros(self.G[ntype].num_nodes, dtype=torch.bool)
                mask[test_idx[ntype]] = 1
                self.G[ntype].test_mask = mask

    def sample(self, batch: HeteroData):
        X = {}
        X["x_dict"] = {ntype: x for ntype, x in batch.x_dict.items() if x.size(0)}
        X["edge_index_dict"] = {metapath: edge_index for metapath, edge_index in batch.edge_index_dict.items() if
                                "associated" not in metapath[1]}
        X["global_node_index"] = {ntype: nid for ntype, nid in batch.nid_dict.items() if nid.numel()}
        X['sizes'] = {ntype: size for ntype, size in batch.num_nodes_dict.items() if size}
        X['batch_size'] = batch.batch_size_dict

        if hasattr(batch, "sequence_dict") and hasattr(self, "seq_tokenizer"):
            X["sequences"] = {}
            for ntype in X["global_node_index"]:
                X["sequences"][ntype] = self.seq_tokenizer.encode_sequences(batch, ntype=ntype, max_length=None)

        y_dict = {ntype: y for ntype, y in batch.y_dict.items() if y.size(0)}
        if len(y_dict) == 1:
            y_dict = y_dict[list(y_dict.keys()).pop()]

            if y_dict.dim() == 2 and y_dict.size(1) == 1:
                y_dict = y_dict.squeeze(-1)
            elif y_dict.dim() == 1:
                weights = (y_dict >= 0).to(torch.float)

        elif len(y_dict) > 1:
            weights = {}
            for ntype, label in y_dict.items():
                if label.dim() == 2 and label.size(1) == 1:
                    y_dict[ntype] = label.squeeze(-1)

                if label.dim() == 1:
                    weights[ntype] = (y_dict >= 0).to(torch.float)
                elif label.dim() == 2:
                    weights[ntype] = (label.sum(1) > 0).to(torch.float)
        else:
            weights = None

        return X, y_dict, weights

    def train_dataloader(self, collate_fn=None, batch_size=128, num_workers=10, **kwargs):
        if self.neighbor_loader == "NeighborLoader":
            Loader = NeighborLoader
        elif self.neighbor_loader == "HGTLoader":
            Loader = HGTLoader

        dataset = Loader(self.G,
                         num_neighbors=self.num_neighbors,
                         batch_size=batch_size,
                         # directed=True,
                         transform=self.sample,
                         input_nodes=(self.head_node_type, self.G[self.head_node_type].train_mask),
                         shuffle=True,
                         num_workers=num_workers,
                         **kwargs)

        return dataset

    def valid_dataloader(self, collate_fn=None, batch_size=128, num_workers=5, **kwargs):
        if self.neighbor_loader == "NeighborLoader":
            Loader = NeighborLoader
        elif self.neighbor_loader == "HGTLoader":
            Loader = HGTLoader

        dataset = NeighborLoader(self.G, num_neighbors=self.num_neighbors,
                                 batch_size=batch_size,
                                 # directed=False,
                                 transform=self.sample,
                                 input_nodes=(self.head_node_type, self.G[self.head_node_type].valid_mask),
                                 shuffle=False, num_workers=num_workers, **kwargs)

        return dataset

    def test_dataloader(self, collate_fn=None, batch_size=128, num_workers=5, **kwargs):
        if self.neighbor_loader == "NeighborLoader":
            Loader = NeighborLoader
        elif self.neighbor_loader == "HGTLoader":
            Loader = HGTLoader

        dataset = Loader(self.G, num_neighbors=self.num_neighbors,
                         batch_size=batch_size,
                         # directed=False,
                         transform=self.sample,
                         input_nodes=(self.head_node_type, self.G[self.head_node_type].test_mask),
                         shuffle=False, num_workers=num_workers, **kwargs)

        return dataset


class HeteroLinkPredDataset(HeteroNodeClfDataset):
    def __init__(self, dataset: HeteroData, seq_tokenizer: SequenceTokenizer = None,
                 neighbor_loader: str = "NeighborLoader",
                 neighbor_sizes: Union[List[int], Dict[str, List[int]]] = [128, 128], node_types: List[str] = None,
                 metapaths: List[Tuple[str, str, str]] = None, head_node_type: str = None, edge_dir: str = "in",
                 reshuffle_train: float = None, add_reverse_metapaths: bool = True, inductive: bool = False, **kwargs):
        super().__init__(dataset, seq_tokenizer, neighbor_loader, neighbor_sizes, node_types, metapaths, head_node_type,
                         edge_dir, reshuffle_train, add_reverse_metapaths, inductive, **kwargs)
        if self.neighbor_loader == "NeighborLoader":
            Loader = NeighborLoader
        elif self.neighbor_loader == "HGTLoader":
            Loader = HGTLoader

        self.neighbor_sampler = Loader(self.G,
                                       num_neighbors=self.num_neighbors,
                                       batch_size=1,
                                       # directed=True,
                                       transform=self.sample,
                                       shuffle=True,
                                       num_workers=10,
                                       **kwargs)

    def add_ontology_edges(self, ontology: Ontology, train_date='2017-06-15', valid_date='2017-11-15',
                           metapaths: List[Tuple[str, str, str]] = None):
        all_go = set(ontology.network.nodes).intersection(ontology.data.index)
        go_nodes = np.array(list(all_go))

        # Edges between GO terms
        edge_types = {e for u, v, e in ontology.network.edges}
        go_ntype = "GO_term"
        edge_index_dict = ontology.to_scipy_adjacency(nodes=go_nodes, edge_types=edge_types,
                                                      format="pyg", d_ntype=go_ntype)
        for metapath, edge_index in edge_index_dict.items():
            if edge_index.size(1) < 200: continue
            self.G[metapath].edge_index = edge_index

        # Cls node attrs
        for attr, values in ontology.data.loc[go_nodes][["name", "namespace", "def"]].iteritems():
            self.G[go_ntype][attr] = values.to_numpy()

        self.G[go_ntype]['nid'] = torch.arange(len(go_nodes), dtype=torch.long)
        self.G[go_ntype].num_nodes = len(go_nodes)

        # Edges between RNA nodes and GO terms
        train_go_ann, valid_go_ann, test_go_ann = ontology.annotation_train_val_test_split(
            train_date=train_date, valid_date=valid_date, groupby=["gene_name"])

        for go_ann in [train_go_ann, valid_go_ann, test_go_ann]:
            triples = {}
            # Positive links
            nx_graph = nx.from_pandas_edgelist(go_ann["go_id"].dropna().explode().to_frame().reset_index(),
                                               source="gene_name", target="go_id", create_using=nx.DiGraph)

            metapath = (self.head_node_type, "associated", go_ntype)
            triples[metapath] = get_edge_index(nx_graph, nodes_A=self.nodes[metapath[0]], nodes_B=go_nodes)

            outputs = triples[metapath]

    def train_dataloader(self, collate_fn=None, batch_size=128, num_workers=10, **kwargs):
        if self.neighbor_loader == "NeighborLoader":
            Loader = NeighborLoader
        elif self.neighbor_loader == "HGTLoader":
            Loader = HGTLoader

        dataset = Loader(self.G,
                         num_neighbors=self.num_neighbors,
                         batch_size=batch_size,
                         # directed=True,
                         transform=self.sample,
                         input_nodes=(self.head_node_type, self.G[self.head_node_type].train_mask),
                         shuffle=True,
                         num_workers=num_workers,
                         **kwargs)

        return dataset

    def valid_dataloader(self, collate_fn=None, batch_size=128, num_workers=5, **kwargs):
        if self.neighbor_loader == "NeighborLoader":
            Loader = NeighborLoader
        elif self.neighbor_loader == "HGTLoader":
            Loader = HGTLoader

        dataset = NeighborLoader(self.G, num_neighbors=self.num_neighbors,
                                 batch_size=batch_size,
                                 # directed=False,
                                 transform=self.sample,
                                 input_nodes=(self.head_node_type, self.G[self.head_node_type].valid_mask),
                                 shuffle=False, num_workers=num_workers, **kwargs)

        return dataset

    def test_dataloader(self, collate_fn=None, batch_size=128, num_workers=5, **kwargs):
        if self.neighbor_loader == "NeighborLoader":
            Loader = NeighborLoader
        elif self.neighbor_loader == "HGTLoader":
            Loader = HGTLoader

        dataset = Loader(self.G, num_neighbors=self.num_neighbors,
                         batch_size=batch_size,
                         # directed=False,
                         transform=self.sample,
                         input_nodes=(self.head_node_type, self.G[self.head_node_type].test_mask),
                         shuffle=False, num_workers=num_workers, **kwargs)

        return dataset
