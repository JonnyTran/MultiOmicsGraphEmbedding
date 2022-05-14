from pprint import pprint
from typing import List, Tuple, Union, Dict

import networkx as nx
import numpy as np
import pandas as pd
import torch
from openomics.database.ontology import GeneOntology
from torch import Tensor
from torch.utils.data import DataLoader
from torch_geometric.data import HeteroData
from torch_sparse.tensor import SparseTensor

from moge.dataset.PyG.neighbor_sampler import NeighborLoader, HGTLoader
from moge.dataset.graph import HeteroGraphDataset
from moge.dataset.sequences import SequenceTokenizer
# from torch_geometric.loader import HGTLoader, NeighborLoader
from moge.dataset.utils import get_edge_index
from moge.model.PyG.utils import num_edges


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

    @classmethod
    def from_pyg_heterodata(cls, hetero: HeteroData, classes: List[str], nodes: Dict[str, pd.Index], **kwargs):
        self = cls(dataset=hetero, metapaths=hetero.edge_types, add_reverse_metapaths=False,
                   edge_dir="in", **kwargs)
        self.classes = classes
        self.nodes = nodes
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

    def add_ontology_edges(self, ontology, train_date='2017-06-15', valid_date='2017-11-15', go_ntype="GO_term"):
        all_go = set(ontology.network.nodes).intersection(ontology.data.index)

        # Order nodes with classes nodes first
        if self.classes is not None:
            go_nodes = np.concatenate([self.classes, np.array(list(set(all_go) - set(self.classes)))])
        else:
            go_nodes = np.array(list(all_go))

        # Edges between GO terms
        edge_types = {e for u, v, e in ontology.network.edges}
        edge_index_dict = ontology.to_scipy_adjacency(nodes=go_nodes, edge_types=edge_types,
                                                      reverse=True,
                                                      format="pyg", d_ntype=go_ntype)
        for metapath, edge_index in edge_index_dict.items():
            if edge_index.size(1) < 200: continue
            self.G[metapath].edge_index = edge_index
            self.metapaths.append(metapath)

        # Cls node attrs
        for attr, values in ontology.data.loc[go_nodes][["name", "namespace", "def"]].iteritems():
            self.G[go_ntype][attr] = values.to_numpy()

        self.G[go_ntype]['nid'] = torch.arange(len(go_nodes), dtype=torch.long)
        self.G[go_ntype].num_nodes = len(go_nodes)
        self.num_nodes_dict[go_ntype] = len(go_nodes)
        self.node_types.append(go_ntype)

        self.nodes[go_ntype] = pd.Index(go_nodes)

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

    def create_graph_sampler(self, batch_size: int, node_mask: Tensor, transform_fn=None, num_workers=10, **kwargs):
        if self.neighbor_loader == "NeighborLoader":
            self.num_neighbors = {
                etype: self.neighbor_sizes \
                # if etype[1] != 'associated' else [-1, ] * len(self.neighbor_sizes)
                for etype in self.metapaths}

        elif self.neighbor_loader == "HGTLoader":
            self.num_neighbors = {
                ntype: self.neighbor_sizes \
                    if ntype != 'GO_term' else [self.num_nodes_dict["GO_term"], ] * len(self.neighbor_sizes)
                for ntype in self.node_types}

        print(f"{self.neighbor_loader} neighbor_sizes:")
        pprint(self.num_neighbors, width=300)

        if self.neighbor_loader == "NeighborLoader":
            Loader = NeighborLoader
        elif self.neighbor_loader == "HGTLoader":
            Loader = HGTLoader

        dataset = Loader(self.G,
                         num_neighbors=self.num_neighbors,
                         batch_size=batch_size,
                         # directed=True,
                         transform=transform_fn,
                         input_nodes=(self.head_node_type, node_mask),
                         shuffle=True,
                         num_workers=num_workers,
                         **kwargs)
        return dataset

    def transform(self, batch: HeteroData):
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
                if not hasattr(batch[ntype], "sequence"): continue
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
        dataset = self.create_graph_sampler(batch_size, node_mask=self.G[self.head_node_type].train_mask,
                                            transform_fn=self.transform, num_workers=num_workers)

        return dataset

    def valid_dataloader(self, collate_fn=None, batch_size=128, num_workers=5, **kwargs):
        dataset = self.create_graph_sampler(batch_size, node_mask=self.G[self.head_node_type].valid_mask,
                                            transform_fn=self.transform, num_workers=num_workers)

        return dataset

    def test_dataloader(self, collate_fn=None, batch_size=128, num_workers=5, **kwargs):
        dataset = self.create_graph_sampler(batch_size, node_mask=self.G[self.head_node_type].test_mask,
                                            transform_fn=self.transform, num_workers=num_workers)

        return dataset

    def split_edge_index_by_go_namespace(self, edge_index_dict: Dict[Tuple[str, str, str], Tensor],
                                         batch_to_global: Dict[str, Tensor] = None,
                                         edge_pos: Dict[Tuple[str, str, str], Tensor] = None,
                                         mode=None) -> Dict[Tuple[str, str, str], Tensor]:
        out_edge_index_dict = {}

        for metapath, edge_index in edge_index_dict.items():
            tail_type = metapath[-1]

            if edge_index.size(0) == 2:
                if batch_to_global is not None:
                    go_terms = batch_to_global[tail_type][edge_index[1]]
                else:
                    go_terms = edge_index[1]

                go_namespaces = self.go_namespace[go_terms]

                for namespace in np.unique(go_namespaces):
                    mask = go_namespaces == namespace
                    out_edge_index_dict[metapath[:-1] + (namespace,)] = edge_index[:, mask]

            elif mode == "head_batch":
                if batch_to_global is not None:
                    go_terms = batch_to_global[tail_type][edge_pos[metapath][1]]
                else:
                    go_terms = edge_pos[metapath][1]

                go_namespaces = self.go_namespace[go_terms]

                for namespace in np.unique(go_namespaces):
                    mask = go_namespaces == namespace
                    out_edge_index_dict[metapath[:-1] + (namespace,)] = edge_index[mask]

        return out_edge_index_dict


class HeteroLinkPredDataset(HeteroNodeClfDataset):
    def __init__(self, dataset: HeteroData,
                 pred_metapaths: List[Tuple[str, str, str]] = [],
                 negative_sampling_size=10,
                 seq_tokenizer: SequenceTokenizer = None,
                 neighbor_loader: str = "NeighborLoader",
                 neighbor_sizes: Union[List[int], Dict[str, List[int]]] = [128, 128], node_types: List[str] = None,
                 metapaths: List[Tuple[str, str, str]] = None, head_node_type: str = None, edge_dir: str = "in",
                 reshuffle_train: float = None, add_reverse_metapaths: bool = True, inductive: bool = False, **kwargs):
        super().__init__(dataset, seq_tokenizer, neighbor_loader, neighbor_sizes, node_types, metapaths, head_node_type,
                         edge_dir, reshuffle_train, add_reverse_metapaths, inductive, **kwargs)
        self.negative_sampling_size = negative_sampling_size
        self.pred_metapaths = pred_metapaths
        self.neighbor_sizes = neighbor_sizes
        self.multilabel = False

        self.graph_sampler = self.create_graph_sampler(batch_size=1,
                                                       node_mask=torch.ones(self.G[self.head_node_type].num_nodes),
                                                       transform_fn=super().transform,
                                                       num_workers=0)

    def add_ontology_edges(self, ontology: GeneOntology, train_date='2017-06-15', valid_date='2017-11-15',
                           go_ntype="GO_term", metapaths: List[Tuple[str, str, str]] = None):
        all_go = set(ontology.network.nodes).intersection(ontology.data.index)
        go_nodes = np.array(list(all_go))
        self.go_ntype = go_ntype

        # Edges between GO terms
        edge_types = {e for u, v, e in ontology.network.edges}

        edge_index_dict = ontology.to_scipy_adjacency(nodes=go_nodes, edge_types=edge_types,
                                                      reverse=True,
                                                      format="pyg", d_ntype=go_ntype)
        for metapath, edge_index in edge_index_dict.items():
            if edge_index.size(1) < 100: continue
            self.G[metapath].edge_index = edge_index
            self.metapaths.append(metapath)

        # Cls node attrs
        for attr, values in ontology.data.loc[go_nodes][["name", "namespace", "def"]].iteritems():
            self.G[go_ntype][attr] = values.to_numpy()

        self.G[go_ntype]['nid'] = torch.arange(len(go_nodes), dtype=torch.long)
        self.G[go_ntype].num_nodes = len(go_nodes)
        self.num_nodes_dict[go_ntype] = len(go_nodes)
        self.node_types.append(go_ntype)
        self.go_namespace = self.G[go_ntype].namespace

        # Set sequence
        self.nodes[go_ntype] = pd.Index(go_nodes)
        self.G[go_ntype]["sequence"] = pd.Series(self.G[go_ntype]["name"] + ":" + self.G[go_ntype]["def"],
                                                 index=self.nodes[go_ntype])

        # Edges between RNA nodes and GO terms
        train_go_ann, valid_go_ann, test_go_ann = ontology.annotation_train_val_test_split(
            train_date=train_date, valid_date=valid_date, groupby=["gene_name"])

        self.triples_pos = {}
        self.triples_neg = {}
        pos_train_valid_test_sizes = []
        neg_train_valid_test_sizes = []
        metapath = (self.head_node_type, "associated", go_ntype)
        for go_ann in [train_go_ann, valid_go_ann, test_go_ann]:
            # True Positive links (undirected)
            nx_graph = nx.from_pandas_edgelist(go_ann["go_id"].dropna().explode().to_frame().reset_index(),
                                               source="gene_name", target="go_id", create_using=nx.Graph)

            edge_index = get_edge_index(nx_graph, nodes_A=self.nodes[metapath[0]], nodes_B=go_nodes)
            pos_train_valid_test_sizes.append(edge_index.size(1))
            self.triples_pos.setdefault(metapath, []).append(edge_index)

            # True Negative links (undirected)
            nx_graph = nx.from_pandas_edgelist(go_ann["neg_go_id"].dropna().explode().to_frame().reset_index(),
                                               source="gene_name", target="neg_go_id", create_using=nx.Graph)

            edge_index = get_edge_index(nx_graph, nodes_A=self.nodes[metapath[0]], nodes_B=go_nodes)
            self.triples_neg.setdefault(metapath, []).append(edge_index)
            neg_train_valid_test_sizes.append(edge_index.size(1))

        print("pos_train_valid_test_sizes", pos_train_valid_test_sizes)
        print("neg_train_valid_test_sizes", neg_train_valid_test_sizes)
        self.pred_metapaths.append(metapath)

        self.triples_pos = {metapath: torch.cat(li_edge_index, dim=1) \
                            for metapath, li_edge_index in self.triples_pos.items()}
        self.triples_neg = {metapath: torch.cat(li_edge_index, dim=1) \
                            for metapath, li_edge_index in self.triples_neg.items()}

        self.triples_pos_adj = {metapath: SparseTensor.from_edge_index(edge_index,
                                                                       sparse_sizes=(len(self.nodes[metapath[0]]),
                                                                                     len(self.nodes[metapath[-1]]))) \
                                for metapath, edge_index in self.triples_pos.items()}

        # Train/valid/test positive edges
        self.training_idx = torch.arange(0, pos_train_valid_test_sizes[0])
        self.validation_idx = torch.arange(pos_train_valid_test_sizes[0],
                                           pos_train_valid_test_sizes[0] + pos_train_valid_test_sizes[1])
        self.testing_idx = torch.arange(pos_train_valid_test_sizes[0] + pos_train_valid_test_sizes[1],
                                        pos_train_valid_test_sizes[0] + pos_train_valid_test_sizes[1] + \
                                        pos_train_valid_test_sizes[2])

        # Train/valid/test positive edges
        self.training_idx_neg = torch.arange(0, neg_train_valid_test_sizes[0])
        self.validation_idx_neg = torch.arange(neg_train_valid_test_sizes[0],
                                               neg_train_valid_test_sizes[0] + neg_train_valid_test_sizes[1])
        self.testing_idx_neg = torch.arange(neg_train_valid_test_sizes[0] + neg_train_valid_test_sizes[1],
                                            neg_train_valid_test_sizes[0] + neg_train_valid_test_sizes[1] + \
                                            neg_train_valid_test_sizes[2])

        # Reinstantiate graph sampler since hetero graph was modified
        self.graph_sampler = self.create_graph_sampler(batch_size=1,
                                                       node_mask=torch.ones(self.G[self.head_node_type].num_nodes),
                                                       transform_fn=super().transform,
                                                       num_workers=0)

    def get_prior(self):
        pos_count = sum([edge_index.size(1) for edge_index in self.triples_pos.values()])
        neg_count = sum([edge_index.size(1) for edge_index in self.triples_neg.values()])

        return torch.tensor(pos_count / (pos_count + neg_count))

    @staticmethod
    def get_relabled_edge_index(edge_index_dict: Dict[Tuple[str, str, str], Tensor],
                                global_node_index: Dict[str, Tensor],
                                local2batch: Dict[str, Tensor] = None,
                                mode=None) -> Dict[Tuple[str, str, str], Tensor]:
        if local2batch is None:
            local2batch = {
                node_type: dict(zip(
                    global_node_index[node_type].numpy(),
                    range(len(global_node_index[node_type])))
                ) for node_type in global_node_index}

        renamed_edge_index_dict = {}
        for metapath, edge_index in edge_index_dict.items():
            head_type, edge_type, tail_type = metapath

            if edge_index.size(0) == 2 and mode is None:
                sources = edge_index[0].apply_(local2batch[head_type].get)
                targets = edge_index[-1].apply_(local2batch[tail_type].get)

                renamed_edge_index_dict[metapath] = torch.stack([sources, targets], dim=1).t()

            elif mode == "head_batch":
                renamed_edge_index_dict[metapath] = edge_index.apply_(local2batch[head_type].get)
            elif mode == "tail_batch":
                renamed_edge_index_dict[metapath] = edge_index.apply_(local2batch[tail_type].get)
            else:
                raise Exception("Must provide an edge_index with shape (2, N) or pass 'head_batch' or 'tail_batch' "
                                "in `mode`.")

        return renamed_edge_index_dict

    def transform(self, edge_idx: List[int], mode=None):
        if not isinstance(edge_idx, torch.LongTensor):
            edge_idx = torch.LongTensor(edge_idx)

        edge_pos = {metapath: self.triples_pos[metapath][:, edge_idx] for metapath in self.triples_pos}

        # True negative edges
        edge_neg = {metapath: self.triples_neg[metapath][:, self.training_idx_neg if mode == "train" else \
                                                                self.validation_idx_neg if mode == "valid" else \
                                                                    self.testing_idx_neg if mode == "test" else []] \
                    for metapath in self.triples_neg}

        # If ensures same number of true neg edges to true pos edges
        if num_edges(edge_neg) > edge_idx.numel():
            edge_neg = {metapath: edge_index[:, torch.multinomial(torch.ones(edge_index.size(1)),
                                                                  num_samples=edge_idx.numel(),
                                                                  replacement=False)] \
                        for metapath, edge_index in edge_neg.items()}

        # Get all nodes induced by sampled edges
        if num_edges(edge_neg):
            query_edges = {metapath: torch.cat([edge_pos[metapath], edge_neg[metapath]], dim=1) \
                           for metapath in edge_pos}
        else:
            query_edges = edge_pos
        query_nodes = self.gather_node_set(query_edges)

        # Get subgraph induced by neighborhood hopping from the query nodes
        X, _, _ = self.graph_sampler.transform_fn(self.graph_sampler.collate_fn(query_nodes))

        # Edge_pos must be global index, not batch index
        edge_pos_split = self.split_edge_index_by_go_namespace(edge_pos, batch_to_global=None)
        head_batch, tail_batch = self.generate_negative_sampling(
            edge_pos_split,
            global_node_index=X["global_node_index"])

        # Rename node index from global to batch
        local2batch = {
            ntype: dict(zip(
                X["global_node_index"][ntype].numpy(),
                range(len(X["global_node_index"][ntype])))
            ) for ntype in X["global_node_index"]}

        edge_pos = self.get_relabled_edge_index(edge_pos, global_node_index=X["global_node_index"],
                                                local2batch=local2batch)
        edge_pos = self.split_edge_index_by_go_namespace(edge_pos, batch_to_global=X["global_node_index"])

        y = {"edge_pos": edge_pos, }
        if num_edges(edge_neg):
            edge_neg = self.get_relabled_edge_index(edge_neg, global_node_index=X["global_node_index"],
                                                    local2batch=local2batch, )
            y['edge_neg'] = self.split_edge_index_by_go_namespace(edge_neg, batch_to_global=X["global_node_index"])

        # Negative sampling
        y.update({"head_batch": head_batch, "tail_batch": tail_batch, })

        edge_weights = None
        return X, y, edge_weights

    def generate_negative_sampling(self, edge_pos: Dict[Tuple[str, str, str], Tensor],
                                   global_node_index: Dict[str, Tensor]) -> \
            Tuple[Dict[Tuple[str, str, str], Tensor], Dict[Tuple[str, str, str], Tensor]]:
        head_batch = {}
        tail_batch = {}

        for metapath, edge_index in edge_pos.items():
            head_type, tail_type = metapath[0], metapath[-1]
            adj: SparseTensor = self.triples_pos_adj[metapath] if metapath in self.triples_pos_adj \
                else self.triples_pos_adj[metapath[:-1] + (self.go_ntype,)]

            head_neg_nodes = global_node_index[head_type]
            head_prob_dist = 1 - adj[head_neg_nodes, edge_index[1]].to_dense().T
            head_batch[metapath] = torch.multinomial(head_prob_dist, num_samples=self.negative_sampling_size // 2,
                                                     replacement=True)

            tail_neg_nodes = global_node_index[tail_type if tail_type in global_node_index else self.go_ntype]
            tail_prob_dist = 1 - adj[edge_index[0], tail_neg_nodes].to_dense()
            for go_type in ['biological_process', 'cellular_component', 'molecular_function']:
                # Only generate negative tail_batch within BPO, CCO, or MFO terms of the positive edge's tail go_type
                if go_type != tail_type: continue
                go_terms_mask = self.go_namespace[tail_neg_nodes] != go_type
                tail_prob_dist[:, go_terms_mask] = 0

            tail_batch[metapath] = torch.multinomial(tail_prob_dist, num_samples=self.negative_sampling_size // 2,
                                                     replacement=True)

        return head_batch, tail_batch

    def gather_node_set(self, edge_index_dict: Dict[Tuple[str, str, str], Tensor]) -> Dict[str, Tensor]:
        nodes = {}
        for metapath, edge_index in edge_index_dict.items():
            nodes.setdefault(metapath[0], []).append(edge_index[0])
            nodes.setdefault(metapath[-1], []).append(edge_index[1])

        nodes = {ntype: torch.unique(torch.cat(nids, dim=0)) for ntype, nids in nodes.items()}

        return nodes

    def train_dataloader(self, collate_fn=None, batch_size=128, num_workers=10, **kwargs):
        dataset = DataLoader(self.training_idx, batch_size=batch_size,
                             collate_fn=lambda idx: self.transform(idx, mode="train"),
                             shuffle=True,
                             num_workers=num_workers)
        return dataset

    def valid_dataloader(self, collate_fn=None, batch_size=128, num_workers=5, **kwargs):
        dataset = DataLoader(self.validation_idx, batch_size=batch_size,
                             collate_fn=lambda idx: self.transform(idx, mode="valid"),
                             shuffle=False,
                             num_workers=num_workers)
        return dataset

    def test_dataloader(self, collate_fn=None, batch_size=128, num_workers=5, **kwargs):
        dataset = DataLoader(self.testing_idx, batch_size=batch_size,
                             collate_fn=lambda idx: self.transform(idx, mode="test"),
                             shuffle=False,
                             num_workers=num_workers)
        return dataset
