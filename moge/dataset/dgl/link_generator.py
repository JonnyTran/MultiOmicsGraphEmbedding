from typing import List, Union, Optional

import dgl
import numpy as np
import torch
from dgl.dataloading import EdgePredictionSampler
from dgl.heterograph import DGLBlock, DGLHeteroGraph
from ogb.linkproppred import DglLinkPropPredDataset
from pandas import Index
from torch import Tensor

from .node_generator import DGLNodeSampler
from ..utils import get_relabled_edge_index, is_negative, is_reversed, unreverse_metapath
from ...network.base import SEQUENCE_COL
from ...network.hetero import HeteroNetwork


class DGLLinkSampler(DGLNodeSampler):
    def __init__(self, dataset: Union[DglLinkPropPredDataset, DGLHeteroGraph], sampler: str, neighbor_sizes=None,
                 negative_sampler="uniform", negative_sampling_size=100, pred_metapaths=None, neg_pred_metapaths=None,
                 node_types=None, metapaths=None, head_node_type=None, edge_dir=True, reshuffle_train: float = None,
                 add_reverse_metapaths=True, inductive=True, **kwargs):
        super().__init__(dataset, sampler=sampler, neighbor_sizes=neighbor_sizes, node_types=node_types,
                         metapaths=metapaths, head_node_type=head_node_type, edge_dir=edge_dir,
                         reshuffle_train=reshuffle_train, add_reverse_metapaths=add_reverse_metapaths,
                         inductive=inductive, **kwargs)

        self.negative_sampling_size = negative_sampling_size
        self.eval_negative_sampling_size = 1000

        self.pred_metapaths = pred_metapaths if pred_metapaths else []
        self.neg_pred_metapaths = neg_pred_metapaths if neg_pred_metapaths else []

        self.negative_sampler = negative_sampler
        self.link_sampler = self.get_link_sampler(self.G, negative_sampling_size=negative_sampling_size,
                                                  negative_sampler=negative_sampler, neighbor_sizes=neighbor_sizes,
                                                  neighbor_sampler=sampler, edge_dir=edge_dir)

    def get_link_sampler(self, G: DGLHeteroGraph, negative_sampling_size: int, negative_sampler: str,
                         neighbor_sizes: List[str], neighbor_sampler: str, edge_dir="in"):
        neighbor_sampler = self.get_neighbor_sampler(G, neighbor_sizes=neighbor_sizes, sampler=neighbor_sampler,
                                                     edge_dir=edge_dir)

        if negative_sampler.lower() == "Uniform".lower():
            Sampler = dgl.dataloading.negative_sampler.Uniform
        elif negative_sampler.lower() == "GlobalUniform".lower():
            Sampler = dgl.dataloading.negative_sampler.GlobalUniform
        elif negative_sampler.lower() == "PerSourceUniform".lower():
            Sampler = dgl.dataloading.negative_sampler.PerSourceUniform
        else:
            raise Exception(f"{negative_sampler} must be in one of [Uniform, GlobalUniform, PerSourceUniform]")
        link_sampler = EdgePredictionSampler(sampler=neighbor_sampler,
                                             exclude="reverse_types" if self.use_reverse else "self",
                                             reverse_etypes=self.reverse_etypes if self.use_reverse else None,
                                             reverse_eids=self.reverse_eids if self.use_reverse else None,
                                             negative_sampler=Sampler(negative_sampling_size))

        return link_sampler

    @classmethod
    def from_heteronetwork(cls, network: HeteroNetwork, node_attr_cols: List[str] = None,
                           target: str = None, min_count: int = None,
                           expression=False, sequence=False, add_reverse_metapaths=True,
                           label_subset: Optional[Union[Index, np.ndarray]] = None,
                           ntype_subset: Optional[List[str]] = None, split_namespace=False, **kwargs):
        G, classes, nodes, training_idx, validation_idx, testing_idx = \
            network.to_dgl_heterograph(node_attr_cols=node_attr_cols, target=target, min_count=min_count,
                                       expression=expression, sequence=sequence,
                                       label_subset=label_subset, ntype_subset=ntype_subset)

        self = cls(dataset=G, metapaths=G.canonical_etypes, add_reverse_metapaths=add_reverse_metapaths,
                   edge_dir="in", **kwargs)
        self.network = network
        self.classes = classes
        self.nodes = nodes
        self._name = network._name if hasattr(network, '_name') else ""

        self.split_namespace = split_namespace
        if split_namespace:
            for ntype in self.G.node_types:
                if "namespace" in self.G[ntype]:
                    self.go_namespace = self.G[ntype]["namespace"]
                    self.ntype_mapping = {namespace: ntype for namespace in np.unique(self.go_namespace)}

        self.training_idx, self.validation_idx, self.testing_idx = training_idx, validation_idx, testing_idx

        if hasattr(network, 'pred_metapaths') and not set(network.pred_metapaths).issubset(self.pred_metapaths):
            self.pred_metapaths.extend(network.pred_metapaths)
        if hasattr(network, 'neg_pred_metapaths') and not set(network.neg_pred_metapaths).issubset(
                self.neg_pred_metapaths):
            self.neg_pred_metapaths.extend(network.neg_pred_metapaths)

        if self.use_reverse:
            for metapath in self.G.canonical_etypes:
                if is_negative(metapath) and is_reversed(metapath):
                    print("removed", metapath)
                    self.G.remove_edges(eids=self.G.edges(etype=metapath, form='eid'), etype=metapath)

                elif self.pred_metapaths and unreverse_metapath(metapath) in self.pred_metapaths:
                    print("removed", metapath)
                    self.G.remove_edges(eids=self.G.edges(etype=metapath, form='eid'), etype=metapath)

                elif self.neg_pred_metapaths and unreverse_metapath(metapath) in self.pred_metapaths:
                    print("removed", metapath)
                    self.G.remove_edges(eids=self.G.edges(etype=metapath, form='eid'), etype=metapath)

            self.metapaths = [metapath for metapath in self.G.canonical_etypes if self.G.num_edges(etype=metapath)]

        return self

    def process_dgl_heterodata(self, graph: dgl.DGLHeteroGraph):
        self.G = graph

        self.node_types = graph.ntypes

        self.num_nodes_dict = {ntype: graph.num_nodes(ntype) for ntype in graph.ntypes}
        self.global_node_index = {ntype: torch.arange(graph.num_nodes(ntype)) for ntype in graph.ntypes}

        self.x_dict = graph.ndata["feat"]

        self.y_dict = {}
        for ntype, labels in self.y_dict.items():
            if labels.dim() == 2 and labels.shape[1] == 1:
                labels = labels.squeeze(1)
            graph.nodes[ntype].data["labels"] = labels

        self.metapaths = graph.canonical_etypes

    def process_DglLinkDataset_hetero(self, dataset: DglLinkPropPredDataset):
        graph: dgl.DGLHeteroGraph = dataset[0]
        self._name = dataset.name

        if self.node_types is None:
            self.node_types = graph.ntypes

        self.num_nodes_dict = {ntype: graph.num_nodes(ntype) for ntype in graph.ntypes}
        self.global_node_index = {ntype: torch.arange(graph.num_nodes(ntype)) for ntype in graph.ntypes}
        self.x_dict = graph.ndata["feat"]

        self.y_dict = {}
        for ntype, labels in self.y_dict.items():
            if labels.dim() == 2 and labels.shape[1] == 1:
                labels = labels.squeeze(1)
            graph.nodes[ntype].data["labels"] = labels

        if self.head_node_type is None:
            if self.y_dict:
                self.head_node_type = list(self.y_dict.keys())[0]
            else:
                self.head_node_type = self.node_types[0]

        self.metapaths = graph.canonical_etypes
        self.G_train = graph.__copy__()

        # Train/valid/test split of triples
        split_edge = dataset.get_edge_split()
        train_triples, valid_triples, test_triples = split_edge["train"], split_edge["valid"], split_edge["test"]

        self.triples = {}
        for key in train_triples.keys():
            if isinstance(train_triples[key], torch.Tensor):
                self.triples[key] = torch.cat([valid_triples[key], test_triples[key], train_triples[key]], dim=0)
            else:
                self.triples[key] = np.array(valid_triples[key] + test_triples[key] + train_triples[key])

        for key in valid_triples.keys():
            if is_negative(key):  # either head_neg or tail_neg
                self.triples[key] = torch.cat([valid_triples[key], test_triples[key]], dim=0)

        # Get index of train/valid/test edge_id
        self.training_idx, self.validation_idx, self.testing_idx = {}, {}, {}
        for triples, trainvalidtest_idx in zip([train_triples, valid_triples, test_triples],
                                               [self.training_idx, self.validation_idx, self.testing_idx]):
            edges_pos, edges_neg = get_relabled_edge_index(triples, global_node_index=self.global_node_index,
                                                           metapaths=self.metapaths, format="dgl")

            # Add valid and test edges to graph, since graph only contains train edges
            for metapath, (src, dst) in edges_pos.items():
                if triples is not train_triples:
                    graph.add_edges(src, dst, etype=metapath)
                trainvalidtest_idx[metapath] = graph.edge_ids(src, dst, etype=metapath)
            print(sum([len(eids) for eids in trainvalidtest_idx.values()]))

        self.G = graph

    def get_metapaths(self):
        return self.metapaths

    def get_train_ratio(self):
        n_train_edges = sum([len(eids) for eids in self.training_idx.values()])
        n_valid_edges = sum([len(eids) for eids in self.validation_idx.values()])
        n_test_edges = sum([len(eids) for eids in self.testing_idx.values()])
        return n_train_edges / (n_train_edges + n_valid_edges + n_test_edges)

    def get_collate_fn(self, collate_fn: str, mode=None):
        raise NotImplementedError()

    def sample(self, iloc, mode):
        raise NotImplementedError()

    def full_batch(self, edge_idx: Tensor = None, mode="test", device="cpu"):
        if edge_idx is None:
            edge_idx = {m: torch.cat([self.training_idx[m], self.validation_idx[m], self.testing_idx[m]]) \
                        for m in self.training_idx.keys()}

        total_batch_size = sum(eid.numel() for eid in edge_idx.values())

        if mode == "test":
            loader = self.test_dataloader(batch_size=total_batch_size, indices=edge_idx)
        else:
            loader = self.train_dataloader(batch_size=total_batch_size, indices=edge_idx)

        if device != "cpu":
            pass
        return next(iter(loader))

    def train_dataloader(self, collate_fn=None, batch_size=128, num_workers=0, indices=None, **kwargs):
        if self.inductive:
            graph = self.transform_heterograph(self.G, edge_mask=self.G.edata["train_mask"])
            if indices is None:
                indices = {etype: torch.arange(graph.num_edges(etype=etype)) for etype in
                           self.pred_metapaths + self.neg_pred_metapaths}

            graph_sampler = self.get_link_sampler(graph, negative_sampling_size=self.negative_sampling_size,
                                                  negative_sampler=self.negative_sampler,
                                                  neighbor_sizes=self.neighbor_sizes,
                                                  neighbor_sampler=self.sampler, edge_dir=self.edge_dir)
        else:
            graph = self.G
            if indices is None:
                indices = self.training_idx
            graph_sampler = self.link_sampler

        # sampler = LinkPredPyGCollator(**args)
        dataloader = dgl.dataloading.DataLoader(graph, indices=indices, graph_sampler=graph_sampler,
                                                batch_size=batch_size, shuffle=True, drop_last=False,
                                                num_workers=num_workers)

        return dataloader

    def valid_dataloader(self, collate_fn=None, batch_size=128, num_workers=0, indices=None, **kwargs):
        if self.inductive:
            edge_mask = {etype: valid_mask | self.G.edata["train_mask"][etype] \
                         for etype, valid_mask in self.G.edata["valid_mask"].items()}
            graph = self.transform_heterograph(self.G, edge_mask=edge_mask)
            if indices is None:
                indices = {etype: torch.arange(graph.num_edges(etype=etype)) for etype in
                           self.pred_metapaths + self.neg_pred_metapaths}

            graph_sampler = self.get_link_sampler(graph, negative_sampling_size=self.negative_sampling_size,
                                                  negative_sampler=self.negative_sampler,
                                                  neighbor_sizes=self.neighbor_sizes,
                                                  neighbor_sampler=self.sampler, edge_dir=self.edge_dir)
        else:
            graph = self.G
            if indices is None:
                indices = self.validation_idx
            graph_sampler = self.link_sampler

        dataloader = dgl.dataloading.DataLoader(graph, indices=indices, graph_sampler=graph_sampler,
                                                batch_size=batch_size, shuffle=False, drop_last=False,
                                                num_workers=num_workers)
        return dataloader

    def test_dataloader(self, collate_fn=None, batch_size=128, num_workers=4, indices=None, **kwargs):
        if self.inductive:
            edge_mask = {etype: test_mask | self.G.edata["train_mask"][etype] | self.G.edata["valid_mask"][etype] \
                         for etype, test_mask in self.G.edata["test_mask"].items()}
            graph = self.transform_heterograph(self.G, edge_mask=edge_mask)
            if indices is None:
                indices = {etype: torch.arange(graph.num_edges(etype=etype)) for etype in
                           self.pred_metapaths + self.neg_pred_metapaths}

            graph_sampler = self.get_link_sampler(graph, negative_sampling_size=self.negative_sampling_size,
                                                  negative_sampler=self.negative_sampler,
                                                  neighbor_sizes=self.neighbor_sizes,
                                                  neighbor_sampler=self.sampler, edge_dir=self.edge_dir)
        else:
            graph = self.G
            if indices is None:
                indices = self.testing_idx
            graph_sampler = self.link_sampler

        dataloader = dgl.dataloading.DataLoader(graph, indices=indices, graph_sampler=graph_sampler,
                                                batch_size=batch_size, shuffle=False, drop_last=False,
                                                num_workers=num_workers)
        return dataloader


class LinkPredPyGCollator(EdgePredictionSampler):
    def __init__(self, sampler, exclude=None, reverse_eids=None, reverse_etypes=None, negative_sampler=None,
                 prefetch_labels=None, seq_tokenizer=None):
        if seq_tokenizer is not None:
            self.seq_tokenizer = seq_tokenizer
        super().__init__(sampler, exclude, reverse_eids, reverse_etypes, negative_sampler, prefetch_labels)

    def sample(self, g, seed_edges):
        input_nodes, pos_graph, neg_graph, blocks = super().sample(g, seed_edges)
        blocks: List[DGLBlock]

        X = {}
        for i, block in enumerate(blocks):
            edge_index_dict = {}
            for metapath in block.canonical_etypes:
                if block.num_edges(etype=metapath) == 0:
                    continue
                edge_index_dict[metapath] = torch.stack(block.edges(etype=metapath, order="srcdst"), dim=0)
            X.setdefault("edge_index", []).append(edge_index_dict)

            sizes = {}
            for ntype in block.ntypes:
                sizes[ntype] = torch.tensor((block.num_src_nodes(ntype), block.num_dst_nodes(ntype)))
            X.setdefault("sizes", []).append(sizes)

            global_node_index = {ntype: nid for ntype, nid in block.srcdata[dgl.NID].items() if nid.numel() > 0}
            X.setdefault("global_node_index", []).append(global_node_index)

        X["x_dict"] = {ntype: feat \
                       for ntype, feat in blocks[0].srcdata["feat"].items() \
                       if feat.size(0) != 0}
        if len(X["x_dict"]) == 0:
            X.pop("x_dict")

        if SEQUENCE_COL in blocks[0].srcdata and len(blocks[0].srcdata[SEQUENCE_COL]):
            for ntype, feat in blocks[0].srcdata[SEQUENCE_COL].items():
                X.setdefault("sequences", {})[ntype] = self.seq_tokenizer.encode_sequences(X, ntype=ntype,
                                                                                           max_length=None)

        edges = {"edge_pos": {}, "edge_neg": {}}
        for etype in pos_graph.etypes:
            if pos_graph.num_edges(etype=etype) == 0: continue
            edges["edge_pos"][etype] = torch.stack(pos_graph.edges(etype=etype, order='srcdst'), dim=0)

        for etype in neg_graph.etypes:
            if neg_graph.num_edges(etype=etype) == 0: continue
            edges["edge_neg"][etype] = torch.stack(neg_graph.edges(etype=etype, order='srcdst'), dim=0)

        return X, edges, {}
