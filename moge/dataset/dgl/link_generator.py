import dgl
import numpy as np
import torch
from dgl.dataloading import EdgePredictionSampler
from ogb.linkproppred import DglLinkPropPredDataset

from .node_generator import DGLNodeSampler
from ..utils import get_relabled_edge_index
from ...model.PyG import is_negative
from ...network.base import SEQUENCE_COL


class DGLLinkSampler(DGLNodeSampler):
    def __init__(self, dataset: DglLinkPropPredDataset, sampler: str, neighbor_sizes=None,
                 negative_sampler="uniform", negative_sampling_size=100,
                 node_types=None, metapaths=None, head_node_type=None, edge_dir=True, reshuffle_train: float = None,
                 add_reverse_metapaths=True, inductive=True, ):
        super().__init__(dataset, sampler=sampler, neighbor_sizes=neighbor_sizes, node_types=node_types,
                         metapaths=metapaths, head_node_type=head_node_type, edge_dir=edge_dir,
                         reshuffle_train=reshuffle_train, add_reverse_metapaths=add_reverse_metapaths,
                         inductive=inductive)
        self.negative_sampling_size = negative_sampling_size
        self.eval_negative_sampling_size = 1000
        if negative_sampler.lower() == "uniform".lower():
            self.init_negative_sampler = dgl.dataloading.negative_sampler.Uniform
        elif negative_sampler.lower() == "GlobalUniform".lower():
            self.init_negative_sampler = dgl.dataloading.negative_sampler.GlobalUniform
        elif negative_sampler.lower() == "PerSourceUniform".lower():
            self.init_negative_sampler = dgl.dataloading.negative_sampler.PerSourceUniform

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

            # Add valid and test edges to graph
            for metapath, (src, dst) in edges_pos.items():
                if triples is not train_triples:
                    graph.add_edges(src, dst, etype=metapath)
                trainvalidtest_idx[metapath] = graph.edge_ids(src, dst, etype=metapath)
            print(sum([len(eids) for eids in trainvalidtest_idx.values()]))

        self.G = graph

    def get_metapaths(self):
        return self.G.canonical_etypes

    def get_train_ratio(self):
        n_train_edges = sum([len(eids) for eids in self.training_idx.values()])
        n_valid_edges = sum([len(eids) for eids in self.validation_idx.values()])
        n_test_edges = sum([len(eids) for eids in self.testing_idx.values()])
        return n_train_edges / (n_train_edges + n_valid_edges + n_test_edges)

    def get_collate_fn(self, collate_fn: str, mode=None):
        raise NotImplementedError()

    def sample(self, iloc, mode):
        raise NotImplementedError()

    def train_dataloader(self, collate_fn=None, batch_size=128, num_workers=0, **kwargs):
        if self.inductive:
            graph = self.G.edge_subgraph(self.training_idx, relabel_nodes=True, store_ids=True)
        else:
            graph = self.G

        sampler = dgl.dataloading.as_edge_prediction_sampler(
            self.neighbor_sampler, exclude="self",
            negative_sampler=self.init_negative_sampler(self.negative_sampling_size))

        dataloader = dgl.dataloading.DataLoader(graph, indices=self.training_idx, graph_sampler=sampler,
                                                batch_size=batch_size, shuffle=True, drop_last=False,
                                                num_workers=num_workers)

        return dataloader

    def valid_dataloader(self, collate_fn=None, batch_size=128, num_workers=0, **kwargs):
        graph = self.G

        sampler = dgl.dataloading.as_edge_prediction_sampler(
            self.neighbor_sampler, exclude="self",
            negative_sampler=self.init_negative_sampler(self.eval_negative_sampling_size))

        dataloader = dgl.dataloading.DataLoader(graph, indices=self.validation_idx, graph_sampler=sampler,
                                                batch_size=batch_size, shuffle=False, drop_last=False,
                                                num_workers=num_workers)
        return dataloader

    def test_dataloader(self, collate_fn=None, batch_size=128, num_workers=4, **kwargs):
        graph = self.G

        sampler = dgl.dataloading.as_edge_prediction_sampler(
            self.neighbor_sampler, exclude="self",
            negative_sampler=self.init_negative_sampler(self.eval_negative_sampling_size))

        dataloader = dgl.dataloading.DataLoader(graph, indices=self.training_idx, graph_sampler=sampler,
                                                batch_size=batch_size, shuffle=False, drop_last=False,
                                                num_workers=num_workers)
        return dataloader


class LATTELinkPredPyGCollator(EdgePredictionSampler):
    def __init__(self, sampler, exclude=None, reverse_eids=None, reverse_etypes=None, negative_sampler=None,
                 prefetch_labels=None, seq_tokenizer=None):
        if seq_tokenizer is not None:
            self.seq_tokenizer = seq_tokenizer
        super().__init__(sampler, exclude, reverse_eids, reverse_etypes, negative_sampler, prefetch_labels)

    def sample(self, g, seed_edges):
        input_nodes, pos_graph, neg_graph, blocks = super().sample(g, seed_edges)

        X = {}
        for i, block in enumerate(blocks):
            edge_index_dict = {}
            for metapath in block.canonical_etypes:
                if block.num_edges(etype=metapath) == 0:
                    continue
                edge_index_dict[metapath] = torch.stack(block.edges(etype=metapath), dim=0)

            X.setdefault("edge_index", []).append(edge_index_dict)

            size = {}
            for ntype in block.ntypes:
                size[ntype] = (None if block.num_src_nodes(ntype) == 0 else block.num_src_nodes(ntype),
                               None if block.num_dst_nodes(ntype) == 0 else block.num_dst_nodes(ntype))
            X.setdefault("sizes", []).append(size)

            X.setdefault("global_node_index", []).append(
                {ntype: nid for ntype, nid in block.srcdata[dgl.NID].items() if nid.numel() > 0})

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
            if pos_graph.num_edge(etype=etype) == 0: continue
            edges["edge_pos"][etype] = torch.stack(pos_graph.edges(etype=etype, order='srcdst'), dim=0)

        for etype in neg_graph.etypes:
            if neg_graph.num_edge(etype=etype) == 0: continue
            edges["edge_neg"][etype] = torch.stack(neg_graph.edges(etype=etype, order='srcdst'), dim=0)

        return X, edges, None
