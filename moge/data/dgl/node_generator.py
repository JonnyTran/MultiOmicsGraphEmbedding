from collections import defaultdict
import dgl
import numpy as np
import pandas as pd
import torch
from ogb.nodeproppred import DglNodePropPredDataset
from torch.utils.data import DataLoader

from dgl.dataloading import BlockSampler, NodeCollator
from dgl import convert, utils, batch
from dgl import backend as F
from dgl.dataloading.dataloader import _prepare_tensor_dict, _prepare_tensor
from dgl import utils as dglutils
from moge.data.network import HeteroNetDataset
from .samplers import ImportanceSampler, MultiLayerNeighborSampler


class DGLNodeSampler(HeteroNetDataset):
    def __init__(self, dataset: DglNodePropPredDataset,
                 sampler: str,
                 embedding_dim=None,
                 neighbor_sizes=None,
                 node_types=None,
                 metapaths=None,
                 head_node_type=None,
                 edge_dir=True,
                 reshuffle_train: float = None,
                 add_reverse_metapaths=True,
                 inductive=True):
        self.neighbor_sizes = neighbor_sizes
        self.embedding_dim = embedding_dim
        super().__init__(dataset, node_types=node_types, metapaths=metapaths, head_node_type=head_node_type,
                         edge_dir=edge_dir, reshuffle_train=reshuffle_train,
                         add_reverse_metapaths=add_reverse_metapaths, inductive=inductive)
        assert isinstance(self.G, dgl.DGLHeteroGraph)

        if add_reverse_metapaths:
            self.G = self.create_heterograph(self.G, add_reverse=True)

        self.degree_counts = self.compute_node_degrees(add_reverse_metapaths)

        if sampler is None:
            print("Using Full Multilayer sampler")
            self.neighbor_sampler = dgl.dataloading.MultiLayerFullNeighborSampler(n_layers=len(self.neighbor_sizes))

        elif sampler == "ImportanceSampler":
            self.neighbor_sampler = ImportanceSampler(fanouts=neighbor_sizes,
                                                      metapaths=self.get_metapaths(),  # Original metapaths only
                                                      degree_counts=self.degree_counts,
                                                      edge_dir=edge_dir)
        else:
            raise Exception("Use one of", ["ImportanceSampler"])

    def create_heterograph(self, g: dgl.DGLHeteroGraph, add_reverse=False):
        reversed_g = g.reverse(copy_edata=True, share_edata=True)

        relations = {}
        for metapath in g.canonical_etypes:
            # Original edges
            src, dst = g.all_edges(etype=metapath[1])
            relations[metapath] = (src, dst)

            # Reverse edges
            if add_reverse:
                reverse_metapath = self.reverse_metapath_name(metapath)
                assert reverse_metapath not in relations
                src, dst = reversed_g.all_edges(etype=metapath[1])
                relations[reverse_metapath] = (src, dst)

        new_g = dgl.heterograph(relations, num_nodes_dict=self.num_nodes_dict, idtype=torch.int64)

        # copy_ndata:
        for ntype in g.ntypes:
            for k, v in g.nodes[ntype].data.items():
                new_g.nodes[ntype].data[k] = v.detach().clone()

        node_frames = utils.extract_node_subframes(new_g,
                                                   nodes=[new_g.nodes(ntype) for ntype in new_g.ntypes],
                                                   store_ids=True)
        utils.set_new_frames(new_g, node_frames=node_frames)

        eids = []
        for metapath in new_g.canonical_etypes:
            eid = F.copy_to(F.arange(0, new_g.number_of_edges(metapath)), new_g.device)
            eids.append(eid)

        edge_frames = utils.extract_edge_subframes(new_g, eids, store_ids=True)
        utils.set_new_frames(new_g, edge_frames=edge_frames)

        return new_g

    def compute_node_degrees(self, add_reverse_metapaths):
        dfs = []
        for metapath in self.G.canonical_etypes:
            head_type, tail_type = metapath[0], metapath[-1]
            relation = self.get_metapaths().index(metapath)

            src, dst = self.G.all_edges(etype=metapath)

            df = pd.DataFrame()
            df["head"] = src.numpy()
            df["tail"] = dst.numpy()
            df["head_type"] = head_type
            df["relation"] = relation
            df["tail_type"] = tail_type

            dfs.append(df)

        df = pd.concat(dfs)

        head_counts = df.groupby(["head", "relation", "head_type"])["tail"].count().astype(float)

        if add_reverse_metapaths:
            head_counts.index = head_counts.index.set_names(["nid", "relation", "ntype"])
            return head_counts

        # For directed graphs, use both
        else:
            tail_counts = df.groupby(["tail", "relation", "tail_type"])["head"].count().astype(float)
            tail_counts.index = tail_counts.index.set_levels(levels=-tail_counts.index.get_level_values(1) - 1,
                                                             level=1,
                                                             verify_integrity=False, )
            head_counts.index = head_counts.index.set_names(["nid", "relation", "ntype"])
            tail_counts.index = tail_counts.index.set_names(["nid", "relation", "ntype"])
            return head_counts.append(tail_counts)  # (node_id, relation, ntype): count

    def process_DglNodeDataset_hetero(self, dataset: DglNodePropPredDataset):
        graph, labels = dataset[0]
        self._name = dataset.name
        self.G = graph

        if self.node_types is None:
            self.node_types = graph.ntypes

        self.num_nodes_dict = {ntype: graph.num_nodes(ntype) for ntype in self.node_types}
        self.y_dict = labels

        # Process labels
        for ntype, labels in self.y_dict.items():
            if labels.dim() == 2 and labels.shape[1] == 1:
                labels = labels.squeeze(1)
            graph.nodes[ntype].data["labels"] = labels

        # Process head_node_type for classification
        if self.head_node_type is None:
            if self.y_dict is not None:
                self.head_node_type = list(self.y_dict.keys())[0]
            else:
                self.head_node_type = self.node_types[0]

        # Process node data
        if "year" in graph.ndata:
            for ntype in graph.ntypes:
                if "year" in graph.nodes[ntype].data:
                    graph.nodes[ntype].data["feat"] = torch.cat([graph.nodes[ntype].data["feat"],
                                                                 graph.nodes[ntype].data["year"]], dim=1)

        self.init_node_embeddings(graph)

        self.metapaths = graph.canonical_etypes

        split_idx = dataset.get_idx_split()
        self.training_idx, self.validation_idx, self.testing_idx = split_idx["train"][self.head_node_type], \
                                                                   split_idx["valid"][self.head_node_type], \
                                                                   split_idx["test"][self.head_node_type]

    def init_node_embeddings(self, graph):
        for ntype in graph.ntypes:
            if "feat" not in graph.nodes[ntype].data:
                if self.node_attr_size:
                    embedding_dim = self.node_attr_size
                else:
                    embedding_dim = self.embedding_dim

                embed = torch.nn.Embedding(graph.num_nodes(ntype), embedding_dim)
                graph.nodes[ntype].data["feat"] = embed.weight

    def process_DglNodeDataset_homo(self, dataset: DglNodePropPredDataset):
        graph, labels = dataset[0]
        self._name = dataset.name
        self.G = graph

        if self.node_types is None:
            self.node_types = graph.ntypes

        self.num_nodes_dict = {ntype: graph.num_nodes(ntype) for ntype in self.node_types}

        if self.head_node_type is None:
            self.head_node_type = self.node_types[0]

        if labels.dim() == 2 and labels.size(1) == 1:
            labels = labels.squeeze(1)

        self.x_dict = {self.head_node_type: graph.ndata["feat"]} if "feat" in graph.ndata else {}

        graph.nodes[self.head_node_type].data["labels"] = labels
        self.y_dict = {self.head_node_type: labels}

        self.init_node_embeddings(graph)

        self.metapaths = graph.canonical_etypes

        split_idx = dataset.get_idx_split()
        self.training_idx, self.validation_idx, self.testing_idx = split_idx["train"], split_idx["valid"], split_idx[
            "test"]


    @property
    def node_attr_shape(self):
        if "feat" not in self.G.ndata:
            node_attr_shape = {}
        else:
            node_attr_shape = {ntype: self.G.nodes[ntype].data["feat"].size(1) \
                               for ntype in self.G.ntypes if "feat" in self.G.nodes[ntype].data}
        return node_attr_shape

    @property
    def edge_attr_shape(self):
        if "feat" not in self.G.edata:
            edge_attr_shape = {}
        else:
            edge_attr_shape = {etype: self.G.edata["feat"].size(1) for etype in self.G.etypes}
        return edge_attr_shape

    def get_metapaths(self):
        return self.G.canonical_etypes

    def get_collate_fn(self, collate_fn: str, mode=None):
        raise NotImplementedError()

    def collate_pyg(self, outputs):
        print(outputs)
        return outputs

    def sample(self, iloc, mode):
        raise NotImplementedError()

    def train_dataloader(self, collate_fn=None, batch_size=128, num_workers=12, **kwargs):
        if self.inductive:
            nodes = {ntype: self.G.nodes(ntype) for ntype in self.node_types if ntype != self.head_node_type}
            nodes[self.head_node_type] = self.training_idx

            graph = dgl.node_subgraph(self.G, nodes)
        else:
            graph = self.G

        collator = dgl.dataloading.NodeCollator(graph, nids={self.head_node_type: self.training_idx},
                                                block_sampler=self.neighbor_sampler)
        dataloader = DataLoader(collator.dataset, collate_fn=collator.collate,
                                batch_size=batch_size, shuffle=True, drop_last=False, num_workers=num_workers)

        # dataloader = dgl.dataloading.NodeDataLoader(
        #     graph, nids={self.head_node_type: self.training_idx},
        #     block_sampler=self.neighbor_sampler,
        #     batch_size=batch_size, shuffle=True, num_workers=num_workers)

        return dataloader

    def valid_dataloader(self, collate_fn=None, batch_size=128, num_workers=4, **kwargs):
        if self.inductive:
            nodes = {ntype: self.G.nodes(ntype) for ntype in self.node_types if ntype != self.head_node_type}
            nodes[self.head_node_type] = torch.tensor(np.union1d(self.training_idx, self.validation_idx))
            graph = dgl.node_subgraph(self.G, nodes)
        else:
            graph = self.G

        collator = dgl.dataloading.NodeCollator(graph, nids={self.head_node_type: self.validation_idx},
                                                block_sampler=self.neighbor_sampler)
        dataloader = DataLoader(collator.dataset, collate_fn=collator.collate,
                                batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)

        # dataloader = dgl.dataloading.NodeDataLoader(
        #     graph, nids={self.head_node_type: self.validation_idx},
        #     block_sampler=self.neighbor_sampler,
        #     batch_size=batch_size, shuffle=True, num_workers=num_workers)
        return dataloader

    def test_dataloader(self, collate_fn=None, batch_size=128, num_workers=4, **kwargs):
        graph = self.G

        collator = dgl.dataloading.NodeCollator(graph, nids={self.head_node_type: self.testing_idx},
                                                block_sampler=self.neighbor_sampler)
        dataloader = DataLoader(collator.dataset, collate_fn=collator.collate,
                                batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)

        # dataloader = dgl.dataloading.NodeDataLoader(
        #     graph, nids={self.head_node_type: self.testing_idx},
        #     block_sampler=self.neighbor_sampler,
        #     batch_size=batch_size, shuffle=True, num_workers=num_workers)
        return dataloader


class LATTEPyGCollator(dgl.dataloading.NodeCollator):
    def collate(self, items):
        if isinstance(items[0], tuple):
            # returns a list of pairs: group them by node types into a dict
            items = dglutils.group_as_dict(items)
            items = _prepare_tensor_dict(self.g, items, 'items', self._is_distributed)
        else:
            items = _prepare_tensor(self.g, items, 'items', self._is_distributed)

        blocks = self.block_sampler.sample_blocks(self.g, items)
        # output_nodes = blocks[-1].dstdata[dgl.NID]
        # input_nodes = blocks[0].srcdata[dgl.NID]

        layer_dicts = []
        for b in blocks:
            X = {}
            X["edge_index_dict"] = {}
            for metapath in b.canonical_etypes:
                if b.num_edges(etype=metapath) == 0:
                    continue
                X["edge_index_dict"][metapath] = torch.stack(b.all_edges(etype=b.canonical_etypes[1]), dim=0)

            X["x_dict"] = {k: v for k, v in b.ndata["feat"].items() if v.size(0) != 0}
            X["global_node_index"] = {ntype: b.nodes(ntype) for ntype in b.ntypes}

            layer_dicts.append(X)

        return layer_dicts
