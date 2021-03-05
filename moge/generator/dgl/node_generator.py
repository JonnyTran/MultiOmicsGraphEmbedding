import dgl
import numpy as np
import pandas as pd
import torch
from ogb.nodeproppred import DglNodePropPredDataset
from torch.utils.data import DataLoader

from dgl.dataloading import BlockSampler
from moge.generator.network import HeteroNetDataset
from .samplers import ImportanceSampler, MultiLayerNeighborSampler


class DGLNodeSampler(HeteroNetDataset):
    def __init__(self, dataset: DglNodePropPredDataset,
                 sampler: BlockSampler,
                 neighbor_sizes=None,
                 node_types=None,
                 metapaths=None,
                 head_node_type=None,
                 edge_dir=True,
                 reshuffle_train: float = None,
                 add_reverse_metapaths=True,
                 inductive=True):
        self.neighbor_sizes = neighbor_sizes
        super().__init__(dataset, node_types=node_types, metapaths=metapaths, head_node_type=head_node_type,
                         edge_dir=edge_dir, reshuffle_train=reshuffle_train,
                         add_reverse_metapaths=add_reverse_metapaths, inductive=inductive)
        assert isinstance(self.G, (dgl.DGLGraph, dgl.DGLHeteroGraph))

        if add_reverse_metapaths:
            relations = {}

            for etype in self.G.etypes:
                rel_g = self.G.edge_type_subgraph([etype, ])
                relations[self.G.to_canonical_etype(etype)] = rel_g.all_edges()

                rel_reverse_name = self.get_reverse_metapath_name(self.G.to_canonical_etype(etype), None)
                # rel_reverse = dgl.heterograph({rel_reverse_name: rel_g.reverse().all_edges()})
                relations[rel_reverse_name] = rel_g.reverse().all_edges()  # rel_reverse.all_edges()

            new_g = dgl.heterograph(relations)

            for ntype in self.G.ntypes:
                for k, v in self.G.nodes[ntype].data.items():
                    new_g.nodes[ntype].data[k] = v

            self.G = new_g
        #     self.G = dgl.to_bidirected(self.G, copy_ndata=True)

        self.degree_counts = self.compute_node_degrees(add_reverse_metapaths)

        if sampler is None:
            print("Using Full Multilayer sampler")
            self.neighbor_sampler = dgl.dataloading.MultiLayerFullNeighborSampler(n_layers=len(self.neighbor_sizes))
        elif sampler == "ImportanceSampler":
            self.neighbor_sampler = ImportanceSampler(fanouts=neighbor_sizes,
                                                      metapaths=self.get_metapaths(),
                                                      degree_counts=self.degree_counts,
                                                      edge_dir=edge_dir)
        else:
            raise Exception

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

        head_counts = df.groupby(["head", "relation", "head_type"])["tail"].count()

        if add_reverse_metapaths:
            head_counts.index = head_counts.index.set_names(["nid", "relation", "ntype"])
            return head_counts

        # For directed graphs, use both
        else:
            tail_counts = df.groupby(["tail", "relation", "tail_type"])["head"].count()
            tail_counts.index = tail_counts.index.set_levels(levels=-tail_counts.index.get_level_values(1) - 1,
                                                             level=1,
                                                             verify_integrity=False, )
            head_counts.index = head_counts.index.set_names(["nid", "relation", "ntype"])
            tail_counts.index = tail_counts.index.set_names(["nid", "relation", "ntype"])
            return head_counts.append(tail_counts)  # (node_id, relation, ntype): count

    def process_DglNodeDataset_hetero(self, dataset: DglNodePropPredDataset):
        graph, labels = dataset[0]
        self._name = dataset.name

        if self.node_types is None:
            self.node_types = graph.ntypes

        self.num_nodes_dict = {ntype: graph.num_nodes(ntype) for ntype in self.node_types}
        self.y_dict = labels

        self.x_dict = graph.ndata["feat"]

        for ntype, labels in self.y_dict.items():
            if labels.dim() == 2 and labels.shape[1] == 1:
                labels = labels.squeeze(1)
            graph.nodes[ntype].data["labels"] = labels

        if self.head_node_type is None:
            if self.y_dict is not None:
                self.head_node_type = list(self.y_dict.keys())[0]
            else:
                self.head_node_type = self.node_types[0]

        self.metapaths = graph.canonical_etypes

        split_idx = dataset.get_idx_split()
        self.training_idx, self.validation_idx, self.testing_idx = split_idx["train"][self.head_node_type], \
                                                                   split_idx["valid"][self.head_node_type], \
                                                                   split_idx["test"][self.head_node_type]

        self.G = graph

    def get_metapaths(self):
        return self.G.canonical_etypes

    def get_collate_fn(self, collate_fn: str, mode=None):
        raise NotImplementedError()

    def sample(self, iloc, mode):
        raise NotImplementedError()

    def train_dataloader(self, collate_fn=None, batch_size=128, num_workers=12, **kwargs):
        if self.inductive:
            nodes = {ntype: self.G.nodes(ntype) for ntype in self.node_types if ntype != self.head_node_type}
            nodes[self.head_node_type] = self.training_idx

            graph = dgl.node_subgraph(self.G, nodes)
        else:
            graph = self.G

        collator = dgl.dataloading.NodeCollator(graph, {self.head_node_type: self.training_idx}, self.neighbor_sampler)
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

        collator = dgl.dataloading.NodeCollator(graph, {self.head_node_type: self.validation_idx},
                                                self.neighbor_sampler)
        dataloader = DataLoader(collator.dataset, collate_fn=collator.collate,
                                batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
        #
        # dataloader = dgl.dataloading.NodeDataLoader(
        #     graph, nids={self.head_node_type: self.validation_idx},
        #     block_sampler=self.neighbor_sampler,
        #     batch_size=batch_size, shuffle=True, num_workers=num_workers)
        return dataloader

    def test_dataloader(self, collate_fn=None, batch_size=128, num_workers=4, **kwargs):
        graph = self.G

        collator = dgl.dataloading.NodeCollator(graph, {self.head_node_type: self.testing_idx},
                                                self.neighbor_sampler)
        dataloader = DataLoader(collator.dataset, collate_fn=collator.collate,
                                batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)

        # dataloader = dgl.dataloading.NodeDataLoader(
        #     graph, nids={self.head_node_type: self.testing_idx},
        #     block_sampler=self.neighbor_sampler,
        #     batch_size=batch_size, shuffle=True, num_workers=num_workers)
        return dataloader
