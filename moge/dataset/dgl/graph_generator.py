import dgl
import tqdm
import numpy as np
import torch
from ogb.graphproppred import DglGraphPropPredDataset
from torch.utils.data import DataLoader
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

from moge.data.network import HeteroNetDataset


class DGLGraphSampler(HeteroNetDataset):
    def __init__(self, dataset: DglGraphPropPredDataset, embedding_dim=None, add_self_loop=False, node_types=None,
                 metapaths=None, head_node_type=None, edge_dir=True, reshuffle_train: float = None,
                 add_reverse_metapaths=True, inductive=True):
        self.embedding_dim = embedding_dim
        self.add_self_loop = add_self_loop

        super().__init__(dataset, node_types=node_types, metapaths=metapaths, head_node_type=head_node_type,
                         edge_dir=edge_dir, reshuffle_train=reshuffle_train,
                         add_reverse_metapaths=add_reverse_metapaths, inductive=inductive)

    def process_DglGraphDataset_homo(self, dataset: DglGraphPropPredDataset):
        a_graph, _ = dataset[0]
        self._name = dataset.name

        if self.node_types is None:
            self.node_types = a_graph.ntypes
        self.metapaths = a_graph.canonical_etypes
        self.num_nodes_dict = {ntype: a_graph.num_nodes(ntype) for ntype in self.node_types}

        if self.head_node_type is None:
            self.head_node_type = self.node_types[0]

        if "feat" not in a_graph.ndata:
            for g in tqdm.tqdm(dataset.graphs, desc="Instantiating embeddings for graphs"):
                embed = torch.nn.Embedding(g.num_nodes(), self.embedding_dim)
                g.ndata["feat"] = embed.weight

        elif "ogbg-mol" in dataset.name:
            self.atom_encoder = AtomEncoder(emb_dim=self.embedding_dim)
            self.bond_encoder = BondEncoder(emb_dim=self.embedding_dim)

            for g in tqdm.tqdm(dataset.graphs, desc="Instantiating molecular embeddings for graphs"):
                atom_emb = self.atom_encoder(g.ndata["feat"])  # x is input atom feature
                edge_emb = self.bond_encoder(g.edata["feat"])  # edge_attr is input edge feature

                g.ndata["feat"] = atom_emb
                g.edata["feat"] = edge_emb

        if self.add_self_loop:
            for g in tqdm.tqdm(dataset.graphs, desc="Adding self-loop for graphs"):
                g = dgl.transform.add_self_loop(g)

        self.dataset = dataset

        if dataset.labels.dim() == 2 and dataset.labels.size(1) == 1:
            dataset.labels = dataset.labels.squeeze(1)
        self.labels = dataset.labels

        split_idx = dataset.get_idx_split()
        self.training_idx, self.validation_idx, self.testing_idx = split_idx["train"], split_idx["valid"], split_idx[
            "test"]

    @property
    def node_attr_shape(self):
        if "feat" not in self.dataset.graphs[0].ndata:
            node_attr_shape = {}
        else:
            node_attr_shape = {ntype: self.dataset.graphs[0].nodes[ntype].data["feat"].size(1) \
                               for ntype in self.dataset.graphs[0].ntypes}
        return node_attr_shape

    @property
    def edge_attr_shape(self):
        if "feat" not in self.dataset.graphs[0].edata:
            edge_attr_shape = {}
        else:
            edge_attr_shape = {etype: self.dataset.graphs[0].edata["feat"].size(1) \
                               for etype in self.dataset.graphs[0].etypes}
        return edge_attr_shape

    def get_metapaths(self):
        return self.metapaths

    def get_collate_fn(self, collate_fn: str, mode=None):
        raise NotImplementedError()

    def sample(self, iloc, mode):
        raise NotImplementedError()

    def train_dataloader(self, collate_fn=None, batch_size=128, num_workers=12, **kwargs):
        collator = dgl.dataloading.GraphCollator()
        dataloader = DataLoader(self.dataset[self.training_idx], collate_fn=collator.collate,
                                batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
        return dataloader

    def valid_dataloader(self, collate_fn=None, batch_size=128, num_workers=4, **kwargs):
        collator = dgl.dataloading.GraphCollator()
        dataloader = DataLoader(self.dataset[self.validation_idx], collate_fn=collator.collate,
                                batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)
        return dataloader

    def test_dataloader(self, collate_fn=None, batch_size=128, num_workers=4, **kwargs):
        collator = dgl.dataloading.GraphCollator()
        dataloader = DataLoader(self.dataset[self.testing_idx], collate_fn=collator.collate,
                                batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)
        return dataloader
