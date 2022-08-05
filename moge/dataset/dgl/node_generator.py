from collections import defaultdict
from typing import List, Dict, Union, Iterable, Tuple, Optional

import dgl
import numpy as np
import pandas as pd
import torch
from dgl import AddReverse
from dgl import utils as dglutils
from dgl.init import zero_initializer
from dgl.sampling import RandomWalkNeighborSampler
from dgl.utils import prepare_tensor_dict, prepare_tensor
from ogb.nodeproppred import DglNodePropPredDataset
from sklearn.preprocessing import LabelBinarizer
from torch import Tensor
from torch.utils.data import DataLoader

from moge.dataset.graph import HeteroGraphDataset
from moge.model.utils import tensor_sizes
from moge.network.hetero import HeteroNetwork
from moge.network.sequence import BertSequenceTokenizer
from .samplers import ImportanceSampler
from .. import HeteroNeighborGenerator
from ..utils import one_hot_encoder, reverse_metapath
from ...network.base import SEQUENCE_COL


class DGLNodeSampler(HeteroGraphDataset):
    def __init__(self, dataset: DglNodePropPredDataset,
                 sampler: str = "MultiLayerNeighborSampler",
                 embedding_dim=None,
                 neighbor_sizes=None,
                 node_types=None,
                 metapaths=None,
                 head_node_type=None,
                 edge_dir=True,
                 reshuffle_train: float = None,
                 add_reverse_metapaths=True,
                 init_embeddings=False,
                 inductive=False, **kwargs):
        self.neighbor_sizes = neighbor_sizes
        self.embedding_dim = embedding_dim
        super().__init__(dataset, node_types=node_types, metapaths=metapaths, head_node_type=head_node_type,
                         edge_dir=edge_dir, reshuffle_train=reshuffle_train,
                         add_reverse_metapaths=add_reverse_metapaths, inductive=inductive, **kwargs)
        assert isinstance(self.G, dgl.DGLHeteroGraph)

        if add_reverse_metapaths:
            self.G = self.transform_heterograph(self.G, add_reverse=True)
        elif "feat" in self.G.edata and self.G.edata["feat"]:
            self.G = self.transform_heterograph(self.G, decompose_etypes=True, add_reverse=add_reverse_metapaths)

        if init_embeddings:
            self.init_node_embeddings(self.G)

        self.degree_counts = self.compute_node_degrees(add_reverse_metapaths)

        fanouts = []
        for layer, fanout in enumerate(self.neighbor_sizes):
            fanouts.append({etype: fanout for etype in self.G.canonical_etypes})

        if sampler == "MultiLayerNeighborSampler":
            print("Using MultiLayerNeighborSampler")
            self.neighbor_sampler = dgl.dataloading.MultiLayerNeighborSampler(fanouts)

        elif sampler == "ImportanceSampler":
            print("Using ImportanceSampler", )
            self.neighbor_sampler = ImportanceSampler(fanouts=fanouts,
                                                      metapaths=self.get_metapaths(),  # Original metapaths only
                                                      degree_counts=self.degree_counts,
                                                      edge_dir=edge_dir)
        else:
            print("Using MultiLayerFullNeighborSampler")
            self.neighbor_sampler = dgl.dataloading.MultiLayerFullNeighborSampler(len(self.neighbor_sizes))

    @classmethod
    def from_heteronetwork(cls, g: dgl.DGLHeteroGraph, labels: Union[Tensor, Dict[str, Tensor]],
                           num_classes: int, train_idx: Dict[str, Tensor], val_idx, test_idx,
                           **kwargs):
        if "classes" in kwargs:
            classes = kwargs.pop("classes")
        else:
            classes = None

        self = cls(dataset=g, metapaths=g.canonical_etypes, **kwargs)

        self.node_types = g.ntypes

        self.y_dict = {}
        if not isinstance(labels, dict):
            self.y_dict[self.head_node_type] = labels
            self.G.nodes[self.head_node_type].data["label"] = labels  # [:self.G.num_nodes(self.head_node_type)]
        elif isinstance(labels, dict):
            if self.head_node_type is not None and isinstance(self.head_node_type, str):
                self.y_dict = labels
                self.G.nodes[self.head_node_type].data["label"] = labels[self.head_node_type]

        self.n_classes = num_classes
        if isinstance(labels, dict):
            label = list(labels.values()).pop()
            if not isinstance(label, Tensor):
                label = torch.tensor(label)
            self.multilabel = True if label.dim() > 1 and label.size(1) > 1 else False
        else:
            self.multilabel = True if labels.dim() > 1 and labels.size(1) > 1 else False

        self.classes = classes

        if isinstance(train_idx, dict) and isinstance(self.head_node_type, str):
            self.training_idx = torch.tensor(train_idx[self.head_node_type])
            self.validation_idx = torch.tensor(val_idx[self.head_node_type])
            self.testing_idx = torch.tensor(test_idx[self.head_node_type])

        elif isinstance(train_idx, dict) and isinstance(self.head_node_type, Iterable):
            self.training_idx = {ntype: train_idx[ntype] for ntype in self.head_node_type}
            self.validation_idx = {ntype: val_idx[ntype] for ntype in self.head_node_type}
            self.testing_idx = {ntype: test_idx[ntype] for ntype in self.head_node_type}
        else:
            self.training_idx = torch.tensor(train_idx)
            self.validation_idx = torch.tensor(val_idx)
            self.testing_idx = torch.tensor(test_idx)

        return self

    @classmethod
    def from_cogdl_graph(cls, gtn_dataset, **kwargs):
        dataset = HeteroNeighborGenerator(gtn_dataset, neighbor_sizes=kwargs["neighbor_sizes"],
                                          node_types=kwargs["node_types"],
                                          head_node_type=kwargs["head_node_type"],
                                          metapaths=kwargs["metapaths"],
                                          add_reverse_metapaths=False)
        for ntype in kwargs["node_types"]:
            if ntype != dataset.head_node_type:
                dataset.x_dict[ntype] = dataset.x_dict[dataset.head_node_type]

        # # Relabel node IDS based on node type
        node_idx = {}
        for ntype in dataset.node_types:
            for m, eid in dataset.edge_index_dict.items():
                if m[0] == ntype:
                    node_idx.setdefault(ntype, []).append(eid[0].unique())
                elif m[-1] == ntype:
                    node_idx.setdefault(ntype, []).append(eid[1].unique())
            node_idx[ntype] = torch.cat(node_idx[ntype]).unique().sort().values

        print(tensor_sizes(node_idx))

        relabel_nodes = {node_type: defaultdict(lambda: -1,
                                                dict(zip(node_idx[node_type].numpy(),
                                                         range(node_idx[node_type].size(0))))) \
                         for node_type in node_idx}

        # Create heterograph
        relations = {m: (eid[0].apply_(relabel_nodes[m[0]].get).numpy(),
                         eid[1].apply_(relabel_nodes[m[-1]].get).numpy())
                     for m, eid in dataset.edge_index_dict.items()}
        print({m: (np.unique(eid[0]).shape, np.unique(eid[1]).shape) for m, eid in relations.items()})

        g: dgl.DGLHeteroGraph = dgl.heterograph(relations)

        for ntype, ndata in dataset.x_dict.items():
            if ntype in g.ntypes:
                print(ntype, g.num_nodes(ntype), node_idx[ntype].shape)
                g.nodes[ntype].data["feat"] = ndata[node_idx[ntype]]

        # Labels
        labels = dataset.y_dict[dataset.head_node_type][g.nodes(dataset.head_node_type)]

        self = cls.from_heteronetwork(g=g, labels=labels,
                                      num_classes=dataset.n_classes,
                                      train_idx=dataset.training_idx,
                                      val_idx=dataset.validation_idx,
                                      test_idx=dataset.testing_idx,
                                      sampler=kwargs["sampler"],
                                      neighbor_sizes=kwargs["neighbor_sizes"],
                                      head_node_type=dataset.head_node_type,
                                      add_reverse_metapaths=kwargs["add_reverse_metapaths"],
                                      inductive=kwargs["inductive"])
        return self

    def transform_heterograph(self, g: dgl.DGLHeteroGraph, add_reverse=False,
                              decompose_etypes=False, nodes_subset: Dict[str, Tensor] = None) -> dgl.DGLHeteroGraph:
        if decompose_etypes:
            relations = {}
            for metapath in g.canonical_etypes:
                # Original edges
                src, dst, eid = g.all_edges(etype=metapath[1], form="all")
                relations[metapath] = (src, dst)

                # Separate edge types by each non-zero entry in the `g.edata["feat"]` vector, with length = number of etypes
                if decompose_etypes:
                    relations = {}
                    edge_reltype = g.edata["feat"].argmax(1)
                    assert src.size(0) == edge_reltype.size(0)

                    for edge_type in range(g.edata["feat"].size(1)):
                        mask = edge_reltype == edge_type
                        metapath = (self.head_node_type, str(edge_type), self.head_node_type)
                        relations[metapath] = (src[mask], dst[mask])

            g = dgl.heterograph(relations, num_nodes_dict=self.num_nodes_dict)

        if nodes_subset:
            relations = {}
            for metapath in g.canonical_etypes:
                src, dst, eid = g.all_edges(etype=metapath[1], form="all")
                src, dst, _ = self.filter_edges(src, dst, nodes_subset=nodes_subset, metapath=metapath)
                relations[metapath] = (src, dst)

            num_nodes_old = {ntype: g.num_nodes(ntype) for ntype in g.ntypes}
            g = dgl.heterograph(relations, num_nodes_dict=self.num_nodes_dict)
            num_nodes_removed = dict(num_nodes - g.num_nodes(ntype) \
                                         if ntype in g.ntypes else 0 \
                                     for ntype, num_nodes in num_nodes_old.items())
            print(f"Removed {num_nodes_removed} nodes")

        if add_reverse:
            self.reverse_etypes, self.reverse_eids = {}, {}
            transform = AddReverse(copy_edata=True, sym_new_etype=True)
            new_g: dgl.DGLHeteroGraph = transform(g)

            # Get mapping between orig eid to reversed eid
            for metapath in g.canonical_etypes:
                rev_metapath = reverse_metapath(metapath)
                src_rev, dst_rev, eid_rev = new_g.all_edges(etype=rev_metapath[1], form="all")
                # print(metapath, eid[:10], (src[0], dst[0]))
                # print(rev_metapath, eid_rev[:10], (src_rev[0], dst_rev[0]))
                self.reverse_eids[metapath] = eid_rev
                self.reverse_etypes[metapath] = rev_metapath

            assert new_g.num_nodes() == g.num_nodes() and len(new_g.canonical_etypes) > len(g.canonical_etypes)
            print(f"Added reverse edges with {len(new_g.canonical_etypes) - len(g.canonical_etypes)} new etypes")

        return new_g

    def filter_edges(self, src: Tensor, dst: Tensor,
                     nodes_subset: Dict[str, Tensor], metapath: Tuple[str, str, str],
                     eid: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        for i, ntype in enumerate([metapath[0], metapath[-1]]):
            if ntype in nodes_subset and \
                    nodes_subset[ntype].size(0) < (src if i == 0 else dst).unique().size(0):
                if i == 0:
                    mask = np.isin(src, nodes_subset[ntype]) & np.isin(src, [-1], invert=True)
                elif i == 1:
                    mask = np.isin(dst, nodes_subset[ntype]) & np.isin(dst, [-1], invert=True)

                src = src[mask]
                dst = dst[mask]
                if eid is not None:
                    eid = eid[mask]

        return src, dst, eid

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
            graph.nodes[ntype].data["label"] = labels

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
                    lb = LabelBinarizer()
                    year_onehot = lb.fit_transform(graph.nodes[ntype].data["year"].numpy())
                    year_onehot = torch.from_numpy(year_onehot)

                    graph.nodes[ntype].data["feat"] = torch.cat([graph.nodes[ntype].data["feat"],
                                                                 year_onehot], dim=1)

        self.metapaths = graph.canonical_etypes

        split_idx = dataset.get_idx_split()
        self.training_idx, self.validation_idx, self.testing_idx = split_idx["train"][self.head_node_type], \
                                                                   split_idx["valid"][self.head_node_type], \
                                                                   split_idx["test"][self.head_node_type]

    def process_DglNodeDataset_homo(self, dataset: DglNodePropPredDataset):
        graph, labels = dataset[0]
        self.G = graph
        self._name = dataset.name

        if self.node_types is None:
            self.node_types = graph.ntypes

        self.num_nodes_dict = {ntype: graph.num_nodes(ntype) for ntype in self.node_types}

        if self.head_node_type is None:
            self.head_node_type = self.node_types[0]

        if labels.dim() == 2 and labels.size(1) == 1:
            labels = labels.squeeze(1)

        self.x_dict = {self.head_node_type: graph.ndata["feat"]} if "feat" in graph.ndata else {}

        graph.nodes[self.head_node_type].data["label"] = labels
        self.y_dict = {self.head_node_type: labels}

        self.metapaths = graph.canonical_etypes

        split_idx = dataset.get_idx_split()
        self.training_idx, self.validation_idx, self.testing_idx = split_idx["train"], split_idx["valid"], split_idx[
            "test"]

    def init_node_embeddings(self, graph: dgl.DGLHeteroGraph, ntype_key="species"):
        if self.node_attr_size:
            embedding_dim = self.node_attr_size
        else:
            embedding_dim = self.embedding_dim

        for ntype in graph.ntypes:
            graph.set_n_initializer(zero_initializer, field="feat", ntype=ntype)

            if ntype_key in graph.nodes[ntype].data and "feat" not in graph.nodes[ntype].data:
                onehot = one_hot_encoder(graph.nodes[ntype].data[ntype_key])
                print("onehot", onehot.shape, onehot)
                graph.nodes[ntype].data["feat"] = onehot.requires_grad_(True)

            # elif "feat" not in graph.nodes[ntype].data:
            #     self.node_embedding = torch.nn.Embedding(graph.num_nodes(ntype), embedding_dim)
            #     print(f"Initialized Embedding({graph.num_nodes(ntype)}, {embedding_dim}) for ntype: {ntype}")
            #     graph.nodes[ntype].data["feat"] = self.node_embedding.weight
            #
            #     assert graph.nodes[ntype].data["feat"].requires_grad

            # if ntype_key in graph.nodes[ntype].data:
            #     species = graph.nodes[ntype].data[ntype_key].unique()
            #     self.ntype_embedding = torch.nn.Embedding(species.size(0), embedding_dim)
            #     print(f"Initialized ntype_embedding({species.size(0)}, {embedding_dim}) for ntype: {ntype}")
            #     for species_id in species:
            #         nmask = (graph.nodes[ntype].data[ntype_key] == species_id).squeeze(-1)
            #         graph.nodes[ntype].data["feat"][nmask, :] = \
            #             graph.nodes[ntype].data["feat"][nmask, :] + self.ntype_embedding.weight[(species == species_id).nonzero().item()].unsqueeze(0)

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

    def get_metapaths(self, **kwargs):
        return self.G.canonical_etypes

    def sample(self, iloc, **kwargs):
        loader = self.train_dataloader(**kwargs)
        # return loader.collate_fn(iloc)
        return next(iter(loader))

    def get_training_subgraph(self):
        nodes = {ntype: self.G.nodes(ntype) for ntype in self.node_types if ntype != self.head_node_type}
        nodes[self.head_node_type] = torch.tensor(self.training_idx, dtype=torch.long)
        graph = self.transform_heterograph(self.G, nodes_subset=nodes)

        print("Removed edges incident to test nodes from the training subgraph for inductive node classification: \n",
              graph)
        return graph

    def train_dataloader(self, collate_fn=None, batch_size=128, num_workers=0, **kwargs):
        if self.inductive:
            graph = self.get_training_subgraph()
        else:
            graph = self.G

        if isinstance(self.head_node_type, str) and isinstance(self.training_idx, (Tensor, np.ndarray)):
            seed_nodes = {self.head_node_type: self.training_idx}
        elif isinstance(self.head_node_type, str) and isinstance(self.training_idx, dict):
            seed_nodes = {self.head_node_type: self.training_idx[self.head_node_type]}
        else:
            assert isinstance(self.training_idx, dict) and len(self.head_node_type) == len(self.training_idx)
            seed_nodes = self.training_idx

        if collate_fn == "neighbor_sampler":
            collator = NodeClfPyGCollator(graph, nids=seed_nodes,
                                          graph_sampler=self.neighbor_sampler)
        else:
            collator = dgl.dataloading.NodeCollator(graph, nids=seed_nodes,
                                                    graph_sampler=self.neighbor_sampler)

        if hasattr(self, "network") and hasattr(self, "tokenizer"):
            self.network: HeteroNetwork
            self.tokenizer: BertSequenceTokenizer

            def collate_fn(idx):
                if collate_fn == "neighbor_sampler":
                    X, y_dict, weights = collator.collate(idx)
                    node_names = {ntype: self.network.nodes[ntype][nid.numpy()] \
                                  for ntype, nid in X["global_node_index"][0].items()}

                    for ntype, names in node_names.items():
                        sequences = self.network.multiomics[ntype].annotations.loc[names, "sequence"]

                        output = self.tokenizer.one_hot_encode(ntype, sequences.to_list())
                        X["x_dict"][ntype].ndata["input_ids"] = output["input_ids"]
                        X["x_dict"][ntype].ndata["attention_mask"] = output["attention_mask"]
                        X["x_dict"][ntype].ndata["token_type_ids"] = output["token_type_ids"]

                    return X, y_dict, weights

                else:
                    input_nodes, seeds, blocks = collator.collate(idx)

                    node_names = {ntype: self.network.nodes[ntype][nid.numpy()] \
                                  for ntype, nid in input_nodes.items()}

                    for ntype, names in node_names.items():
                        sequences = self.network.multiomics[ntype].annotations.loc[names, "sequence"]

                        output = self.tokenizer.one_hot_encode(ntype, sequences.to_list())
                        blocks[0].nodes[ntype].ndata["input_ids"] = output["input_ids"]
                        blocks[0].nodes[ntype].ndata["attention_mask"] = output["attention_mask"]
                        blocks[0].nodes[ntype].ndata["token_type_ids"] = output["token_type_ids"]

                    return input_nodes, seeds, blocks
        else:
            collate_fn = collator.collate

        dataloader = DataLoader(collator.dataset, collate_fn=collate_fn,
                                batch_size=batch_size, shuffle=True, drop_last=False, num_workers=num_workers)

        return dataloader

    def valid_dataloader(self, collate_fn=None, batch_size=128, num_workers=0, **kwargs):
        graph = self.G

        if isinstance(self.head_node_type, str) and isinstance(self.validation_idx, (Tensor, np.ndarray)):
            seed_nodes = {self.head_node_type: self.validation_idx}
        elif isinstance(self.head_node_type, str) and isinstance(self.validation_idx, dict):
            seed_nodes = {self.head_node_type: self.validation_idx[self.head_node_type]}
        else:
            assert isinstance(self.validation_idx, dict) and len(self.head_node_type) == len(self.validation_idx)
            seed_nodes = self.validation_idx

        if collate_fn == "neighbor_sampler":
            collator = NodeClfPyGCollator(graph, nids=seed_nodes,
                                          graph_sampler=self.neighbor_sampler)
        else:
            collator = dgl.dataloading.NodeCollator(graph, nids=seed_nodes,
                                                    graph_sampler=self.neighbor_sampler)

        dataloader = DataLoader(collator.dataset, collate_fn=collator.collate,
                                batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)

        return dataloader

    def test_dataloader(self, collate_fn=None, batch_size=128, num_workers=0, **kwargs):
        graph = self.G

        if isinstance(self.head_node_type, str) and isinstance(self.testing_idx, (Tensor, np.ndarray)):
            seed_nodes = {self.head_node_type: self.testing_idx}
        elif isinstance(self.head_node_type, str) and isinstance(self.testing_idx, dict):
            seed_nodes = {self.head_node_type: self.testing_idx[self.head_node_type]}
        else:
            assert isinstance(self.testing_idx, dict) and len(self.head_node_type) == len(self.testing_idx)
            seed_nodes = self.testing_idx

        if collate_fn == "neighbor_sampler":
            collator = NodeClfPyGCollator(graph, nids=seed_nodes,
                                          graph_sampler=self.neighbor_sampler)
        else:
            collator = dgl.dataloading.NodeCollator(graph, nids=seed_nodes,
                                                    graph_sampler=self.neighbor_sampler)

        dataloader = DataLoader(collator.dataset, collate_fn=collator.collate,
                                batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)

        return dataloader


class NodeClfPyGCollator(dgl.dataloading.NodeCollator):
    def collate(self, items):
        if isinstance(items[0], tuple):
            # returns a list of pairs: group them by node types into a dict
            items = dglutils.group_as_dict(items)
            items = prepare_tensor_dict(self.g, items, 'items')
        else:
            items = prepare_tensor(self.g, items, 'items')

        blocks: List[dgl.DGLHeteroGraph] = self.graph_sampler.sample_blocks(self.g, items)
        print(len(blocks))
        output_nodes = blocks[-1].dstdata[dgl.NID]
        input_nodes = blocks[0].srcdata[dgl.NID]

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
            X[SEQUENCE_COL] = {ntype: feat \
                               for ntype, feat in blocks[0].srcdata[SEQUENCE_COL].items() \
                               if feat.size(0) != 0}
            X["seq_len"] = {ntype: feat \
                            for ntype, feat in blocks[0].srcdata["seq_len"].items() \
                            if feat.size(0) != 0}

        y_dict = blocks[-1].dstdata["label"]
        weights = None

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

        return X, y_dict, weights


class HANSampler(object):
    def __init__(self, g, metapath_list, num_neighbors):
        self.sampler_list: List[RandomWalkNeighborSampler] = []
        for metapath in metapath_list:
            # note: random walk may get same route(same edge), which will be removed in the sampled graph.
            # So the sampled graph's edges may be less than num_random_walks(num_neighbors).
            self.sampler_list.append(RandomWalkNeighborSampler(G=g,
                                                               num_traversals=1,
                                                               termination_prob=0,
                                                               num_random_walks=num_neighbors,
                                                               num_neighbors=num_neighbors,
                                                               metapath=metapath))

    def sample_blocks(self, seeds):
        block_list = []
        for sampler in self.sampler_list:
            frontier = sampler(seeds)
            # add self loop
            frontier = dgl.remove_self_loop(frontier)
            frontier.add_edges(torch.tensor(seeds), torch.tensor(seeds))
            block = dgl.to_block(frontier, seeds)
            block_list.append(block)

        if isinstance(seeds, list):
            seeds = torch.stack(seeds, dim=0)

        return seeds, block_list
