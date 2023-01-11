from collections import defaultdict
from typing import List, Dict, Union, Tuple, Optional, Callable

import dgl
import numpy as np
import pandas as pd
import torch
from dgl import AddReverse
from dgl import utils as dglutils
from dgl.dataloading import BlockSampler
from dgl.sampling import RandomWalkNeighborSampler
from dgl.utils import prepare_tensor_dict, prepare_tensor
from logzero import logger
from ogb.nodeproppred import DglNodePropPredDataset
from pandas import Index, DataFrame
from sklearn.preprocessing import LabelBinarizer
from torch import Tensor
from torch.utils.data import DataLoader

from moge.dataset.graph import HeteroGraphDataset
from moge.model.tensor import tensor_sizes
from moge.network.hetero import HeteroNetwork
from moge.preprocess.metapaths import reverse_metapath, unreverse_metapath, is_reversed, is_negative
from .samplers import ImportanceSampler
from .utils import copy_ndata
from ..PyG.node_generator import HeteroNeighborGenerator
from ...network.base import SEQUENCE_COL


class DGLNodeGenerator(HeteroGraphDataset):
    def __init__(self, dataset: Union[DglNodePropPredDataset, dgl.DGLHeteroGraph],
                 sampler: str = "MultiLayerNeighborSampler",
                 neighbor_sizes=None,
                 node_types=None,
                 metapaths=None,
                 head_node_type=None,
                 edge_dir=True,
                 reshuffle_train: float = None,
                 add_reverse_metapaths=True,
                 inductive=False, decompose_etypes=True, **kwargs):
        self.sampler = sampler
        self.edge_dir = edge_dir
        self.neighbor_sizes = neighbor_sizes
        super().__init__(dataset, node_types=node_types, metapaths=metapaths, head_node_type=head_node_type,
                         edge_dir=edge_dir, add_reverse_metapaths=add_reverse_metapaths, inductive=inductive, **kwargs)
        assert isinstance(self.G, dgl.DGLHeteroGraph)

        if add_reverse_metapaths:
            self.G = self.transform_heterograph(self.G, add_reverse=True)
        elif decompose_etypes and "feat" in self.G.edata and self.G.edata["feat"]:
            self.G = self.transform_heterograph(self.G, decompose_etypes=decompose_etypes,
                                                add_reverse=add_reverse_metapaths)

        self.neighbor_sampler = self.get_neighbor_sampler(neighbor_sizes=neighbor_sizes, etypes=self.G.canonical_etypes,
                                                          sampler=sampler, edge_dir=edge_dir)

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
            graph.nodes[ntype].data["label"] = labels

        self.metapaths = graph.canonical_etypes

    @classmethod
    def from_heteronetwork(cls, network: HeteroNetwork, head_node_type, sampler, neighbor_sizes,
                           node_attr_cols: List[str] = None,
                           target: str = None, min_count: int = None,
                           expression=False, sequence=False, add_reverse_metapaths=True,
                           labels_subset: Optional[Union[Index, np.ndarray]] = None,
                           ntype_subset: Optional[List[str]] = None,
                           exclude_etypes: Optional[List[Tuple[str, str, str]]] = None,
                           split_namespace=False, **kwargs):
        G, classes, nodes, training_idx, validation_idx, testing_idx = \
            network.to_dgl_heterograph(node_attr_cols=node_attr_cols, target=target, min_count=min_count,
                                       labels_subset=labels_subset, head_node_type=head_node_type,
                                       ntype_subset=ntype_subset, exclude_etypes=exclude_etypes, sequence=sequence,
                                       expression=expression, train_test_split="node_id")

        self = cls(dataset=G, metapaths=G.canonical_etypes, add_reverse_metapaths=add_reverse_metapaths,
                   head_node_type=head_node_type, sampler=sampler, neighbor_sizes=neighbor_sizes,
                   edge_dir="in", **kwargs)
        self.network = network
        self.classes = classes
        self.n_classes = len(classes)
        self.nodes = nodes
        self._name = network._name if hasattr(network, '_name') else ""
        self.y_dict = G.ndata["label"]
        if isinstance(self.y_dict, dict):
            self.multilabel = any((label.sum(1) > 1).any() if label.dim() == 2 else False \
                                  for label in self.y_dict.values())
        else:
            self.multilabel = (self.y_dict.sum(1) > 1).any() if self.y_dict.dim() == 2 else False
        self.process_classes()

            # Whether to use split namespace
        self.split_namespace = split_namespace
        if split_namespace:
            self.nodes_namespace = {}
            self.ntype_mapping = {}
            for ntype, df in network.annotations.items():
                if "namespace" in df.columns:
                    self.nodes_namespace[ntype] = network.annotations[ntype]["namespace"].loc[self.nodes[ntype]]
                    self.ntype_mapping.update(
                        {namespace: ntype for namespace in np.unique(self.nodes_namespace[ntype])})

        self.training_idx, self.validation_idx, self.testing_idx = training_idx, validation_idx, testing_idx

        # Remove metapaths that have information in the predictions
        self.pred_metapaths = network.pred_metapaths if hasattr(network, "pred_metapaths") else []
        self.neg_pred_metapaths = network.neg_pred_metapaths if hasattr(network, "neg_pred_metapaths") else []
        if hasattr(network, 'pred_metapaths') and not set(network.pred_metapaths).issubset(self.pred_metapaths):
            self.pred_metapaths.extend(network.pred_metapaths)
        if hasattr(network, 'neg_pred_metapaths') and \
                not set(network.neg_pred_metapaths).issubset(self.neg_pred_metapaths):
            self.neg_pred_metapaths.extend(network.neg_pred_metapaths)

        if self.use_reverse:
            for metapath in self.G.canonical_etypes:
                if is_negative(metapath) and is_reversed(metapath):
                    logger.info(f"Removed {metapath}")
                    self.G.remove_edges(eids=self.G.edges(etype=metapath, form='eid'), etype=metapath)

                elif self.pred_metapaths and (metapath in self.pred_metapaths or
                                              unreverse_metapath(metapath) in self.pred_metapaths):
                    logger.info(f"Removed {metapath}")
                    self.G.remove_edges(eids=self.G.edges(etype=metapath, form='eid'), etype=metapath)

                elif self.neg_pred_metapaths and (metapath in self.neg_pred_metapaths or
                                                  unreverse_metapath(metapath) in self.neg_pred_metapaths):
                    logger.info(f"Removed {metapath}")
                    self.G.remove_edges(eids=self.G.edges(etype=metapath, form='eid'), etype=metapath)

            self.metapaths = [metapath for metapath in self.G.canonical_etypes if self.G.num_edges(etype=metapath)]

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

        self = cls.from_heteronetwork(g, labels=labels,
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

    def process_ogb_DglNodeDataset_hetero(self, dataset: DglNodePropPredDataset):
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

    def process_ogb_DglNodeDataset_homo(self, dataset: DglNodePropPredDataset):
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

    def transform_heterograph(self, G: dgl.DGLHeteroGraph,
                              add_reverse=False,
                              decompose_etypes=False,
                              nodes_subset: Dict[str, Tensor] = None,
                              edge_mask: Dict[str, Tensor] = None,
                              drop_empty_etypes=True,
                              verbose=True) -> dgl.DGLHeteroGraph:
        if decompose_etypes:
            edge_index_dict = {}
            for metapath in G.canonical_etypes:
                # Original edges
                src, dst, eid = G.all_edges(etype=metapath, form="all")
                edge_index_dict[metapath] = (src, dst)

                # Separate edge types by each non-zero entry in the `g.edata["feat"]` vector, with length = number of etypes
                if decompose_etypes:
                    edge_index_dict = {}
                    edge_reltype = G.edata["feat"].argmax(1)
                    assert src.size(0) == edge_reltype.size(0)

                    for edge_type in range(G.edata["feat"].size(1)):
                        mask = edge_reltype == edge_type
                        metapath = (self.head_node_type, str(edge_type), self.head_node_type)
                        edge_index_dict[metapath] = (src[mask], dst[mask])

            G = dgl.heterograph(edge_index_dict, num_nodes_dict=self.num_nodes_dict)

        if edge_mask:
            edge_index_dict = {}
            for metapath in G.canonical_etypes:
                src, dst, eid = self.filter_edges(G.all_edges(etype=metapath, form="all"), metapath=metapath,
                                                  edge_mask=edge_mask)
                if drop_empty_etypes and len(src) == 0: continue
                edge_index_dict[metapath] = (src, dst)

            new_g = dgl.heterograph(edge_index_dict, num_nodes_dict=self.num_nodes_dict)
            num_edges_removed = {etype: G.num_edges(etype=etype) - new_g.num_edges(etype=etype) \
                                 for etype in new_g.canonical_etypes \
                                 if (G.num_edges(etype=etype) - new_g.num_edges(etype=etype))}

            new_g = copy_ndata(old_g=G, new_g=new_g)
            logger.info(f"Removed edges: {num_edges_removed}") if verbose else None

        if nodes_subset:
            edge_index_dict = {}
            for metapath in G.canonical_etypes:
                src, dst, eid = self.filter_edges(G.all_edges(etype=metapath, form="all"), metapath=metapath,
                                                  nids_subset=nodes_subset)
                if drop_empty_etypes and len(src) == 0: continue

                edge_index_dict[metapath] = (src, dst)

            num_nodes_old = {ntype: G.num_nodes(ntype) for ntype in G.ntypes}
            new_g = dgl.heterograph(edge_index_dict, num_nodes_dict=self.num_nodes_dict)
            num_nodes_removed = dict(num_nodes - G.num_nodes(ntype) \
                                         if ntype in G.ntypes else 0 \
                                     for ntype, num_nodes in num_nodes_old.items())

            new_g = copy_ndata(old_g=G, new_g=new_g)
            logger.info(f"Removed nodes: {num_nodes_removed}") if verbose else None

        if add_reverse:
            self.reverse_etypes, self.reverse_eids = {}, {}
            transform = AddReverse(copy_edata=True, sym_new_etype=False)
            new_g: dgl.DGLHeteroGraph = transform(G)

            # Get mapping between orig eid to reversed eid
            for metapath in G.canonical_etypes:
                rev_metapath = reverse_metapath(metapath)
                if rev_metapath not in new_g.canonical_etypes: continue
                src_rev, dst_rev, eid_rev = new_g.all_edges(etype=rev_metapath, form="all")
                # print(metapath, eid[:10], (src[0], dst[0]))
                # print(rev_metapath, eid_rev[:10], (src_rev[0], dst_rev[0]))
                self.reverse_eids[metapath] = eid_rev
                self.reverse_etypes[metapath] = rev_metapath

            self.metapaths = new_g.canonical_etypes
            assert new_g.num_nodes() == G.num_nodes() and len(new_g.canonical_etypes) >= len(G.canonical_etypes)

            logger.info(
                f"Added reverse edges with {len(new_g.canonical_etypes) - len(G.canonical_etypes)} new etypes") if verbose else None

        return new_g

    def filter_edges(self, edges: Tuple[Tensor, Tensor, Optional[Tensor]], metapath: Tuple[str, str, str],
                     nids_subset: Dict[str, Tensor] = None, edge_mask: Dict[str, Tensor] = None) \
            -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        src, dst, eid = edges

        if edge_mask is not None:
            mask = edge_mask[metapath] if isinstance(edge_mask, dict) else edge_mask
            assert len(mask) == len(src), f"{metapath}, edges {len(src)}, mask {len(mask)}"

            src = src[mask]
            dst = dst[mask]
            if eid is not None:
                eid = eid[mask]

        for i, ntype in enumerate([metapath[0], metapath[-1]]):
            if nids_subset and ntype in nids_subset and \
                    nids_subset[ntype].size(0) < (src if i == 0 else dst).unique().size(0):
                if i == 0:
                    mask = np.isin(src, nids_subset[ntype]) & np.isin(src, [-1], invert=True)
                elif i == 1:
                    mask = np.isin(dst, nids_subset[ntype]) & np.isin(dst, [-1], invert=True)

                src = src[mask]
                dst = dst[mask]
                if eid is not None:
                    eid = eid[mask]

        return src, dst, eid

    def compute_node_degrees(self, undirected):
        dfs = []
        metapaths = self.G.canonical_etypes

        for metapath in self.G.canonical_etypes:
            head_type, tail_type = metapath[0], metapath[-1]
            relation = metapaths.index(metapath)

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

        if undirected:
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

    def split_array_by_namespace(self, labels: Union[Tensor, Dict[str, Tensor], np.ndarray], ntype=None):
        assert hasattr(self, "nodes_namespace")
        if ntype is None:
            ntype = self.go_ntype if hasattr(self, "go_ntype") else "GO_term"
        nodes_namespaces = self.nodes_namespace[ntype].loc[self.classes]

        y_dict = {}
        for namespace in np.unique(nodes_namespaces):
            mask = nodes_namespaces == namespace

            if isinstance(labels, (Tensor, np.ndarray, DataFrame)):
                y_dict[namespace] = labels[:, mask]
            elif isinstance(labels, dict):
                for ntype, labels in labels.items():
                    y_dict.setdefault(ntype, {})[namespace] = labels[:, mask]

        return y_dict

    @property
    def edge_index_dict(self):
        return {etype: torch.stack(self.G.edges(etype=etype, form="uv"), dim=0) \
                for etype in self.G.canonical_etypes}

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

    def get_neighbor_sampler(self, neighbor_sizes, etypes, sampler: str = "MultiLayerNeighborSampler", edge_dir="in",
                             set_fanouts: Dict[Tuple, int] = None):
        if neighbor_sizes is None:
            neighbor_sizes = self.neighbor_sizes
        if edge_dir is None:
            edge_dir = self.edge_dir

        fanouts = []
        for layer, fanout in enumerate(neighbor_sizes):
            etypes_fanout = {etype: fanout for etype in etypes}
            if set_fanouts:
                etypes_fanout.update(set_fanouts)
            fanouts.append(etypes_fanout)

        if sampler == "MultiLayerNeighborSampler":
            neighbor_sampler = dgl.dataloading.MultiLayerNeighborSampler(fanouts)

        elif sampler == "ImportanceSampler":
            neighbor_sampler = ImportanceSampler(fanouts=fanouts,
                                                 metapaths=self.get_metapaths(),  # Original metapaths only
                                                 degree_counts=self.degree_counts,
                                                 edge_dir=edge_dir)
        else:
            neighbor_sampler = dgl.dataloading.MultiLayerFullNeighborSampler(len(self.neighbor_sizes))
        return neighbor_sampler

    def get_node_collator(self, graph: dgl.DGLHeteroGraph, seed_nodes: Dict[str, Tensor],
                          collate_fn: Union[str, Callable], graph_sampler: BlockSampler,
                          add_idx: List[Tuple[str, int]] = None) \
            -> Tuple[dgl.dataloading.NodeCollator, Callable]:
        """

        Args:
            graph ():
            seed_nodes (Dict[str,Tensor]):
            collate_fn (Union[str, Callable]): Either a string `"NodeClfPyGCollator"` or a callable function which
                transforms each batch. If passing a function, it takes in `input_nodes, seeds, blocks` arguments and
                returns `input_nodes, seeds, blocks`.
            graph_sampler ():
            add_idx ():

        Returns:

        """
        if collate_fn == NodeClfPyGCollator.__name__:
            collator = NodeClfPyGCollator(graph, nids=seed_nodes, graph_sampler=graph_sampler)
        else:
            collator = dgl.dataloading.NodeCollator(graph, nids=seed_nodes, graph_sampler=graph_sampler)

        if collate_fn is not None:
            def _collate_fn(idx: List[Tuple[str, int]]):
                if add_idx:
                    idx.extend(add_idx)

                if collate_fn == NodeClfPyGCollator.__name__:
                    X, y_dict, weights = collator.collate(idx)
                    if hasattr(self, "network") and hasattr(self, "tokenizer"):
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

                    if hasattr(self, "network") and hasattr(self, "tokenizer"):
                        node_names = {ntype: self.network.nodes[ntype][nid.numpy()] \
                                      for ntype, nid in input_nodes.items()}

                        for ntype, names in node_names.items():
                            sequences = self.network.multiomics[ntype].annotations.loc[names, "sequence"]

                            output = self.tokenizer.one_hot_encode(ntype, sequences.to_list())
                            blocks[0].nodes[ntype].ndata["input_ids"] = output["input_ids"]
                            blocks[0].nodes[ntype].ndata["attention_mask"] = output["attention_mask"]
                            blocks[0].nodes[ntype].ndata["token_type_ids"] = output["token_type_ids"]

                    if callable(collate_fn):
                        return collate_fn(input_nodes, seeds, blocks)

                    else:
                        return input_nodes, seeds, blocks
        else:
            _collate_fn = collator.collate

        return collator, _collate_fn

    def train_dataloader(self, collate_fn=None, batch_size=128, num_workers=0, indices=None, **kwargs):
        if self.inductive:
            graph = self.transform_heterograph(self.G, nodes_subset=self.training_idx)
        else:
            graph = self.G

        if indices is not None:
            seed_nodes = indices
        elif isinstance(self.head_node_type, str) and isinstance(self.training_idx, (Tensor, np.ndarray)):
            seed_nodes = {self.head_node_type: self.training_idx}
        elif isinstance(self.head_node_type, str) and isinstance(self.training_idx, dict):
            seed_nodes = {self.head_node_type: self.training_idx[self.head_node_type]}
        else:
            assert isinstance(self.training_idx, dict) and len(self.head_node_type) == len(self.training_idx)
            seed_nodes = self.training_idx

        collator, collate_fn = self.get_node_collator(graph, seed_nodes, collate_fn, self.neighbor_sampler)

        dataloader = DataLoader(collator.dataset, collate_fn=collate_fn,
                                batch_size=batch_size, shuffle=True, drop_last=False, num_workers=num_workers)

        return dataloader

    def valid_dataloader(self, collate_fn=None, batch_size=128, num_workers=0, indices=None, **kwargs):
        graph = self.G

        if indices is not None:
            seed_nodes = indices
        elif isinstance(self.head_node_type, str) and isinstance(self.validation_idx, (Tensor, np.ndarray)):
            seed_nodes = {self.head_node_type: self.validation_idx}
        elif isinstance(self.head_node_type, str) and isinstance(self.validation_idx, dict):
            seed_nodes = {self.head_node_type: self.validation_idx[self.head_node_type]}
        else:
            assert isinstance(self.validation_idx, dict) and len(self.head_node_type) == len(self.validation_idx)
            seed_nodes = self.validation_idx

        collator, collate_fn = self.get_node_collator(graph, seed_nodes, collate_fn, self.neighbor_sampler)

        dataloader = DataLoader(collator.dataset, collate_fn=collate_fn,
                                batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)

        return dataloader

    def test_dataloader(self, collate_fn=None, batch_size=128, num_workers=0, indices=None, **kwargs):
        graph = self.G

        if indices is not None:
            seed_nodes = indices
        elif isinstance(self.head_node_type, str) and isinstance(self.testing_idx, (Tensor, np.ndarray)):
            seed_nodes = {self.head_node_type: self.testing_idx}
        elif isinstance(self.head_node_type, str) and isinstance(self.testing_idx, dict):
            seed_nodes = {self.head_node_type: self.testing_idx[self.head_node_type]}
        else:
            assert isinstance(self.testing_idx, dict) and len(self.head_node_type) == len(self.testing_idx)
            seed_nodes = self.testing_idx

        collator, collate_fn = self.get_node_collator(graph, seed_nodes, collate_fn, self.neighbor_sampler)

        dataloader = DataLoader(collator.dataset, collate_fn=collate_fn,
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

        input_nodes, seeds, blocks = self.graph_sampler.sample_blocks(self.g, items)

        output_nodes = blocks[-1].dstdata[dgl.NID]

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
