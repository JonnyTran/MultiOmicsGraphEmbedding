from typing import List, Union, Optional, Dict, Tuple

import dgl
import dgl.transforms as T
import numpy as np
import pandas as pd
import torch
from dgl import compact_graphs, EID, NID
from dgl.dataloading import EdgePredictionSampler, Sampler, as_edge_prediction_sampler, find_exclude_eids
from dgl.heterograph import DGLBlock, DGLHeteroGraph
from dgl.utils import to_device
from logzero import logger
from ogb.linkproppred import DglLinkPropPredDataset
from pandas import Index, DataFrame, Series
from torch import Tensor
from torch_sparse import SparseTensor
from umap import UMAP

from .node_generator import DGLNodeGenerator
from ..PyG.hetero_generator import HeteroLinkPredDataset
from ..utils import get_relabled_edge_index, is_negative, is_reversed, unreverse_metapath, untag_negative_metapath, \
    edge_index_to_adjs
from ...model.PyG.utils import num_edges
from ...model.utils import tensor_sizes
from ...network.base import SEQUENCE_COL
from ...network.hetero import HeteroNetwork


class DGLLinkGenerator(DGLNodeGenerator):
    nodes_namespace: Dict[str, Series]
    def __init__(self, dataset: Union[DglLinkPropPredDataset, DGLHeteroGraph], sampler: str, neighbor_sizes=None,
                 negative_sampler="uniform", negative_sampling_size=100, pred_metapaths=None, neg_pred_metapaths=None,
                 node_types=None, metapaths=None, head_node_type=None, edge_dir=True, reshuffle_train: float = None,
                 add_reverse_metapaths=True, inductive=True, exclude="all_pred", **kwargs):
        super().__init__(dataset, sampler=sampler, neighbor_sizes=neighbor_sizes, node_types=node_types,
                         metapaths=metapaths, head_node_type=head_node_type, edge_dir=edge_dir, exclude=exclude,
                         reshuffle_train=reshuffle_train, add_reverse_metapaths=add_reverse_metapaths,
                         inductive=inductive, **kwargs)

        self.negative_sampling_size = negative_sampling_size
        self.eval_negative_sampling_size = 1000

        if not hasattr(self, 'pred_metapaths') and pred_metapaths:
            self.pred_metapaths = pred_metapaths
        if not hasattr(self, 'neg_pred_metapaths') and neg_pred_metapaths:
            self.neg_pred_metapaths = neg_pred_metapaths
        else:
            self.neg_pred_metapaths = []

        self.negative_sampler = negative_sampler
        self.exclude = exclude
        self.link_sampler = self.get_link_sampler(self.G, negative_sampling_size=negative_sampling_size,
                                                  negative_sampler=negative_sampler, neighbor_sizes=neighbor_sizes,
                                                  neighbor_sampler=sampler, edge_dir=edge_dir, exclude=exclude)

    @classmethod
    def from_heteronetwork(cls, network: HeteroNetwork, node_attr_cols: List[str] = None,
                           target: str = None, min_count: int = None,
                           expression=False, sequence=False,
                           add_reverse_metapaths=True,
                           label_subset: Optional[Union[Index, np.ndarray]] = None,
                           ntype_subset: Optional[List[str]] = None,
                           split_namespace=False, **kwargs):
        G, classes, nodes, training_idx, validation_idx, testing_idx = \
            network.to_dgl_heterograph(node_attr_cols=node_attr_cols, target=target, min_count=min_count,
                                       expression=expression, sequence=sequence,
                                       label_subset=label_subset, ntype_subset=ntype_subset)

        self = cls(dataset=G, metapaths=G.canonical_etypes, add_reverse_metapaths=add_reverse_metapaths,
                   pred_metapaths=network.pred_metapaths, neg_pred_metapaths=network.neg_pred_metapaths,
                   edge_dir="in", **kwargs)
        self.network = network
        self.classes = classes
        self.n_classes = len(classes) if classes is not None else None
        self.nodes = nodes
        self._name = network._name if hasattr(network, '_name') else ""
        self.head_node_type = list(
            {metapath[0] for metapath in self.pred_metapaths} | {metapath[-1] for metapath in self.pred_metapaths})

        self.split_namespace = split_namespace
        if split_namespace:
            self.nodes_namespace = {}
            self.ntype_mapping = {}
            for ntype, df in network.annotations.items():
                if "namespace" in df.columns:
                    self.nodes_namespace[ntype] = network.annotations[ntype]["namespace"].loc[self.nodes[ntype]]
                    self.ntype_mapping.update(
                        {namespace: ntype for namespace in np.unique(self.nodes_namespace[ntype])})

        # Train/valid/test edge id's
        self.training_idx = {metapath: eidx for metapath, eidx in training_idx.items() \
                             if metapath in self.pred_metapaths + self.neg_pred_metapaths}
        self.validation_idx = {metapath: eidx for metapath, eidx in validation_idx.items() \
                               if metapath in self.pred_metapaths + self.neg_pred_metapaths}
        self.testing_idx = {metapath: eidx for metapath, eidx in testing_idx.items() \
                            if metapath in self.pred_metapaths + self.neg_pred_metapaths}

        # Remove pred metapaths if they were reversed to prevent data leakage
        if self.use_reverse:
            for metapath in self.G.canonical_etypes:
                if is_negative(metapath) and is_reversed(metapath):
                    logger.info(f"Removed {metapath}")
                    self.G.remove_edges(eids=self.G.edges(etype=metapath, form='eid'), etype=metapath)

                elif self.pred_metapaths and unreverse_metapath(metapath) in self.pred_metapaths:
                    logger.info(f"Removed {metapath}")
                    self.G.remove_edges(eids=self.G.edges(etype=metapath, form='eid'), etype=metapath)

                elif self.neg_pred_metapaths and unreverse_metapath(metapath) in self.pred_metapaths:
                    logger.info(f"Removed {metapath}")
                    self.G.remove_edges(eids=self.G.edges(etype=metapath, form='eid'), etype=metapath)

            self.metapaths = [metapath for metapath in self.G.canonical_etypes if self.G.num_edges(etype=metapath)]

        return self

    def process_DglLinkDataset_hetero(self, ogbl: DglLinkPropPredDataset):
        graph: dgl.DGLHeteroGraph = ogbl[0]
        self._name = ogbl.name

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
        split_edge = ogbl.get_edge_split()
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
        for triples, trainvalidtest_idx, mask_name in zip([train_triples, valid_triples, test_triples],
                                                          [self.training_idx, self.validation_idx, self.testing_idx],
                                                          ["train_mask", "valid_mask", "test_mask"]):
            edges_pos, edges_neg = get_relabled_edge_index(triples, global_node_index=self.global_node_index,
                                                           metapaths=self.metapaths, format="dgl")

            # Add valid and test edges to graph, since graph only contains train edges
            for metapath, (src, dst) in edges_pos.items():
                if triples is not train_triples:
                    graph.add_edges(src, dst, etype=metapath)
                trainvalidtest_idx[metapath] = graph.edge_ids(src, dst, etype=metapath)

                edge_mask = torch.zeros(graph.num_edges(etype=metapath), dtype=torch.bool)
                edge_mask[trainvalidtest_idx[metapath]] = 1
                graph.edges[metapath].data[mask_name] = edge_mask

            logger.info(f"{mask_name} num edges: {sum([len(eids) for eids in trainvalidtest_idx.values()])}")

        self.pred_metapaths = list(self.validation_idx.keys())
        self.G = graph

    def get_metapaths(self):
        return self.metapaths

    def get_train_ratio(self):
        n_train_edges = sum([len(eids) for eids in self.training_idx.values()])
        n_valid_edges = sum([len(eids) for eids in self.validation_idx.values()])
        n_test_edges = sum([len(eids) for eids in self.testing_idx.values()])
        return n_train_edges / (n_train_edges + n_valid_edges + n_test_edges)

    def full_batch(self, edge_idx: Tensor = None, mode="test", batch_size=None, device="cpu"):
        T.AddMetaPaths()

        if edge_idx is None:
            edge_idx = {m: torch.cat([self.training_idx[m], self.validation_idx[m], self.testing_idx[m]]) \
                        for m in self.training_idx.keys()}

        if not batch_size:
            batch_size = sum(eid.numel() for eid in edge_idx.values())

        if mode == "test":
            loader = self.test_dataloader(batch_size=batch_size, indices=edge_idx)
        else:
            loader = self.train_dataloader(batch_size=batch_size, indices=edge_idx)

        outputs = next(iter(loader))
        if device != "cpu":
            outputs = to_device(outputs, device)

        return outputs

    def get_node_metadata(self, global_node_index: Dict[str, Tensor],
                          embeddings: Dict[str, Tensor],
                          weights: Optional[Dict[str, Series]] = None,
                          losses: Dict[str, Tensor] = None) -> DataFrame:
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
        if not hasattr(self, "node_metadata"):
            self.create_node_metadata(self.network, nodes=self.nodes)

        global_node_index = {ntype: nids.numpy() for ntype, nids in global_node_index.items() \
                             if ntype in embeddings}

        # Concatenated list of node embeddings and other metadata
        nodes_emb = {ntype: embeddings[ntype].detach().numpy() for ntype in embeddings}
        nodes_emb = np.concatenate([nodes_emb[ntype] for ntype in global_node_index])

        nodes_train_valid_test = np.vstack([
            np.concatenate([self.G.nodes[ntype].data['train_mask'][nids].numpy() \
                            for ntype, nids in global_node_index.items()]),
            np.concatenate([self.G.nodes[ntype].data['valid_mask'][nids].numpy() \
                            for ntype, nids in global_node_index.items()]),
            np.concatenate([self.G.nodes[ntype].data['test_mask'][nids].numpy() \
                            for ntype, nids in global_node_index.items()])],
        ).T
        nodes_train_valid_test = np.array(["Train", "Valid", "Test"])[nodes_train_valid_test.argmax(1)]

        if losses:
            node_losses = np.concatenate([losses[ntype] if ntype in losses else \
                                              [None] * global_node_index[ntype].size \
                                          for ntype in global_node_index])
        else:
            node_losses = None

        # Metadata
        # Build node metadata dataframe from concatenated lists of node metadata for multiple ntypes
        df = pd.DataFrame(
            {"node": np.concatenate([self.nodes[ntype][global_node_index[ntype]] for ntype in global_node_index]),
             "ntype": np.concatenate([[ntype] * global_node_index[ntype].shape[0] for ntype in global_node_index]),
             "train_valid_test": nodes_train_valid_test,
             "loss": node_losses},
            index=pd.Index(np.concatenate([global_node_index[ntype] for ntype in global_node_index]), name="nid"))

        tsne = UMAP(n_components=2, n_jobs=-1)
        # tsne = MulticoreTSNE.MulticoreTSNE(n_components=2, n_jobs=-1)
        nodes_pos = tsne.fit_transform(nodes_emb)
        nodes_pos = {node_name: pos for node_name, pos in zip(df.index, nodes_pos)}
        df[['pos1', 'pos2']] = np.vstack(df.index.map(nodes_pos))

        # Reset index
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
                if isinstance(weights[ntype], Tensor) else weights[ntype] for ntype in weights}
            nodes_weight = np.concatenate([nodes_weight[ntype] for ntype in global_node_index]).astype(bool)

            return df.loc[nodes_weight]

        return df

    def get_link_sampler(self, G: DGLHeteroGraph, negative_sampling_size: int, negative_sampler: str,
                         neighbor_sizes: List[str], neighbor_sampler: str, edge_dir="in", exclude="all_pred",
                         collate_fn=None) -> Sampler:
        if G is None:
            G = self.G
        if negative_sampling_size is None:
            negative_sampling_size = self.negative_sampling_size
        if neighbor_sizes is None:
            neighbor_sizes = self.neighbor_sizes
        if neighbor_sampler is None:
            neighbor_sampler = self.sampler
        if edge_dir is None:
            edge_dir = self.edge_dir

        neighbor_sampler = self.get_neighbor_sampler(neighbor_sizes=neighbor_sizes, etypes=G.canonical_etypes,
                                                     sampler=neighbor_sampler, edge_dir=edge_dir)

        if isinstance(negative_sampler, str):
            if negative_sampler.lower() == "Uniform".lower():
                negative_sampler = dgl.dataloading.negative_sampler.Uniform(negative_sampling_size)
            elif negative_sampler.lower() == "GlobalUniform".lower():
                negative_sampler = dgl.dataloading.negative_sampler.GlobalUniform(negative_sampling_size)
            elif negative_sampler.lower() == "PerSourceUniform".lower():
                negative_sampler = dgl.dataloading.negative_sampler.PerSourceUniform(negative_sampling_size)
            else:
                raise Exception(f"{negative_sampler} must be in one of [Uniform, GlobalUniform, PerSourceUniform]")

        if exclude is None:
            exclude = "reverse_id" if self.use_reverse else "self"
        elif exclude == "all_pred":
            exclude = lambda x: {etype: self.G.edges(etype=self.pred_metapaths[0], form="eid") \
                                 for etype in self.pred_metapaths + self.neg_pred_metapaths}

        options = dict(sampler=neighbor_sampler,
                       exclude=exclude,
                       reverse_etypes=self.reverse_etypes if self.use_reverse else None,
                       reverse_eids=self.reverse_eids if self.use_reverse else None,
                       negative_sampler=negative_sampler)

        if collate_fn == LinkPredPyGCollator.__name__:
            sampler = LinkPredPyGCollator(node_types=self.node_types, nodes_namespace=self.nodes_namespace,
                                          ntype_mapping=self.ntype_mapping, pred_metapaths=self.pred_metapaths,
                                          neg_pred_metapaths=self.neg_pred_metapaths, network=self.network, **options)

        elif collate_fn == EdgePredictionSamplerForNodeClassification.__name__:
            sampler = EdgePredictionSamplerForNodeClassification(classes=self.classes, nodes=self.nodes,
                                                                 pred_metapaths=self.pred_metapaths,
                                                                 neg_pred_metapaths=self.neg_pred_metapaths,
                                                                 **options)

        else:
            sampler = as_edge_prediction_sampler(**options)

        return sampler


    def train_dataloader(self, collate_fn=None, batch_size=128, num_workers=0, indices=None, drop_last=False,
                         device=None, verbose=False, **kwargs):
        if self.inductive:
            graph = self.transform_heterograph(self.G, edge_mask=self.G.edata["train_mask"], verbose=verbose)
            graph_sampler = self.get_link_sampler(graph, negative_sampling_size=self.negative_sampling_size,
                                                  negative_sampler=self.negative_sampler,
                                                  neighbor_sizes=self.neighbor_sizes, neighbor_sampler=self.sampler,
                                                  edge_dir=self.edge_dir, exclude=self.exclude, collate_fn=collate_fn)
            indices = {etype: torch.arange(graph.num_edges(etype=etype)) \
                       for etype in self.pred_metapaths + self.neg_pred_metapaths}
        else:
            graph = self.G
            graph_sampler = self.link_sampler

        if indices is None:
            indices = self.training_idx

        logger.info(
            f"Train dataset (inductive={self.inductive}) pred edges: \n{tensor_sizes(indices)}") if verbose else None
        # sampler = LinkPredPyGCollator(**args)
        dataloader = dgl.dataloading.DataLoader(graph, indices=indices, graph_sampler=graph_sampler,
                                                batch_size=batch_size, shuffle=True, drop_last=drop_last,
                                                device=device, num_workers=num_workers)

        return dataloader

    def valid_dataloader(self, collate_fn=None, batch_size=128, num_workers=0, indices=None, drop_last=False,
                         device=None, verbose=False, **kwargs):
        if self.inductive:
            edge_mask = {etype: valid_mask | self.G.edata["train_mask"][etype] \
                         for etype, valid_mask in self.G.edata["valid_mask"].items()}
            graph = self.transform_heterograph(self.G, edge_mask=edge_mask, verbose=verbose)

            graph_sampler = self.get_link_sampler(graph, negative_sampling_size=self.negative_sampling_size,
                                                  negative_sampler=self.negative_sampler,
                                                  neighbor_sizes=self.neighbor_sizes, neighbor_sampler=self.sampler,
                                                  edge_dir=self.edge_dir, exclude=self.exclude, collate_fn=collate_fn)

            indices = {etype: torch.arange(graph.num_edges(etype=etype)) \
                       for etype in self.pred_metapaths + self.neg_pred_metapaths}

        else:
            graph = self.G
            graph_sampler = self.link_sampler

        if indices is None:
            indices = self.validation_idx

        logger.info(
            f"Valid dataset (inductive={self.inductive}) pred edges: \n{tensor_sizes(indices)}") if verbose else None

        dataloader = dgl.dataloading.DataLoader(graph, indices=indices, graph_sampler=graph_sampler,
                                                batch_size=batch_size, shuffle=True, drop_last=drop_last, device=device,
                                                num_workers=num_workers, )
        return dataloader

    def test_dataloader(self, collate_fn=None, batch_size=128, num_workers=0, indices=None, drop_last=False,
                        device=None, verbose=False, **kwargs):
        if self.inductive:
            edge_mask = {etype: test_mask | self.G.edata["train_mask"][etype] | self.G.edata["valid_mask"][etype] \
                         for etype, test_mask in self.G.edata["test_mask"].items()}
            graph = self.transform_heterograph(self.G, edge_mask=edge_mask, verbose=verbose)
            graph_sampler = self.get_link_sampler(graph, negative_sampling_size=self.negative_sampling_size,
                                                  negative_sampler=self.negative_sampler,
                                                  neighbor_sizes=self.neighbor_sizes, neighbor_sampler=self.sampler,
                                                  edge_dir=self.edge_dir, exclude=self.exclude, collate_fn=collate_fn)
            indices = {etype: torch.arange(graph.num_edges(etype=etype)) \
                       for etype in self.pred_metapaths + self.neg_pred_metapaths}
        else:
            graph = self.G
            graph_sampler = self.link_sampler

        if indices is None:
            indices = self.testing_idx

        logger.info(
            f"Test dataset (inductive={self.inductive}) pred edges: \n{tensor_sizes(indices)}") if verbose else None

        dataloader = dgl.dataloading.DataLoader(graph, indices=indices, graph_sampler=graph_sampler,
                                                batch_size=batch_size, shuffle=True, drop_last=drop_last, device=device,
                                                num_workers=num_workers)
        return dataloader

    def train_nodeclf_dataloader(self, collate_fn=None, batch_size=128, num_workers=4, seed_nodes=None, drop_last=False,
                                 device=None, verbose=False, **kwargs):
        if self.inductive:
            graph = self.transform_heterograph(self.G, edge_mask=self.G.edata["train_mask"], verbose=verbose)
        else:
            graph = self.G

        graph_sampler = self.get_link_sampler(graph, negative_sampling_size=0,
                                              negative_sampler=None,
                                              neighbor_sizes=None, neighbor_sampler=self.sampler,
                                              edge_dir=self.edge_dir, exclude=self.exclude,
                                              collate_fn=EdgePredictionSamplerForNodeClassification.__name__)
        head_node_type = graph_sampler.head_node_type
        seed_nodes = {
            head_node_type: torch.arange(self.G.num_nodes(head_node_type))[self.G.ndata["train_mask"][head_node_type]]}

        logger.info(
            f"Test dataset (inductive={self.inductive}) pred edges: \n{tensor_sizes(seed_nodes)}") if verbose else None

        dataloader = dgl.dataloading.DataLoader(graph, indices=seed_nodes, graph_sampler=graph_sampler,
                                                batch_size=batch_size, shuffle=True, drop_last=drop_last, device=device,
                                                num_workers=num_workers)

        return dataloader

    def valid_nodeclf_dataloader(self, collate_fn=None, batch_size=128, num_workers=4, seed_nodes=None, drop_last=False,
                                 device=None, verbose=False, **kwargs):
        if self.inductive:
            edge_mask = {etype: valid_mask | self.G.edata["train_mask"][etype] \
                         for etype, valid_mask in self.G.edata["valid_mask"].items()}
            graph = self.transform_heterograph(self.G, edge_mask=edge_mask, verbose=verbose)
        else:
            graph = self.G

        graph_sampler = self.get_link_sampler(graph, negative_sampling_size=0,
                                              negative_sampler=None,
                                              neighbor_sizes=None, neighbor_sampler=self.sampler,
                                              edge_dir=self.edge_dir, exclude=self.exclude,
                                              collate_fn=EdgePredictionSamplerForNodeClassification.__name__)
        head_node_type = graph_sampler.head_node_type
        seed_nodes = {
            head_node_type: torch.arange(self.G.num_nodes(head_node_type))[self.G.ndata["valid_mask"][head_node_type]]}

        logger.info(
            f"Test dataset (inductive={self.inductive}) pred edges: \n{tensor_sizes(seed_nodes)}") if verbose else None

        dataloader = dgl.dataloading.DataLoader(graph, indices=seed_nodes, graph_sampler=graph_sampler,
                                                batch_size=batch_size, shuffle=True, drop_last=drop_last, device=device,
                                                num_workers=num_workers)

        return dataloader

    def test_nodeclf_dataloader(self, collate_fn=None, batch_size=128, num_workers=4, seed_nodes=None, drop_last=False,
                                device=None, verbose=False, **kwargs):
        if self.inductive:
            edge_mask = {etype: test_mask | self.G.edata["train_mask"][etype] | self.G.edata["valid_mask"][etype] \
                         for etype, test_mask in self.G.edata["test_mask"].items()}
            graph = self.transform_heterograph(self.G, edge_mask=edge_mask, verbose=verbose)
        else:
            graph = self.G

        graph_sampler = self.get_link_sampler(graph, negative_sampling_size=0,
                                              negative_sampler=None,
                                              neighbor_sizes=None, neighbor_sampler=self.sampler,
                                              edge_dir=self.edge_dir, exclude=self.exclude,
                                              collate_fn=EdgePredictionSamplerForNodeClassification.__name__)
        head_node_type = graph_sampler.head_node_type
        seed_nodes = {
            head_node_type: torch.arange(self.G.num_nodes(head_node_type))[self.G.ndata["test_mask"][head_node_type]]}

        logger.info(
            f"Test dataset (inductive={self.inductive}) pred edges: \n{tensor_sizes(seed_nodes)}") if verbose else None

        dataloader = dgl.dataloading.DataLoader(graph, indices=seed_nodes, graph_sampler=graph_sampler,
                                                batch_size=batch_size, shuffle=True, drop_last=drop_last, device=device,
                                                num_workers=num_workers)

        return dataloader


class EdgePredictionSamplerForNodeClassification(EdgePredictionSampler):
    def __init__(self, classes: np.ndarray, nodes: Dict[str, Index],
                 pred_metapaths: List[Tuple], neg_pred_metapaths: List[Tuple],
                 sampler=None, exclude=None, reverse_eids=None, reverse_etypes=None, negative_sampler=None,
                 prefetch_labels=None):
        self.pred_metapaths = pred_metapaths
        self.neg_pred_metapaths = neg_pred_metapaths

        self.nodes = nodes
        self.classes = classes

        self.head_node_type = list({m[0] for m in pred_metapaths + neg_pred_metapaths})[0]
        self.go_ntype = list({m[-1] for m in pred_metapaths + neg_pred_metapaths})[0]

        self.go_nids = torch.tensor(nodes[self.go_ntype].get_indexer_for(classes))
        self.classes_nodes = {self.go_ntype: self.go_nids}
        super().__init__(sampler, exclude, reverse_eids, reverse_etypes, negative_sampler, prefetch_labels)

    def sample(self, g: DGLHeteroGraph, seed_nodes: Dict[str, Tensor]):
        seed_nodes.update(self.classes_nodes)
        exclude = self.exclude

        # Prepare g
        new_g = g.clone()
        for metapath in self.pred_metapaths + self.neg_pred_metapaths:
            new_g.remove_edges(new_g.edges(etype=metapath, form="eid"), etype=metapath)

        labels = new_g.ndata["label"][self.head_node_type][seed_nodes[self.head_node_type]]
        seed_edges = {}

        # Add all positive annotations to graph
        pos_edges = labels.nonzero().T
        pos_edges[1] = self.go_nids[pos_edges[1]]  # Relabel GO_term classes to global node index
        new_g.add_edges(pos_edges[0], pos_edges[1], etype=self.pred_metapaths[0])
        seed_edges[self.pred_metapaths[0]] = torch.arange(pos_edges.size(1))

        # Add all negative annotations to graph
        neg_edges = (1 - labels).nonzero().T
        neg_edges[1] = self.go_nids[neg_edges[1]]  # Relabel GO_term classes to global node index
        new_g.add_edges(neg_edges[0], neg_edges[1], etype=self.neg_pred_metapaths[0])
        seed_edges[self.neg_pred_metapaths[0]] = torch.arange(neg_edges.size(1))

        # Create positive and negative subgraphs
        pair_graph = dgl.edge_subgraph(new_g, seed_edges, relabel_nodes=False, output_device=self.output_device)

        pos_graph = dgl.edge_subgraph(pair_graph, seed_edges,
                                      relabel_nodes=False, output_device=self.output_device)
        neg_graph = pos_graph.clone()
        for metapath in self.pred_metapaths + self.neg_pred_metapaths:
            neg_graph.remove_edges(neg_graph.edges(etype=metapath, form="eid"), etype=metapath)

        eids = pos_graph.edata[EID]

        pos_graph, neg_graph = compact_graphs([pos_graph, neg_graph])

        pos_graph.edata[EID] = eids
        seed_nodes = pos_graph.ndata[NID]

        exclude_eids = find_exclude_eids(
            g, seed_nodes, exclude, self.reverse_eids, self.reverse_etypes,
            self.output_device)

        input_nodes, _, blocks = self.sampler.sample(g, seed_nodes, exclude_eids)

        return self.assign_lazy_features((input_nodes, pos_graph, neg_graph, blocks))


class LinkPredPyGCollator(EdgePredictionSampler):
    def __init__(self, seq_tokenizer=None, node_types=None, nodes_namespace=None, ntype_mapping=None, pred_metapaths=[],
                 neg_pred_metapaths=[], network: HeteroNetwork = None, sampler=None, exclude=None, reverse_eids=None,
                 reverse_etypes=None, negative_sampler=None, prefetch_labels=None):

        self.triples, *_ = network.get_triples(pred_metapaths, positive=True)

        # Adjacency of pos edges (for neg sampling)
        edge_index_dict, _ = get_relabled_edge_index(
            triples=self.triples,
            global_node_index={ntype: torch.arange(len(nodelist)) for ntype, nodelist in network.nodes.items()},
            metapaths=pred_metapaths,
            relation_ids_all=self.triples["relation"].unique())
        self.triples_adj: Dict[Tuple[str, str, str], SparseTensor] = edge_index_to_adjs(edge_index_dict,
                                                                                        nodes=network.nodes)

        self.negative_sampler = negative_sampler
        if seq_tokenizer is not None:
            self.seq_tokenizer = seq_tokenizer

        self.go_ntype = list({m[-1] for m in pred_metapaths})[0]
        self.nodes_namespace = nodes_namespace
        self.ntype_mapping = ntype_mapping
        self.node_types = node_types

        self.pred_metapaths = pred_metapaths
        self.neg_pred_metapaths = neg_pred_metapaths
        super().__init__(sampler, exclude, reverse_eids, reverse_etypes, negative_sampler, prefetch_labels)

    def get_neg_sampling_size(self, *args, **kwargs):
        return self.negative_sampler.k // 2

    def sample(self, g: DGLHeteroGraph, seed_edges: Dict[str, Tensor]):
        input_nodes, pos_graph, neg_graph, blocks = super().sample(g, seed_edges)
        blocks: List[DGLBlock]
        pos_graph: DGLHeteroGraph

        X = {}
        for i, block in enumerate(blocks):
            edge_index_dict = {}
            for metapath in block.canonical_etypes:
                if block.num_edges(etype=metapath) == 0:
                    continue
                edge_index_dict[metapath] = torch.stack(block.edges(etype=metapath, order="srcdst"), dim=0)
            X.setdefault("edge_index_dict", []).append(edge_index_dict)

            sizes = {}
            for ntype in block.ntypes:
                sizes[ntype] = torch.tensor((block.num_src_nodes(ntype), block.num_dst_nodes(ntype)))
            X.setdefault("sizes", []).append(sizes)

            global_node_index = {ntype: nid for ntype, nid in block.srcdata[dgl.NID].items() if nid.numel() > 0}
            X.setdefault("global_node_index", []).append(global_node_index)

        X["x_dict"] = {ntype: feat \
                       for ntype, feat in blocks[0].srcdata["feat"].items() \
                       if feat.size(0) != 0}

        if SEQUENCE_COL in blocks[0].srcdata and len(blocks[0].srcdata[SEQUENCE_COL]):
            for ntype, feat in blocks[0].srcdata[SEQUENCE_COL].items():
                X.setdefault("sequences", {})[ntype] = self.seq_tokenizer.encode_sequences(X, ntype=ntype,
                                                                                           max_length=None)

        # Convert DGL sampled seed edges into PyG format
        output_nodes = {ntype: ids for ntype, ids in pos_graph.dstdata["_ID"].items() if ids.numel()}

        pos_global_edges, pos_edge, neg_edge, head_batch, tail_batch = {}, {}, {}, {}, {}
        for metapath in pos_graph.canonical_etypes:
            if pos_graph.num_edges(etype=metapath) == 0: continue

            if metapath in self.pred_metapaths:
                pos_edge[metapath] = torch.stack(pos_graph.edges(etype=metapath, order='srcdst'), dim=0)

                pos_global_edges[metapath] = pos_edge[metapath].clone().detach()
                pos_global_edges[metapath][0] = input_nodes[metapath[0]][pos_global_edges[metapath][0]]
                pos_global_edges[metapath][1] = input_nodes[metapath[-1]][pos_global_edges[metapath][1]]

            elif metapath in self.neg_pred_metapaths:
                neg_edge[untag_negative_metapath(metapath)] = torch.stack(
                    pos_graph.edges(etype=metapath, order='srcdst'), dim=0)

        # Negative sampling
        if hasattr(self, 'nodes_namespace'):
            pos_global_edges = HeteroLinkPredDataset.split_edge_index_by_nodes_namespace(self, pos_global_edges,
                                                                                         batch_to_global=None)
        head_batch, tail_batch = HeteroLinkPredDataset.generate_negative_sampling(self, edge_pos=pos_global_edges,
                                                                                  global_node_index=output_nodes,
                                                                                  max_negative_sampling_size=500,
                                                                                  mode="test")

        edge_true = {}
        if hasattr(self, 'nodes_namespace'):
            edge_true['edge_pos'] = HeteroLinkPredDataset.split_edge_index_by_nodes_namespace(self,
                                                                                              edge_index_dict=pos_edge,
                                                                                              batch_to_global=input_nodes)
        else:
            edge_true['edge_pos'] = pos_edge

        if num_edges(neg_edge):
            if hasattr(self, 'nodes_namespace'):
                edge_true['edge_neg'] = HeteroLinkPredDataset.split_edge_index_by_nodes_namespace(self,
                                                                                                  edge_index_dict=neg_edge,
                                                                                                  batch_to_global=input_nodes)
            else:
                edge_true['edge_neg'] = neg_edge

        edge_true.update({"head_batch": head_batch, "tail_batch": tail_batch, })

        return X, edge_true, {}
