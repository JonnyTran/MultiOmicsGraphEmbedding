from collections import defaultdict
from typing import Dict, Tuple, Union, List, Any, Optional

import dgl
import networkx as nx
import numpy as np
import pandas as pd
import torch
from openomics import MultiOmics
from openomics.database.ontology import Ontology
from openomics.utils.df import concat_uniques
from pandas import Series, Index
from torch import Tensor
from torch_geometric.data import HeteroData

from moge.dataset.utils import get_edge_index_values, get_edge_index_dict, tag_negative_metapath, \
    untag_negative_metapath
from moge.network.attributed import AttributedNetwork
from moge.network.base import SEQUENCE_COL
from moge.network.train_test_split import TrainTestSplit
from moge.network.utils import filter_multilabel


class HeteroNetwork(AttributedNetwork, TrainTestSplit):
    def __init__(self, multiomics: MultiOmics, node_types: list, layers: Dict[Tuple[str], nx.Graph],
                 annotations=True, ) -> None:
        """

        Args:
            multiomics: MultiOmics object containing annotations
            node_types: Node types
            layers: A dict of edge types tuple and networkx.Graph/Digraph containing heterogeneous edges
            annotations: Whether to process annotation data, default True
        """
        self.multiomics: MultiOmics = multiomics
        self.node_types = node_types
        self.networks: Dict[Tuple, nx.Graph] = {}

        networks = {}
        for src_etype_dst, GraphClass in layers.items():
            networks[src_etype_dst] = GraphClass()

        super(HeteroNetwork, self).__init__(networks=networks, multiomics=multiomics, annotations=annotations)

    def process_network(self):
        self.nodes = {}
        self.node_to_modality = {}

        for source_target, network in self.networks.items():
            for node_type in self.node_types:
                network.add_nodes_from(self.multiomics[node_type].get_genes_list(), modality=node_type)

        for node_type in self.node_types:
            self.nodes[node_type] = self.multiomics[node_type].get_genes_list()

            for gene in self.multiomics[node_type].get_genes_list():
                self.node_to_modality[gene] = self.node_to_modality.setdefault(gene, []) + [node_type, ]
            print(node_type, " nodes:", len(self.nodes[node_type]), self.multiomics[node_type].gene_index)

        print("Total nodes:", len(self.get_node_list()))
        self.nodes: Dict[str, pd.Index] = pd.Series(self.nodes)
        self.node_to_modality = pd.Series(self.node_to_modality)

    def process_annotations(self):
        self.annotations = {}
        for modality in self.node_types:
            annotation = self.multiomics[modality].get_annotations()
            self.annotations[modality] = annotation

        self.annotations = pd.Series(self.annotations)
        print("All annotation columns (union):",
              {col for _, annotations in self.annotations.items() for col in annotations.columns.tolist()})

    def process_feature_tranformer(self, delimiter="\||;", filter_label=None, min_count=0, verbose=False):
        self.delimiter = delimiter
        if not hasattr(self, "all_annotations"):
            annotations_list = []

            for modality in self.node_types:
                annotation = self.multiomics[modality].get_annotations()
                annotation["omic"] = modality
                annotations_list.append(annotation)

            self.all_annotations = pd.concat(annotations_list, join="inner", copy=True)
            self.all_annotations = self.all_annotations.groupby(self.all_annotations.index).agg(
                {k: concat_uniques for k in self.all_annotations.columns})

        print("Annotation columns:", self.all_annotations.columns.tolist()) if verbose else None
        self.feature_transformer = self.get_feature_transformers(self.all_annotations, self.node_list, filter_label,
                                                                 min_count, delimiter, verbose=verbose)

    def add_nodes(self, nodes: List[str], ntype: str, annotations: pd.DataFrame = None):
        self.node_types.append(ntype)
        self.nodes[ntype] = pd.Index(nodes)

        if isinstance(annotations, pd.DataFrame) and not annotations.empty:
            self.annotations[ntype] = annotations.loc[nodes]

    def add_edges(self, edgelist: List[Union[Tuple[str, str]]], etype: Tuple[str, str, str], database: str,
                  directed=True,
                  **kwargs):
        src_type, dst_type = etype[0], etype[-1]
        if etype not in self.networks:
            if directed:
                self.networks[etype] = nx.DiGraph()
            else:
                self.networks[etype] = nx.Graph()

        self.networks[etype].add_edges_from(edgelist, source=src_type, target=dst_type, database=database, **kwargs)
        unq_sources, unq_targets = {u for u, v, *_ in edgelist}, {v for u, v, *_ in edgelist}

        print(f"{etype}: {len(edgelist)} edges added between "
              f"{len(unq_sources)} {src_type}'s and {len(unq_targets)} {dst_type}'s.")

    def add_edges_from_ontology(self, ontology: Ontology, nodes: Optional[List[str]] = None, ntype: str = "GO_term",
                                reverse_edges=True, etypes: List[Tuple[str, str, str]] = []):
        """

        Args:
            ontology (Ontology): an openomics.Ontology object that metadata and contains edges among the terms.
            nodes (List[str]): A subset of nodes from ontology to extract edges from
            ntype (str): Name of the ontology node type
        """
        if nodes is None:
            nodes = ontology.node_list

        # Add ontology nodes
        if "def" in ontology.data.columns:
            ontology.data[SEQUENCE_COL] = ontology.data["def"]
        self.add_nodes(nodes=nodes, ntype=ntype, annotations=ontology.data)

        # Add ontology edges
        edge_types = {e for u, v, e in ontology.network.edges}
        graph = ontology.network
        if reverse_edges:
            graph = nx.reverse(graph, copy=True)

        edge_index_dict = get_edge_index_dict(graph, nodes=nodes, metapaths=edge_types,
                                              format="nx", d_ntype=ntype)
        for metapath, edgelist in edge_index_dict.items():
            if len(edgelist) < 10: continue
            if etypes and (metapath not in etypes and (isinstance(metapath, tuple) and metapath[1] not in etypes)):
                continue
            self.add_edges(edgelist, etype=metapath, database=ontology.name(), directed=True)

    def add_edges_from_annotations(self, ontology: Ontology, nodes: Optional[List[str]],
                                   src_ntype: str, dst_ntype: str,
                                   train_date: str, valid_date: str, test_date: str, split_etype: str = None,
                                   d_etype="associated", src_node_col="gene_name", use_neg_annotations=False):
        """

        Args:
            ontology (): an openomics.Ontology object that contains annotations data.
            nodes (): A dict of all nodes in the Hetero network.
            src_ntype (): The ntype name of annotation source nodes.
            dst_ntype (): The ntype name of annotation destination nodes.
            train_date (): A date format "YYYY-MM-DD" to separate all annotations before this date.
            valid_date (): A date format "YYYY-MM-DD" to separate all annotations before this date and after train_date
            test_date (): A date format "YYYY-MM-DD" to separate all annotations before this date and after valid_date
            split_etype (): column name in the annotation which contain the etype values to split edges by.
            d_etype (): Default etype name for annotation edges when `split_etype` is None.
            src_node_col (): The column name in the annotation dataframe that contain the src_ntype node names.
        """
        # if add_annotation_as_edges and hasattr(self, "network"):
        #     for ntype in self.network.annotations.index.drop(["MessengerRNA", "Protein"], errors="ignore"):
        #         annotations = self.network.annotations[ntype]["go_id"].dropna()
        #         source_ntype = annotations.index.name
        #         nx_graph = nx.from_pandas_edgelist(annotations.explode().to_frame().reset_index().dropna(),
        #                                            source=source_ntype, target="go_id", create_using=nx.Graph)
        #         metapath = (ntype, "associated", go_ntype)
        #         self.metapaths.append(metapath)
        #
        #         edge_index = get_edge_index_values(nx_graph, nodes_A=self.nodes[metapath[0]], nodes_B=go_nodes)
        #         self.G[metapath].edge_index = edge_index
        #         print(metapath, nx_graph.number_of_edges())
        #
        #         if self.use_reverse:
        #             rev_metapath = reverse_metapath_name(metapath)
        #             self.metapaths.append(rev_metapath)
        #
        #             self.G[rev_metapath].edge_index = edge_index[[1, 0], :]
        #             print(rev_metapath, nx_graph.number_of_edges())
        groupby = [src_node_col, split_etype] if split_etype else [src_node_col]
        assert src_ntype in self.node_types
        self._name = f"{src_ntype}-{dst_ntype}-{train_date}"

        # Split annotations between genes and GO terms by dates
        train_ann, valid_ann, test_ann = ontology.annotation_train_val_test_split(
            train_date=train_date, valid_date=valid_date, test_date=test_date,
            groupby=groupby, filter_go_id=nodes)

        if split_etype:
            nx_options = dict(edge_key=split_etype, create_using=nx.MultiGraph, edge_attr=True)
            self.pred_metapaths = [(src_ntype, etype, dst_ntype) \
                                   for etype in train_ann.reset_index()[split_etype].unique()]
        else:
            nx_options = dict(edge_key=None, create_using=nx.Graph, edge_attr=None)
            self.pred_metapaths = [(src_ntype, d_etype, dst_ntype)]

        dst_node_col = ontology.data.index.name
        self.neg_pred_metapaths = []
        node_lists = []

        # Process train, validation, and test annotations
        for go_ann in [train_ann, valid_ann, test_ann]:
            is_train = go_ann is train_ann
            is_valid = go_ann is valid_ann
            is_test = go_ann is test_ann

            if is_train:
                print("Train:")
            elif is_valid:
                print("Valid:")
            elif is_test:
                print("Test:")

            # True Positive links
            pos_annotations = go_ann[dst_node_col].dropna().explode().to_frame().reset_index()
            pos_graph = nx.from_pandas_edgelist(pos_annotations, source=src_node_col, target=dst_node_col, **nx_options)
            if split_etype:
                metapaths = {(src_ntype, etype, dst_ntype) for u, v, etype in pos_graph.edges}
            else:
                metapaths = set(self.pred_metapaths)

            pos_edge_list_dict = get_edge_index_dict(pos_graph, nodes=self.nodes,
                                                     metapaths=metapaths.intersection(self.pred_metapaths), format="nx")


            for etype, edges in pos_edge_list_dict.items():
                if etype not in self.pred_metapaths or len(edges) == 0: continue

                edges = self.label_edge_trainvalidtest(edges, train=is_train, valid=is_valid, test=is_test)
                self.add_edges(edges, etype=etype, database=ontology.name(), directed=True, )

            # True Negative links
            if use_neg_annotations and "neg_go_id" in go_ann.columns:
                neg_annotations = go_ann["neg_go_id"].dropna().explode().to_frame().reset_index()
                neg_graph = nx.from_pandas_edgelist(neg_annotations, source=src_node_col, target="neg_go_id",
                                                    **nx_options)
                if split_etype:
                    metapaths = {(src_ntype, etype, dst_ntype) for u, v, etype in neg_graph.edges}
                else:
                    metapaths = set(self.pred_metapaths)
                neg_edge_list_dict = get_edge_index_dict(neg_graph, nodes=self.nodes,
                                                         metapaths=metapaths.intersection(self.pred_metapaths),
                                                         format="nx")
                for etype, edges in neg_edge_list_dict.items():
                    if etype not in self.pred_metapaths or len(edges) == 0: continue
                    if etype not in self.neg_pred_metapaths:
                        self.neg_pred_metapaths.append(etype)

                    etype = tag_negative_metapath(etype)
                    edges = self.label_edge_trainvalidtest(edges, train=is_train, valid=is_valid, test=is_test)
                    self.add_edges(edges, etype=etype, database=ontology.name(), directed=True, )

            # Train/valid/test split of nodes
            node_lists.append({
                src_ntype: set(pos_annotations[src_node_col].unique()).intersection(self.nodes[src_ntype]),
                dst_ntype: set(pos_annotations[dst_node_col].unique()).intersection(self.nodes[dst_ntype]),
            })

        # Set train/valid/test mask of all nodes on hetero graph
        train_nodes, valid_nodes, test_nodes = node_lists
        self.train_nodes, self.valid_nodes, self.test_nodes = self.get_all_nodes_mask(train_nodes,
                                                                                      valid_nodes, test_nodes)

        # Set train/valid/test mask of edges on hetero graph if they're incident to the train/valid/test nodes
        # for metapath, nxgraph in self.networks.items():
        #     if metapath in self.pred_metapaths or metapath in self.neg_pred_metapaths:
        #         continue
        #     edge_attrs = self.get_all_edges_mask(nxgraph.edges, metapath=metapath, train_nodes=self.train_nodes,
        #                                          valid_nodes=self.valid_nodes, test_nodes=self.test_nodes)
        #     nx.set_edge_attributes(nxgraph, edge_attrs)

    def get_triples(self, metapaths: List[Tuple[str, str, str]], positive: bool = False, negative: bool = False) \
            -> Tuple[Dict[str, Tensor], Tensor, Tensor, Tensor]:
        """

        Args:
            metapaths (): External metapath list where the triple's `relation` field gets its ordinal ID from.
            positive (): Whether to only retrieve positive edge types
            negative (): Whether to only retrieve negative edge types

        Returns:

        """
        assert positive != negative
        triples = defaultdict(lambda: [])
        training_idx, validation_idx, testing_idx = [], [], []

        pred_metapaths = self.pred_metapaths if positive else self.neg_pred_metapaths

        for metapath in pred_metapaths:
            head_type, tail_type = metapath[0], metapath[-1]
            if negative:
                if metapaths and metapath not in metapaths: continue
                metapath_idx = metapaths.index(metapath)
                metapath = tag_negative_metapath(metapath)

            elif positive:
                if metapaths and metapath not in metapaths: continue
                metapath_idx = metapaths.index(metapath)

            edge_index, edge_attr = get_edge_index_values(self.networks[metapath],
                                                          nodes_A=self.nodes[head_type],
                                                          nodes_B=self.nodes[tail_type],
                                                          edge_attrs=["train_mask", "valid_mask", "test_mask"],
                                                          format="pyg")
            num_edges = edge_index.size(1)

            triples["head" if positive else "head_neg"].append(edge_index[0])
            triples["tail" if positive else "tail_neg"].append(edge_index[1])
            triples["relation" if positive else "relation_neg"].append(torch.tensor([metapath_idx] * num_edges))
            # triples["head_type"].extend([head_type] * num_edges)
            # triples["tail_type"].extend([tail_type] * num_edges)

            training_idx.append(edge_attr["train_mask"])
            validation_idx.append(edge_attr["valid_mask"])
            testing_idx.append(edge_attr["test_mask"])

        triples = {key: torch.cat(values, dim=0) if isinstance(values[0], Tensor) else values \
                   for key, values in triples.items()}

        edge_idx = torch.arange(len(triples['head' if positive else "head_neg"]))
        training_idx, validation_idx, testing_idx = edge_idx[torch.cat(training_idx)], \
                                                    edge_idx[torch.cat(validation_idx)], \
                                                    edge_idx[torch.cat(testing_idx)]

        return triples, training_idx, validation_idx, testing_idx

    @property
    def num_nodes_dict(self):
        return {node: nid.shape[0] for node, nid in self.nodes.items()}

    def get_aggregated_network(self):
        G = nx.compose_all(list(self.networks.values()))
        return G

    def filter_sequence_nodes(self):
        """
        Create a `nodes` attr containing dict of <node type: node ids> where node ids are nodes with nonnull sequences
        """
        for ntype in self.node_types:
            if ntype not in self.multiomics.get_omics_list(): continue

            nodes_w_seq = self.multiomics[ntype].annotations.index[
                self.multiomics[ntype].annotations[SEQUENCE_COL].notnull()]
            self.nodes[ntype] = self.nodes[ntype].intersection(nodes_w_seq)

    def to_dgl_heterograph(self, node_attr_cols: List[str] = [], target="go_id", min_count=10,
                           label_subset: Optional[Union[Index, np.ndarray]] = None,
                           ntype_subset: Optional[List[str]] = None,
                           sequence=False, expression=False) \
            -> Tuple[dgl.DGLHeteroGraph, Union[List[str], Series, np.array], Dict[str, List[str]], Any, Any, Any]:
        # Filter node that doesn't have a sequence
        if sequence:
            self.filter_sequence_nodes()
        node_types = self.node_types
        if ntype_subset:
            node_types = [ntype for ntype in node_types if ntype in ntype_subset]

        # Edge index
        edge_index_dict = {}
        edge_attr_dict = {}
        for metapath, etype_graph in self.networks.items():
            head_type, tail_type = metapath[0], metapath[-1]
            if head_type not in node_types or tail_type not in node_types: continue

            edge_index, edge_attr = get_edge_index_values(etype_graph, nodes_A=self.nodes[head_type],
                                                          nodes_B=self.nodes[tail_type],
                                                          edge_attrs=["train_mask", "valid_mask", "test_mask"],
                                                          format="dgl")

            edge_index_dict[metapath] = edge_index
            if edge_attr:
                edge_attr_dict[metapath] = edge_attr

        G: dgl.DGLHeteroGraph = dgl.heterograph(edge_index_dict, num_nodes_dict=self.num_nodes_dict)

        # Add node attributes
        for ntype in G.ntypes:
            annotations: pd.DataFrame = self.annotations[ntype].loc[self.nodes[ntype]]

            for col in annotations.columns \
                    .drop([target, "omic", SEQUENCE_COL], errors="ignore") \
                    .intersection(node_attr_cols if node_attr_cols is not None else []):
                if col in self.feature_transformer:
                    feat_filtered = filter_multilabel(annotations[col], min_count=None,
                                                      dropna=False, delimiter=self.delimiter)

                    feat = self.feature_transformer[col].transform(feat_filtered)
                    print(f"{ntype} added {col}")
                    G.nodes[ntype].data[col] = torch.from_numpy(feat)

            if expression and ntype in self.multiomics.get_omics_list() and hasattr(self.multiomics[ntype],
                                                                                    'expressions'):
                node_expression = self.multiomics[ntype].expressions.T.loc[self.nodes[ntype]].values
                G[ntype].data['expression'] = torch.tensor(node_expression, dtype=torch.float)

            # DNA/RNA sequence
            if sequence and SEQUENCE_COL in annotations:
                if hasattr(self, "tokenizer"):
                    padded_encoding, seq_lens = self.tokenizer.one_hot_encode(ntype,
                                                                              sequences=annotations[SEQUENCE_COL])
                    G[ntype].data[SEQUENCE_COL] = padded_encoding
                    print(f"Added {ntype} sequences ({padded_encoding.shape})to G[ntype].data ")
                else:
                    if not hasattr(G, 'sequences'):
                        G.sequences = {}
                    G.sequences.setdefault(ntype, {})[SEQUENCE_COL] = annotations[SEQUENCE_COL]
                    print(f"Added {ntype} sequences pd.Series to G.sequences")

        # Node labels
        self.process_feature_tranformer(filter_label=target, min_count=min_count)
        if label_subset is not None:
            self.feature_transformer[target].classes_ = np.intersect1d(self.feature_transformer[target].classes_,
                                                                       label_subset, assume_unique=True)
            classes = self.feature_transformer[target].classes_
        else:
            classes = None

        labels = {}
        for ntype in G.ntypes:
            if ntype not in self.annotations or target not in self.annotations[ntype].columns: continue
            y_label = filter_multilabel(self.annotations[ntype].loc[self.nodes[ntype].label_col],
                                        min_count=min_count,
                                        label_subset=label_subset, dropna=False, delimiter=self.delimiter)
            labels[ntype] = self.feature_transformer[target].transform(y_label)
            labels[ntype] = torch.tensor(labels[ntype])

            G.nodes[ntype].data["label"] = labels[ntype]
            num_classes = labels[ntype].shape[1]

        # Train test split
        # training_idx = {ntype: ntype_nids.get_indexer_for(ntype_nids.intersection(self.training.node_list)) \
        #                 for ntype, ntype_nids in self.nodes.to_dict().items()}
        # validation_idx = {ntype: ntype_nids.get_indexer_for(ntype_nids.intersection(self.validation.node_list)) \
        #                   for ntype, ntype_nids in self.nodes.to_dict().items()}
        # testing_idx = {ntype: ntype_nids.get_indexer_for(ntype_nids.intersection(self.testing.node_list)) \
        #                for ntype, ntype_nids in self.nodes.to_dict().items()}

        # Get index of train/valid/test edge_id
        training_idx, validation_idx, testing_idx = {}, {}, {}
        for key_mask, trainvalidtest_idx in zip(['train_mask', 'valid_mask', 'test_mask'],
                                                [training_idx, validation_idx, testing_idx]):
            for metapath, edge_attr in edge_attr_dict.items():
                src, dst = edge_index_dict[metapath]
                trainvalidtest_idx[metapath] = G.edge_ids(src[edge_attr[key_mask]], dst[edge_attr[key_mask]],
                                                          etype=metapath)

        return G, classes, dict(self.nodes), training_idx, validation_idx, testing_idx

    def to_pyg_heterodata(self, node_attr_cols: List[str] = [],
                          target="go_id", min_count=10, label_subset: Optional[Union[Index, np.ndarray]] = None,
                          ntype_subset: Optional[List[str]] = None,
                          exclude_metapaths: List[Tuple[str, str, str]] = None,
                          sequence=False, expression=False) \
            -> Tuple[Union[HeteroData, Any], Union[List[str], Series, np.array], Dict[str, List[str]], Any, Any, Any]:
        # Filter node that doesn't have a sequence
        if sequence:
            self.filter_sequence_nodes()

        hetero = HeteroData()
        node_types = self.node_types
        if ntype_subset:
            node_types = [ntype for ntype in node_types if ntype in ntype_subset]

        # Edge index
        for metapath, nxgraph in self.networks.items():
            head_type, tail_type = metapath[0], metapath[-1]
            if ntype_subset and (head_type not in ntype_subset or tail_type not in ntype_subset): continue
            if exclude_metapaths and (
                    metapath in exclude_metapaths or untag_negative_metapath(metapath) in exclude_metapaths): continue

            hetero[metapath].edge_index, edge_attrs = get_edge_index_values(nxgraph, self.nodes[head_type],
                                                                            self.nodes[tail_type],
                                                                            edge_attrs=["train_mask", "valid_mask",
                                                                                        "test_mask"])

            for edge_attr, edge_value in edge_attrs.items():
                hetero[metapath][edge_attr] = edge_value

        # Add node attributes
        for ntype in node_types:
            annotations: pd.DataFrame = self.annotations[ntype].loc[self.nodes[ntype]]

            for col in annotations.columns \
                    .drop([target, "omic", SEQUENCE_COL], errors="ignore") \
                    .intersection(node_attr_cols if node_attr_cols is not None else []):

                if col in self.feature_transformer:
                    feat_filtered = filter_multilabel(annotations[col], min_count=None,
                                                      dropna=False, delimiter=self.delimiter)
                    feat: np.ndarray = self.feature_transformer[col].transform(feat_filtered)
                    hetero[ntype][col] = torch.tensor(feat, dtype=torch.float)
                else:
                    hetero[ntype][col] = annotations[col].to_numpy()

            if expression and ntype in self.multiomics.get_omics_list() and hasattr(self.multiomics[ntype],
                                                                                    'expressions'):
                node_expression = self.multiomics[ntype].expressions.T.loc[self.nodes[ntype]].values
                hetero[ntype].expression = torch.tensor(node_expression, dtype=torch.float)

            hetero[ntype]['nid'] = torch.arange(len(self.nodes[ntype]), dtype=torch.long)

            # DNA/RNA sequence
            if sequence and SEQUENCE_COL in annotations:
                hetero[ntype][SEQUENCE_COL] = annotations[SEQUENCE_COL]  # .to_numpy()

        # Labels
        if target:
            y_label = filter_multilabel(self.all_annotations[target],
                                        min_count=min_count,
                                        label_subset=label_subset, dropna=False, delimiter=self.delimiter)
            self.feature_transformer[target].fit_transform(y_label)

            ## Filter nodes
            if label_subset is not None:
                self.feature_transformer[target].classes_ = np.intersect1d(self.feature_transformer[target].classes_,
                                                                           label_subset, assume_unique=True)
            classes = self.feature_transformer[target].classes_
            print(f"Selected {len(self.feature_transformer[target].classes_)} classes.")

            ## Node labels
            y_dict = {}
            for ntype in node_types:
                if target not in self.annotations[ntype].columns: continue
                y_label = filter_multilabel(self.annotations[ntype].loc[self.nodes[ntype], target],
                                            min_count=None, label_subset=classes, dropna=False,
                                            delimiter=self.delimiter)
                y_dict[ntype] = self.feature_transformer[target].transform(y_label)
                y_dict[ntype] = torch.tensor(y_dict[ntype])

                hetero[ntype]["y"] = y_dict[ntype]
        else:
            classes = None

        # Train test split (from previously saved node train/test split)
        if hasattr(self, "train_nodes") and self.train_nodes:
            train_idx = {ntype: nodelist.get_indexer_for(nodelist.intersection(self.train_nodes[ntype])) \
                         for ntype, nodelist in self.nodes.items() if ntype in self.train_nodes}
            valid_idx = {ntype: nodelist.get_indexer_for(nodelist.intersection(self.valid_nodes[ntype])) \
                         for ntype, nodelist in self.nodes.items() if ntype in self.valid_nodes}
            test_idx = {ntype: nodelist.get_indexer_for(nodelist.intersection(self.test_nodes[ntype])) \
                        for ntype, nodelist in self.nodes.items() if ntype in self.test_nodes}

            for ntype in node_types:
                hetero[ntype].num_nodes = len(self.nodes[ntype])

                if ntype in train_idx:
                    mask = torch.zeros(hetero[ntype].num_nodes, dtype=torch.bool)
                    mask[train_idx[ntype]] = 1
                    hetero[ntype].train_mask = mask
                else:
                    hetero[ntype].train_mask = torch.zeros(hetero[ntype].num_nodes, dtype=torch.bool)

                if ntype in valid_idx:
                    mask = torch.zeros(hetero[ntype].num_nodes, dtype=torch.bool)
                    mask[valid_idx[ntype]] = 1
                    hetero[ntype].valid_mask = mask
                else:
                    hetero[ntype].valid_mask = torch.zeros(hetero[ntype].num_nodes, dtype=torch.bool)

                if ntype in test_idx:
                    mask = torch.zeros(hetero[ntype].num_nodes, dtype=torch.bool)
                    mask[test_idx[ntype]] = 1
                    hetero[ntype].test_mask = mask
                else:
                    hetero[ntype].test_mask = torch.zeros(hetero[ntype].num_nodes, dtype=torch.bool)

        return hetero, classes, self.nodes.to_dict(), train_idx, valid_idx, test_idx
