import os
import pickle
import pprint
import warnings
from argparse import Namespace
from collections import defaultdict
from collections.abc import Iterable
from os.path import join
from typing import Dict, Tuple, Union, List, Any, Optional, Set

import dask.dataframe as dd
import dgl
import networkx as nx
import numpy as np
import pandas as pd
import torch
import tqdm
from logzero import logger
from pandas import Series, Index, DataFrame
from scipy.sparse import csr_matrix
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_sparse import SparseTensor

from moge.dataset.utils import get_edge_index_values, get_edge_index_dict, tag_negative_metapath, \
    untag_negative_metapath
from moge.network.attributed import AttributedNetwork
from moge.network.base import SEQUENCE_COL
from moge.network.train_test_split import TrainTestSplit
from moge.network.utils import parse_labels
from openomics import MultiOmics
from openomics.database.ontology import Ontology, GeneOntology


class HeteroNetwork(AttributedNetwork, TrainTestSplit):
    def __init__(self, multiomics: MultiOmics, node_types: List, layers: Dict[Tuple[str], nx.Graph],
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
        self.annotations: Dict[str, DataFrame] = {}

        networks = {}
        for src_etype_dst, nxgraph in layers.items():
            if callable(nxgraph):
                networks[src_etype_dst] = nxgraph()
            else:
                networks[src_etype_dst] = nxgraph

        super(HeteroNetwork, self).__init__(networks=networks, multiomics=multiomics, annotations=annotations)

    def __repr__(self):
        nodes = {ntype: nids.size for ntype, nids in self.nodes.items()}
        networks = {'.'.join(metapath): g.__str__() for metapath, g in self.networks.items()}
        annotations = {ntype: df.shape for ntype, df in self.annotations.items()}
        return "HeteroNetwork {}\n{}".format(
            self.multiomics._cohort_name,
            pprint.pformat({'nodes': nodes, 'networks': networks, 'annotations': annotations}, depth=3, width=150))

    @classmethod
    def load(cls, path):
        return
        if isinstance(path, str) and '~' in path:
            path = os.path.expanduser(path)

        self = cls(multiomics=Namespace(), node_types=None, layers=None)

        for metapath, g in self.networks.items():
            network_name = "__".join(metapath) if isinstance(metapath, Iterable) else str(metapath)
            nx.write_gml(g, join(path, f"{network_name}.gml"))

        for ntype, df in self.annotations.items():
            df.to_pickle(join(path, f'{ntype}.pickle'))

        if isinstance(self.nodes, (pd.Index, pd.Series)):
            self.nodes.to_pickle(join(path, 'nodes.pickle'))
        elif isinstance(self.nodes, dict):
            with open(join(path, 'nodes.pickle'), 'wb') as f:
                pickle.dump(self.nodes, f)

        return self

    def save(self, path: str):
        if isinstance(path, str) and '~' in path:
            path = os.path.expanduser(path)

        if not os.path.exists(path):
            os.makedirs(path)

        for metapath, g in self.networks.items():
            network_name = "__".join(metapath) if isinstance(metapath, Iterable) else str(metapath)
            nx.write_gml(g, join(path, f"{network_name}.gml"))

        for ntype, df in self.annotations.items():
            df.to_pickle(join(path, f'{ntype}.pickle'))

        if isinstance(self.nodes, (pd.Index, pd.Series)):
            self.nodes.to_pickle(join(path, 'nodes.pickle'))
        elif isinstance(self.nodes, dict):
            with open(join(path, 'nodes.pickle'), 'wb') as f:
                pickle.dump(self.nodes, f)

    def isolated_nodes(self, ntypes: Set[str] = None) -> Dict[str, Set[str]]:
        """
        Collect all node that are isolated across all `networks`

        Args:
            ntypes (): List of node types to collect

        Returns:
            isolated_nodes (Dict[str, Set[str]]):
        """
        isolated_nodes = {ntype: set(nodelist) \
                          for ntype, nodelist in self.nodes.items() \
                          if not ntypes or ntype in ntypes}

        for metapath, g in self.networks.items():
            g_ntypes = {metapath[0], metapath[-1]}.intersection(isolated_nodes.keys())
            if g_ntypes:
                isolates = nx.isolates(g)
                for ntype in g_ntypes:
                    isolated_nodes[ntype] = isolated_nodes[ntype].intersection(isolates)

        return isolated_nodes

    def get_node_degrees(self, undirected=True) -> pd.DataFrame:
        node2ntype = {node: ntype for ntype, nodes in self.nodes.items() \
                      for node in nodes}

        degree_dfs = []
        for metapath, g in tqdm.tqdm(self.networks.items()):
            degrees = g.degree()
            degrees = pd.Series({k: v for k, v in degrees if k in node2ntype},
                                name=".".join(metapath)).to_frame()
            degrees.index.name = 'node'

            degrees['ntype'] = degrees.index.map(node2ntype)
            degrees = degrees.reset_index().set_index(['ntype', 'node'])
            degree_dfs.append(degrees)

        degree_df = pd.concat(degree_dfs, axis=1)
        degree_df.columns.name = 'etype'
        degree_df = degree_df.fillna(0.0)
        return degree_df

    def process_network(self):
        self.nodes = {}
        self.node_to_modality = {}

        for metapath, network in self.networks.items():
            for node_type in [metapath[0], metapath[-1]]:
                network.add_nodes_from(self.multiomics[node_type].get_genes_list(), modality=node_type)

        for node_type in self.node_types:
            self.nodes[node_type] = self.multiomics[node_type].get_genes_list()

            for gene in self.multiomics[node_type].get_genes_list():
                self.node_to_modality[gene] = self.node_to_modality.setdefault(gene, []) + [node_type, ]
            print(node_type, " nodes:", len(self.nodes[node_type]), self.multiomics[node_type].gene_index)

        print("Total nodes:", len(self.get_all_nodes()))
        self.nodes: Dict[str, pd.Index] = pd.Series(self.nodes)
        self.node_to_modality = pd.Series(self.node_to_modality)

    def process_annotations(self):
        for modality in self.node_types:
            annotation = self.multiomics[modality].get_annotations()
            self.annotations[modality] = annotation

        logger.info("All annotation columns (union): {}".format(
            {col for _, annotations in self.annotations.items() for col in annotations.columns.tolist()}))

    def process_feature_tranformer(self, columns: List[str] = None, ntype_subset: List[str] = None,
                                   delimiter: List[str] = "\||;",
                                   labels_subset=None, min_count=0, verbose=False):
        self.delimiter = delimiter
        dfs = []
        if columns is None:
            logger.warn('process_feature_tranformer(): No `columns` argument was provided, so doing nothing.')
            return

        for ntype in self.node_types:
            if ntype_subset and ntype not in ntype_subset: continue
            df: pd.DataFrame = self.annotations[ntype].drop(columns=[SEQUENCE_COL], errors="ignore")

            if len(df.columns.intersection(columns)):
                dfs.append(df[df.columns.intersection(columns)])

        if dfs:
            all_annotations = pd.concat(dfs, join="inner")
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                self.feature_transformer = self.get_feature_transformers(all_annotations, labels_subset=labels_subset,
                                                                         min_count=min_count, delimiter=delimiter,
                                                                         verbose=verbose)

    def add_nodes(self, nodes: List[str], ntype: str, annotations: pd.DataFrame = None):
        """

        Args:
            nodes (List[str]): List of node names.
            ntype (str): The node type.
            annotations (pd.DataFrame): Updates network's annotation dict for `ntype`.
        """
        if ntype not in self.node_types:
            self.node_types.append(ntype)

        node_index = pd.Index(nodes) if not isinstance(nodes, pd.Index) else nodes
        if ntype not in self.nodes:
            self.nodes[ntype] = node_index
        else:
            self.nodes[ntype].append(node_index.difference(self.nodes[ntype]))

        if isinstance(annotations, pd.DataFrame) and not annotations.empty:
            if ntype not in self.annotations:
                self.annotations[ntype] = annotations.loc[nodes]
            elif ntype in self.annotations and set(nodes).difference(self.annotations[ntype].index):
                new_nodes = set(nodes).difference(self.annotations[ntype].index)
                self.annotations[ntype] = self.annotations[ntype].append(annotations.loc[new_nodes])

        logger.info(f"Added {len(nodes)} {ntype} nodes")

    def add_edges(self, edgelist: Union[nx.Graph, List[Tuple[str, str, Dict]]], etype: Tuple[str, str, str],
                  database: str,
                  directed=True, **kwargs):
        src_type, dst_type = etype[0], etype[-1]
        if etype not in self.networks:
            if directed:
                self.networks[etype] = nx.DiGraph()
            else:
                self.networks[etype] = nx.Graph()

        if isinstance(edgelist, Iterable):
            self.networks[etype].add_edges_from(edgelist, source=src_type, target=dst_type, database=database, **kwargs)
        elif isinstance(edgelist, nx.Graph):
            self.networks[etype].add_edges_from(edgelist.edges.data(), source=src_type, target=dst_type,
                                                database=database,
                                                **kwargs)

        src_nodes, dst_nodes = {u for u, v, *_ in edgelist}, {v for u, v, *_ in edgelist}
        src_nodes = src_nodes.intersection(self.nodes[src_type])
        dst_nodes = dst_nodes.intersection(self.nodes[dst_type])

        logger.info(f"{database} add edges {etype}: "
                    f"({len(src_nodes)}, {self.networks[etype].number_of_edges()}, {len(dst_nodes)}).")

    def add_edges_from_ontology(self, ontology: GeneOntology, nodes: Optional[List[str]] = None,
                                split_ntype: str = None, etypes: Optional[List[str]] = None,
                                reverse_edge_dir=False, d_ntype: str = "GO_term"):
        """
        Add edges between nodes within the ontology.

        Args:
            ontology (Ontology): an openomics.Ontology object that metadata and contains edges among the terms.
            nodes (List[str]): A subset of nodes from ontology to extract edges from. Default None to use all nodes in ontology.
            etypes (List[str]): a subset of edge types (keys) in the nx.MultiGraph. Default None to use all etypes.
            split_ntype (str): name of the column in `ontology.data` df that contain the node type values for each node.
                If splitting the nodes in ontology, then the function will also split the edge types.
            d_ntype (str): default name of the ontology node type if not automatically inferred from metapaths.
        """
        if nodes is None:
            nodes = ontology.node_list

        # Add ontology description as the `sequence` feature
        if "def" in ontology.data.columns:
            ontology.data[SEQUENCE_COL] = ontology.data["def"]

        # Add nodes and node data
        if split_ntype:
            nodes_split = {}
            for d_ntype in ontology.data[split_ntype].unique():
                nodes_subset = ontology.data.query(f'{split_ntype} == "{d_ntype}"').index.intersection(nodes)
                nodes_split[d_ntype] = nodes_subset
                self.add_nodes(nodes=nodes_subset, ntype=d_ntype, annotations=ontology.data.loc[nodes_subset])
        else:
            self.add_nodes(nodes=nodes, ntype=d_ntype, annotations=ontology.data)

        graph = ontology.network
        if reverse_edge_dir:
            graph = nx.reverse(graph, copy=True)
        select_etypes = list({e for u, v, e in ontology.network.edges if not etypes or e in etypes})
        assert len(select_etypes), f"`select_etypes` must not be is empty: {select_etypes}. See `etypes` args: {etypes}"

        # Separate edgelists
        if split_ntype:
            edge_index_dict = {}
            metapaths = []
            for src_type in nodes_split:
                for dst_type in nodes_split:
                    metapaths.extend([(src_type, etype, dst_type) for etype in select_etypes])

            edge_index_dict.update(get_edge_index_dict(graph, nodes=nodes_split, metapaths=metapaths,
                                                       d_ntype=None, format="nx", ))

        else:
            edge_index_dict = get_edge_index_dict(graph, nodes=nodes, metapaths=select_etypes, d_ntype=d_ntype,
                                                  format="nx", )

        # Add ontology edges
        for metapath, edgelist in edge_index_dict.items():
            if len(edgelist) < 10: continue
            self.add_edges(edgelist, etype=metapath, database=ontology.name(), directed=True)

    def add_edges_from_annotations(self, ontology: GeneOntology, filter_dst_nodes: Optional[List[str]], src_ntype: str,
                                   dst_ntype: str, train_date: str, valid_date: str, test_date: str,
                                   src_node_col="gene_name", dst_node_col="go_id", split_etype: str = None,
                                   d_etype="associated", use_neg_annotations=True):
        """

        Args:
            ontology (): an openomics.Ontology object that contains annotations data.
            filter_dst_nodes (): A dict of all nodes in the Hetero network.
            src_ntype (): The ntype name of annotation source nodes.
            dst_ntype (): The ntype name of annotation destination nodes.
            train_date (): A date format "YYYY-MM-DD" to separate all annotations before this date.
            valid_date (): A date format "YYYY-MM-DD" to separate all annotations before this date and after train_date
            test_date (): A date format "YYYY-MM-DD" to separate all annotations before this date and after valid_date
            src_node_col (str): column name in the annotation dataframe that contain the src_ntype node names.
            dst_node_col (str): column name in the annotation dataframe that contain the dst_ntype node names.
            split_etype (str): column name in the annotation which contain the etype values to split edges by. Default None, which treats all annotation edges the same `d_etype` etype.
            d_etype (str): Default etype name for annotation edges when `split_etype` is None.
        """
        groupby = [src_node_col, split_etype] if split_etype else [src_node_col]
        assert src_ntype in self.node_types and dst_ntype in self.node_types

        # Split annotations between genes and GO terms by dates
        train_ann, valid_ann, test_ann = ontology.split_annotations(
            src_node_col=src_node_col, dst_node_col=dst_node_col, groupby=groupby,
            train_date=train_date, valid_date=valid_date, test_date=test_date,
            filter_dst_nodes=filter_dst_nodes)

        if split_etype:
            nx_options = dict(edge_key=split_etype, create_using=nx.MultiGraph, edge_attr=True)
            pred_metapaths = [(src_ntype, etype, dst_ntype) \
                              for etype in train_ann.reset_index()[split_etype].unique()]
        else:
            nx_options = dict(edge_key=None, create_using=nx.Graph, edge_attr=None)
            pred_metapaths = [(src_ntype, d_etype, dst_ntype)]

        dst_node_col = ontology.data.index.name
        node_lists = []

        # Process train, validation, and test annotations
        for go_ann in [train_ann, valid_ann, test_ann]:
            is_train, is_valid, is_test = go_ann is train_ann, go_ann is valid_ann, go_ann is test_ann

            # True Positive links
            pos_annotations = go_ann[dst_node_col].dropna().explode().to_frame().reset_index()
            if isinstance(pos_annotations, dd.DataFrame):
                pos_annotations = pos_annotations.compute()
            pos_graph = nx.from_pandas_edgelist(pos_annotations, source=src_node_col, target=dst_node_col, **nx_options)
            if split_etype:
                metapaths = {(src_ntype, etype, dst_ntype) for u, v, etype in pos_graph.edges}
            else:
                metapaths = set(pred_metapaths)

            pos_edge_list_dict = get_edge_index_dict(pos_graph, nodes=self.nodes,
                                                     metapaths=metapaths.intersection(pred_metapaths), format="nx")

            for etype, edges in pos_edge_list_dict.items():
                if etype not in pred_metapaths or len(edges) == 0: continue

                edges = self.label_edge_trainvalidtest(edges, train=is_train, valid=is_valid, test=is_test)
                self.add_edges(edges, etype=etype, database=ontology.name(), directed=True, )

            # True Negative links
            if use_neg_annotations:
                neg_pred_metapaths = []

                neg_dst_node_col = f"neg_{dst_node_col}"
                neg_annotations = go_ann[neg_dst_node_col].dropna().explode().to_frame().reset_index()
                if isinstance(neg_annotations, dd.DataFrame):
                    neg_annotations = neg_annotations.compute()
                neg_graph = nx.from_pandas_edgelist(neg_annotations, source=src_node_col, target=neg_dst_node_col,
                                                    **nx_options)
                if split_etype:
                    metapaths = {(src_ntype, etype, dst_ntype) for u, v, etype in neg_graph.edges}
                else:
                    metapaths = set(pred_metapaths)
                neg_edge_list_dict = get_edge_index_dict(neg_graph, nodes=self.nodes,
                                                         metapaths=metapaths.intersection(pred_metapaths), format="nx")
                for etype, edges in neg_edge_list_dict.items():
                    if etype not in pred_metapaths or len(edges) == 0: continue
                    neg_etype = tag_negative_metapath(etype)
                    if neg_etype not in neg_pred_metapaths:
                        neg_pred_metapaths.append(neg_etype)

                    edges = self.label_edge_trainvalidtest(edges, train=is_train, valid=is_valid, test=is_test)
                    self.add_edges(edges, etype=neg_etype, database=ontology.name(), directed=True, )

            # Gather train/valid/test split of nodes
            node_lists.append({
                src_ntype: set(self.nodes[src_ntype].intersection(pos_annotations[src_node_col].unique())),
                dst_ntype: set(self.nodes[dst_ntype].intersection(pos_annotations[dst_node_col].unique())),
            })

        # Save the list of positive prediction metapaths
        if not hasattr(self, "pred_metapaths"):
            self.pred_metapaths = pred_metapaths
        else:
            self.pred_metapaths = list(set(self.pred_metapaths + pred_metapaths))
        # Save the list of negative prediction metapaths
        if not hasattr(self, 'neg_pred_metapaths'):
            self.neg_pred_metapaths = []
        if use_neg_annotations:
            self.neg_pred_metapaths = list(set(self.neg_pred_metapaths + neg_pred_metapaths))

        # Set train/valid/test mask of all nodes on hetero graph
        train_nodes, valid_nodes, test_nodes = node_lists
        self.update_traintest_nodes_set(train_nodes, valid_nodes, test_nodes)

    def get_triples(self, all_metapaths: List[Tuple[str, str, str]], positive: bool = False, negative: bool = False) \
            -> Tuple[Dict[str, Tensor], Tensor, Tensor, Tensor]:
        """

        Args:
            all_metapaths (): External metapath list where the triple's `relation` field gets its ordinal ID from.
            positive (): Whether to only retrieve positive edge types
            negative (): Whether to only retrieve negative edge types

        Returns:
            triples, training_idx, validation_idx, testing_idx
        """
        triples = defaultdict(lambda: [])
        training_idx, validation_idx, testing_idx = [], [], []

        pred_metapaths = self.pred_metapaths if positive else self.neg_pred_metapaths

        for metapath in pred_metapaths:
            head_type, tail_type = metapath[0], metapath[-1]
            if all_metapaths and metapath not in all_metapaths: continue
            metapath_idx = all_metapaths.index(metapath)

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

            nodes_w_seq = self.annotations[ntype].index[self.annotations[ntype][SEQUENCE_COL].notnull()]
            self.nodes[ntype] = self.nodes[ntype].intersection(nodes_w_seq)

    def to_dgl_heterograph(self, node_attr_cols: List[str] = [], target="go_id", min_count=10,
                           labels_subset: Optional[Union[Index, np.ndarray]] = None,
                           head_node_type: Optional[str] = None, ntype_subset: Optional[List[str]] = None,
                           exclude_ntypes: Optional[List[str]] = None,
                           exclude_etypes: Optional[List[Union[str, Tuple]]] = None,
                           sequence=False,
                           expression=False,
                           inductive=False,
                           train_test_split="edge_id", **kwargs) \
            -> Tuple[dgl.DGLHeteroGraph, Union[List[str], Series, np.array], Dict[str, List[str]], Any, Any, Any]:
        if node_attr_cols is None:
            node_attr_cols = []

        if target:
            transform_cols = node_attr_cols + [target]
        else:
            transform_cols = node_attr_cols
        if hasattr(self, 'feature_transformer') and self.feature_transformer:
            # Avoid running multiple times for columns that already have feature_transformer
            transform_cols = [col for col in transform_cols if col not in self.feature_transformer]
        self.process_feature_tranformer(columns=transform_cols, ntype_subset=head_node_type, min_count=min_count)

        # Filter node that doesn't have a sequence
        if sequence:
            self.filter_sequence_nodes()

        # Edge index
        edge_index_dict = {}
        edge_attr_dict = {}
        for metapath, etype_graph in tqdm.tqdm(self.networks.items(), desc="Add etype's to DGLHeteroGraph"):
            head_type, etype, tail_type = metapath[0], metapath[1], metapath[-1]
            if ntype_subset and (head_type not in ntype_subset or tail_type not in ntype_subset): continue
            if exclude_ntypes and (head_type in exclude_ntypes or tail_type in exclude_ntypes): continue
            if exclude_etypes and (metapath in exclude_etypes or etype in exclude_etypes or
                                   untag_negative_metapath(metapath) in exclude_etypes): continue

            edge_index, edge_attr = get_edge_index_values(
                etype_graph, nodes_A=self.nodes[head_type], nodes_B=self.nodes[tail_type],
                # edge_attrs=["train_mask", "valid_mask", "test_mask"] \
                #     if "edge" in train_test_split or inductive else None,
                format="dgl")
            edge_index_dict[metapath] = edge_index

            if "edge" in train_test_split:
                if len(edge_attr) == 3:
                    edge_attr_dict[metapath] = edge_attr
                else:
                    edge_attr_dict[metapath] = {
                        key: torch.ones_like(edge_index[0], dtype=torch.bool) \
                        for key in {"train_mask", "valid_mask", "test_mask"}.difference(edge_attr.keys())}

        num_nodes_dict = {ntype: n for ntype, n in self.num_nodes_dict.items() \
                          if ntype_subset is None or ntype in ntype_subset}
        if exclude_ntypes:
            num_nodes_dict = {ntype: n for ntype, n in num_nodes_dict.items() if ntype not in exclude_ntypes}
        G: dgl.DGLHeteroGraph = dgl.heterograph(edge_index_dict, num_nodes_dict=num_nodes_dict)

        # Edge attributes
        for metapath, edge_attr in edge_attr_dict.items():
            for key, values in edge_attr.items():
                G.edges[metapath].data[key] = values

        # Add node attributes
        for ntype in tqdm.tqdm(G.ntypes, desc='Adding node attrs to node types'):
            annotations: pd.DataFrame = self.annotations[ntype].loc[self.nodes[ntype]]

            for col in annotations.columns \
                    .drop([target, "omic", SEQUENCE_COL], errors="ignore") \
                    .intersection(node_attr_cols if node_attr_cols is not None else []):
                if col in self.feature_transformer:
                    feat_filtered = parse_labels(annotations[col], min_count=None,
                                                 dropna=False, delimiter=self.delimiter)

                    feat = self.feature_transformer[col].transform(feat_filtered)
                    print(f"{ntype} added {col}")
                    G.nodes[ntype].data[col] = torch.from_numpy(feat)

            # Expression values
            if expression and ntype in self.multiomics.get_omics_list() and hasattr(self.multiomics[ntype],
                                                                                    'expressions'):
                expressions = self.multiomics[ntype].expressions.T.loc[self.nodes[ntype]]
                if not expressions.empty:
                    G.nodes[ntype].data['feat'] = torch.tensor(expressions.values, dtype=torch.float)

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

            # Get train_mask, valid_mask, test_mask for each node of ntype
            if hasattr(self, "train_nodes") and self.train_nodes:
                for mask_name, trainvalidtest_nodes in zip(["train_mask", "valid_mask", "test_mask"],
                                                           [self.train_nodes[ntype], self.valid_nodes[ntype],
                                                            self.test_nodes[ntype]]):
                    if trainvalidtest_nodes is None or len(trainvalidtest_nodes) == 0: continue
                    nodelist_idx = self.nodes[ntype].get_indexer_for(
                        self.nodes[ntype].intersection(trainvalidtest_nodes))
                    mask = torch.zeros(G.num_nodes(ntype), dtype=torch.bool)
                    mask[nodelist_idx] = 1
                    G.nodes[ntype].data[mask_name] = mask

        # Node labels
        if target is not None:
            if labels_subset is not None:
                self.feature_transformer[target].classes_ = np.intersect1d(self.feature_transformer[target].classes_,
                                                                           labels_subset, assume_unique=True)
            classes = self.feature_transformer[target].classes_

            labels = {}
            for ntype in G.ntypes:
                if ntype not in self.annotations or target not in self.annotations[ntype].columns: continue
                y_label = parse_labels(self.annotations[ntype].loc[self.nodes[ntype], target],
                                       min_count=None, labels_subset=None,
                                       dropna=False, delimiter=self.delimiter)
                labels[ntype] = self.feature_transformer[target].transform(y_label)
                # TODO handle sparse labels
                labels[ntype] = torch.tensor(labels[ntype])

                G.nodes[ntype].data["label"] = labels[ntype]
        else:
            classes = None

        # Get index of train/valid/test edge_id
        training, validation, testing = {}, {}, {}

        if train_test_split == "edge_id":
            for mask_name, trainvalidtest in zip(['train_mask', 'valid_mask', 'test_mask'],
                                                 [training, validation, testing]):
                for metapath, edge_attr in edge_attr_dict.items():
                    src, dst = edge_index_dict[metapath]
                    trainvalidtest[metapath] = G.edge_ids(src[edge_attr[mask_name]], dst[edge_attr[mask_name]],
                                                          etype=metapath)
        elif "node" in train_test_split:
            for mask_name, trainvalidtest in zip(['train_mask', 'valid_mask', 'test_mask'],
                                                 [training, validation, testing]):
                for ntype in G.ntypes:
                    if mask_name not in G.nodes[ntype].data: continue

                    if train_test_split == "node_mask":
                        trainvalidtest[mask_name] = G.nodes[ntype].data[mask_name]
                    elif train_test_split == "node_id":
                        trainvalidtest[ntype] = torch.arange(G.num_nodes(ntype))[G.nodes[ntype].data[mask_name]]
                    else:
                        raise ValueError(f"train_test_split {train_test_split}")
        else:
            raise Exception(
                f"Invalid `train_test_split` argument {train_test_split}. Must be one of [edge_id, node_id, node_mask]")

        return G, classes, dict(self.nodes), training, validation, testing

    def to_pyg_heterodata(self, node_attr_cols: List[str] = [], target: str = "go_id", min_count=10,
                          labels_subset: Optional[Union[Index, np.ndarray]] = None, head_node_type: str = None,
                          ntype_subset: Optional[List[str]] = None, exclude_ntypes: Optional[List[str]] = None,
                          exclude_etypes: List[Union[str, Tuple]] = None, sequence=False, expression=False,
                          inductive=False, train_test_split="node_mask", **kwargs) \
            -> Tuple[Union[HeteroData, Any], Union[List[str], Series, np.array], Dict[str, List[str]], Any, Any, Any]:
        """

        Args:
            node_attr_cols ():
            target ():
            min_count ():
            labels_subset ():
            head_node_type ():
            ntype_subset ():
            exclude_ntypes ():
            exclude_etypes ():
            sequence ():
            expression ():
            inductive ():
            train_test_split ():
            **kwargs ():

        Returns:

        """
        if node_attr_cols is None:
            node_attr_cols = []

        # Filter node that doesn't have a sequence
        if sequence:
            self.filter_sequence_nodes()

        # Build feature binarizer for attr columns and target classes
        if target:
            transform_cols = node_attr_cols + [target]
        else:
            transform_cols = node_attr_cols

        if hasattr(self, 'feature_transformer') and self.feature_transformer:
            # Avoid running multiple times for columns that already have feature_transformer
            transform_cols = [col for col in transform_cols if col not in self.feature_transformer]
        self.process_feature_tranformer(columns=transform_cols, ntype_subset=head_node_type, min_count=min_count)

        node_types = self.node_types
        if ntype_subset:
            node_types = [ntype for ntype in node_types if ntype in ntype_subset]
        if exclude_ntypes:
            node_types = [ntype for ntype in node_types if ntype not in exclude_ntypes]

        # Edge index
        hetero = HeteroData()
        networks_prog = tqdm.tqdm(self.networks, desc="Adding edge_index's to HeteroData")
        for metapath in networks_prog:
            networks_prog.set_description(f"Adding edge_index's to HeteroData: {metapath}")
            head_type, etype, tail_type = metapath[0], metapath[1], metapath[-1]
            if ntype_subset and (head_type not in ntype_subset or tail_type not in ntype_subset): continue
            if exclude_ntypes and (head_type in exclude_ntypes or tail_type in exclude_ntypes): continue
            if exclude_etypes and (metapath in exclude_etypes or etype in exclude_etypes or
                                   untag_negative_metapath(metapath) in exclude_etypes): continue

            hetero[metapath].edge_index, edge_attrs = get_edge_index_values(
                self.networks[metapath],
                nodes_A=self.nodes[head_type], nodes_B=self.nodes[tail_type],
                edge_attrs=["train_mask", "valid_mask", "test_mask"] \
                    if 'edge' in train_test_split or inductive else None,
                format='pyg',
            )

            for edge_attr, edge_value in edge_attrs.items():
                hetero[metapath][edge_attr] = edge_value

        # Add node attributes
        for ntype in tqdm.tqdm(node_types, desc="Adding node attrs to node types"):
            nodelist = self.nodes[ntype]
            hetero[ntype]['nid'] = torch.arange(len(nodelist), dtype=torch.long)
            annotations: pd.DataFrame = self.annotations[ntype].loc[nodelist]

            for col in annotations.columns \
                    .drop([target, "omic", SEQUENCE_COL], errors="ignore") \
                    .intersection(node_attr_cols if node_attr_cols is not None else []):

                if col in self.feature_transformer:
                    feat_filtered = parse_labels(annotations[col], min_count=None, dropna=False,
                                                 delimiter=self.delimiter)
                    feat: np.ndarray = self.feature_transformer[col].transform(feat_filtered)
                    hetero[ntype][col] = torch.tensor(feat, dtype=torch.float)
                else:
                    hetero[ntype][col] = annotations[col].to_numpy()

            # DNA/RNA sequence
            if sequence and SEQUENCE_COL in annotations:
                hetero[ntype][SEQUENCE_COL] = annotations[SEQUENCE_COL]  # .to_numpy()

            if expression and ntype in self.multiomics.get_omics_list() \
                    and hasattr(self.multiomics[ntype], 'expressions'):
                expressions = self.multiomics[ntype].expressions.loc[:, nodelist]  # (num_features, num_samples)

                if hasattr(expressions, 'sparse') and not expressions.empty:
                    csr_mtx: csr_matrix = expressions.sparse.to_coo().T.tocsr()

                    hetero[ntype]['x'] = SparseTensor(rowptr=torch.tensor(csr_mtx.indptr, dtype=torch.long),
                                                      col=torch.tensor(csr_mtx.indices, dtype=torch.long),
                                                      value=torch.tensor(csr_mtx.data, dtype=torch.float),
                                                      sparse_sizes=csr_mtx.shape, is_sorted=True, trust_data=True)
                    logger.info(f'{ntype}, "expressions sparse", {csr_mtx.shape}')

                elif not expressions.empty:
                    hetero[ntype]['x'] = torch.tensor(expressions.values.T, dtype=torch.float)
                    logger.info(f"{ntype}, 'expressions', {hetero[ntype]['x'].shape}")

        # Node labels
        if target:
            if labels_subset is not None:
                self.feature_transformer[target].classes_ = np.intersect1d(
                    self.feature_transformer[target].classes_, labels_subset, assume_unique=True)

            classes = self.feature_transformer[target].classes_

            if len(classes) > 1000:
                self.feature_transformer[target].sparse_output = True

            # Transform labels matrix
            for ntype in hetero.node_types:
                if ntype not in self.annotations or target not in self.annotations[ntype].columns: continue
                y_label = parse_labels(self.annotations[ntype].loc[self.nodes[ntype], target],
                                       min_count=None, labels_subset=None,
                                       dropna=False, delimiter=self.delimiter)

                labels = self.feature_transformer[target].transform(y_label)
                if isinstance(labels, np.ndarray):
                    labels = torch.tensor(labels)

                elif isinstance(labels, csr_matrix):
                    labels = SparseTensor(rowptr=torch.tensor(labels.indptr, dtype=torch.long),
                                          col=torch.tensor(labels.indices, dtype=torch.long),
                                          value=torch.tensor(labels.data, dtype=torch.float),
                                          sparse_sizes=labels.shape, is_sorted=True, trust_data=True)

                hetero[ntype]['y'] = labels
        else:
            classes = None

        # Train test split (from previously saved node train/test split)
        if hasattr(self, "train_nodes") and self.train_nodes:
            train_nodes_idx = {ntype: nodelist.get_indexer_for(nodelist.intersection(self.train_nodes[ntype])) \
                               for ntype, nodelist in self.nodes.items() if self.train_nodes[ntype]}
            valid_nodes_idx = {ntype: nodelist.get_indexer_for(nodelist.intersection(self.valid_nodes[ntype])) \
                               for ntype, nodelist in self.nodes.items() if self.valid_nodes[ntype]}
            test_nodes_idx = {ntype: nodelist.get_indexer_for(nodelist.intersection(self.test_nodes[ntype])) \
                              for ntype, nodelist in self.nodes.items() if self.test_nodes[ntype]}

            for ntype in node_types:
                hetero[ntype].num_nodes = len(self.nodes[ntype])

                if ntype in train_nodes_idx:
                    mask = torch.zeros(hetero[ntype].num_nodes, dtype=torch.bool)
                    mask[train_nodes_idx[ntype]] = 1
                    hetero[ntype].train_mask = mask
                else:
                    hetero[ntype].train_mask = torch.zeros(hetero[ntype].num_nodes, dtype=torch.bool)

                if ntype in valid_nodes_idx:
                    mask = torch.zeros(hetero[ntype].num_nodes, dtype=torch.bool)
                    mask[valid_nodes_idx[ntype]] = 1
                    hetero[ntype].valid_mask = mask
                else:
                    hetero[ntype].valid_mask = torch.zeros(hetero[ntype].num_nodes, dtype=torch.bool)

                if ntype in test_nodes_idx:
                    mask = torch.zeros(hetero[ntype].num_nodes, dtype=torch.bool)
                    mask[test_nodes_idx[ntype]] = 1
                    hetero[ntype].test_mask = mask
                else:
                    hetero[ntype].test_mask = torch.zeros(hetero[ntype].num_nodes, dtype=torch.bool)
        else:
            train_nodes_idx = valid_nodes_idx = test_nodes_idx = None

        # for ntype, valid_nids in valid_nodes_idx.items():

        return hetero, classes, self.nodes[node_types], train_nodes_idx, valid_nodes_idx, test_nodes_idx
