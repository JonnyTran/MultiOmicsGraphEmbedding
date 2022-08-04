from argparse import Namespace
from typing import Dict, Tuple, Union, List, Any, Optional

import dgl
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch_geometric.transforms as T
from openomics import MultiOmics
from openomics.database.ontology import Ontology
from openomics.utils.df import concat_uniques
from pandas import Series, Index
from torch_geometric.data import HeteroData

from moge.dataset.utils import get_edge_index, get_edge_index_dict
from moge.network.attributed import AttributedNetwork, MODALITY_COL
from moge.network.base import SEQUENCE_COL
from moge.network.train_test_split import TrainTestSplit, stratify_train_test
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

    def add_edges(self, edgelist: List[Tuple[str, str]], etype: Tuple[str, str, str], database: str, directed=True,
                  **kwargs):
        src_type, dst_type = etype[0], etype[-1]
        if etype not in self.networks:
            if directed:
                self.networks[etype] = nx.DiGraph()
            else:
                self.networks[etype] = nx.Graph()

        self.networks[etype].add_edges_from(edgelist, source=src_type, target=dst_type, database=database, etype=etype,
                                            **kwargs)
        unq_sources = {u for u, v, *_ in edgelist}
        unq_targets = {v for u, v, *_ in edgelist}

        print(
            f"{etype}: {len(edgelist)} edges added between {len(unq_sources)} {src_type} and {len(unq_targets)} {dst_type}")

    def add_ontology_edges(self, ontology: Ontology, nodes: Optional[List[str]] = None, ntype: str = "GO_term"):
        """

        Args:
            ontology ():
            nodes (List[str]): A subset of nodes from ontology to extract edges from
            ntype (str): Name of the ontology node type
        """
        if nodes is None:
            nodes = ontology.node_list

        # Add ontology nodes
        self.add_nodes(nodes=nodes, ntype=ntype, annotations=ontology.data)

        # Add ontology edges
        edge_types = {e for u, v, e in ontology.network.edges}
        edge_index_dict = get_edge_index_dict(ontology.network, nodes=nodes, metapaths=edge_types,
                                              format="nx", d_ntype=ntype)
        for etype, edgelist in edge_index_dict.items():
            if len(edgelist) < 10: continue
            self.add_edges(edgelist, etype=etype, database=ontology.name(), directed=True)

    def add_ontology_annotations(self, ontology: Ontology, nodes: Optional[List[str]], head_node_type: str,
                                 train_date: str, valid_date: str, test_date: str, split_etype: str = None,
                                 ntype: str = "GO_term"):

        groupby = ["gene_name", split_etype] if split_etype else ["gene_name"]
        assert head_node_type in self.node_types

        # Add ontology annotations to existing ntypes
        train_go_ann, valid_go_ann, test_go_ann = ontology.annotation_train_val_test_split(
            train_date=train_date, valid_date=valid_date, test_date=test_date,
            groupby=groupby, filter_go_id=nodes)
        annotations = pd.concat([train_go_ann, valid_go_ann, test_go_ann], axis=0)

        if split_etype:
            nx_options = dict(edge_key=split_etype, create_using=nx.MultiGraph, edge_attr=True)
            self.pred_metapaths = [(head_node_type, etype, ntype) \
                                   for etype in train_go_ann.reset_index()[split_etype].unique()]
        else:
            nx_options = dict(edge_key=None, create_using=nx.Graph, edge_attr=None)
            self.pred_metapaths = [(head_node_type, "associated", ntype)]

        assert annotations.index.name is not None
        index_col = ontology.data.index.name

        # Process train, validation, and test annotations
        for go_ann in [train_go_ann, valid_go_ann, test_go_ann]:
            # True Positive links (undirected)
            nx_graph = nx.from_pandas_edgelist(go_ann[index_col].explode().to_frame().reset_index().dropna(),
                                               source="gene_name", target=index_col, **nx_options)
            if split_etype:
                metapaths = {(self.head_node_type, e, self.go_ntype) for u, v, e in nx_graph.edges}
            else:
                metapaths = set(self.pred_metapaths)

            edge_list_dict = get_edge_index_dict(nx_graph, nodes=self.nodes,
                                                 metapaths=metapaths.intersection(self.pred_metapaths), format="nx")
            for etype, edgelist in edge_list_dict.items():
                if len(edgelist) == 0: continue
                train_mask = np.ones(len(edgelist), dtype=bool) if go_ann is train_go_ann else np.zeros(len(edgelist),
                                                                                                        dtype=bool)
                valid_mask = np.ones(len(edgelist), dtype=bool) if go_ann is valid_go_ann else np.zeros(len(edgelist),
                                                                                                        dtype=bool)
                test_mask = np.ones(len(edgelist), dtype=bool) if go_ann is test_go_ann else np.zeros(len(edgelist),
                                                                                                      dtype=bool)
                edge_attr = [{'train_mask': train, 'valid_mask': valid, 'test_mask': test} \
                             for train, valid, test in zip(train_mask, valid_mask, test_mask)]

                edgelist = [(u, v, d) for (u, v), d in zip(edgelist, edge_attr)]
                self.add_edges(edgelist, etype=etype, database=ontology.name(), directed=True, )

    @property
    def num_nodes_dict(self):
        return {node: nid.shape[0] for node, nid in self.nodes.items()}

    def get_adjacency_matrix(self, edge_types: Union[Tuple[str], List[Tuple[str]]],
                             node_list=None, method="GAT", output="dense"):
        """

        :param edge_types: either a tuple(str, ...) or [tuple(str, ...), tuple(str, ...)]
        :param node_list (list):
        :return:
        """
        if node_list is None:
            node_list = self.node_list

        if isinstance(edge_types, tuple):
            assert edge_types in self.networks
            adj = self.get_layer_adjacency_matrix(edge_types, node_list, method=method, output=output)

        elif isinstance(edge_types, list) and isinstance(edge_types[0], tuple):
            assert self.networks.issuperset(edge_types)
            adj = {}
            for layer in edge_types:
                adj[layer] = self.get_layer_adjacency_matrix(layer, node_list)
        else:
            raise Exception("edge_types '{}' must be one of {}".format(edge_types, self.node_types))

        return adj

    def get_layer_adjacency_matrix(self, edge_type, node_list=None, method="GAT", output="csr"):
        if edge_type in self.layers_adj:
            adjacency_matrix = self.layers_adj[edge_type]

        # Get adjacency and caches the matrix
        else:
            adjacency_matrix = nx.adjacency_matrix(self.networks[edge_type],
                                                   nodelist=self.node_list)
            # if method == "GAT":
            #     adjacency_matrix = adjacency_matrix + sps.csr_matrix(
            #         np.eye(adjacency_matrix.shape[0]))  # Add self-loops

            self.layers_adj[edge_type] = adjacency_matrix

        if node_list is None or node_list == self.node_list:
            pass
        elif set(node_list) <= set(self.node_list):
            adjacency_matrix = self.slice_adj(adjacency_matrix, node_list, None)
        elif set(node_list) > set(self.node_list):
            raise Exception(f"A node in node_l is not in self.node_list : {set(node_list) - set(self.node_list)}")

        if output == "csr":
            return adjacency_matrix.astype(float)
        elif output == "coo":
            adjacency_matrix = adjacency_matrix.tocoo(copy=True)
            return np.vstack((adjacency_matrix.row, adjacency_matrix.col)).astype("long")
        elif output == "dense":
            return adjacency_matrix.todense()
        else:
            raise Exception("Output must be one of {csr, coo, dense}")

    def split_stratified(self, stratify_label: str, stratify_omic=True, test_size=0.2, min_count=100, max_count=2000,
                         dropna=False, seed=42, verbose=False):
        y_label = filter_multilabel(self.all_annotations[stratify_label], min_count=min_count,
                                    max_count=max_count,
                                    dropna=dropna, delimiter=self.delimiter)
        if stratify_omic:
            omic_type_col = self.all_annotations.loc[y_label.index, MODALITY_COL].str.split("\||:")
            y_label = y_label + omic_type_col

        train_val, test = next(stratify_train_test(y_label=y_label, test_size=test_size, seed=seed))

        if not hasattr(self, "training"):
            self.training = Namespace()
        if not hasattr(self, "validation"):
            self.validation = Namespace()
        if not hasattr(self, "testing"):
            self.testing = Namespace()

        train, valid = next(stratify_train_test(y_label=y_label[train_val], test_size=0.10, seed=seed))

        self.training.node_list = train
        self.validation.node_list = valid
        self.testing.node_list = test

    def get_aggregated_network(self):
        G = nx.compose_all(list(self.networks.values()))
        return G

    def filter_sequence_nodes(self):
        """
        Create a `nodes` attr containing dict of <node type: node ids> where node ids are nodes with nonnull sequences
        """
        for ntype in self.node_types:
            nodes_w_seq = self.multiomics[ntype].annotations.index[
                self.multiomics[ntype].annotations[SEQUENCE_COL].notnull()]
            self.nodes[ntype] = self.nodes[ntype].intersection(nodes_w_seq)

    def merge_annotations(self, node_manifest: pd.DataFrame, ):
        assert "index" in node_manifest.columns and "ntype" in node_manifest.columns

        node_types = node_manifest["ntype"].unique()

        for ntype in node_types:
            if ntype not in self.annotations.keys():
                print(f"No {ntype} in annotations")
                continue

            node_mask = node_manifest["ntype"] == ntype
            annotations = self.annotations[ntype].loc[node_manifest.loc[node_mask, "index"]]
            print(annotations)
            node_manifest.loc[node_mask] = annotations

        return node_manifest

    def to_dgl_heterograph(self, target="go_id", min_count=10, label_subset=None, sequence=False, **kwargs) \
            -> Tuple[Union[HeteroData, Any], Union[List[str], Series, np.array], Dict[str, List[str]]]:
        # Filter node that doesn't have a sequence
        if sequence:
            self.filter_sequence_nodes()

        # Edge index
        edge_index_dict = {}
        for relation, nxgraph in self.networks.items():
            biadj = nx.bipartite.biadjacency_matrix(nxgraph,
                                                    row_order=self.nodes[relation[0]],
                                                    column_order=self.nodes[relation[-1]],
                                                    format="coo")
            edge_index_dict[relation] = (biadj.row, biadj.col)

        G: dgl.DGLHeteroGraph = dgl.heterograph(edge_index_dict, num_nodes_dict=self.num_nodes_dict)

        # Add node attributes
        for ntype in G.ntypes:
            annotations = self.multiomics[ntype].annotations.loc[self.nodes[ntype]]

            for col in self.all_annotations.columns.drop([target, "omic", SEQUENCE_COL], errors="ignore"):
                if col in self.feature_transformer:
                    feat_filtered = filter_multilabel(annotations[col], min_count=None,
                                                      dropna=False, delimiter=self.delimiter)

                    feat = self.feature_transformer[col].transform(feat_filtered)
                    G.nodes[ntype].data[col] = torch.from_numpy(feat)

            # DNA/RNA sequence
            if sequence and SEQUENCE_COL in annotations:
                if hasattr(self, "tokenizer"):
                    padded_encoding, seq_lens = self.tokenizer.one_hot_encode(ntype,
                                                                              sequences=annotations[SEQUENCE_COL])
                    print(f"Added sequences ({padded_encoding.shape}) to {ntype}")
                # G.nodes[ntype].data["seq_len"] = seq_lens
                # G.nodes[ntype].data[SEQUENCE_COL] = annotations[SEQUENCE_COL].to_numpy()

        # Labels
        self.process_feature_tranformer(filter_label=target, min_count=min_count)
        if label_subset is not None:
            self.feature_transformer[target].classes_ = np.intersect1d(self.feature_transformer[target].classes_,
                                                                       label_subset, assume_unique=True)

        labels = {}
        for ntype in G.ntypes:
            if target not in self.multiomics[ntype].annotations.columns: continue
            y_label = filter_multilabel(df=self.multiomics[ntype].annotations.loc[self.nodes[ntype].label_col],
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

        return G, labels, self.nodes.to_dict()

    def to_pyg_heterodata(self, target="go_id", min_count=10,
                          label_subset: Optional[Union[Index, np.ndarray]] = None,
                          ntype_subset: Optional[List[str]] = None,
                          sequence=False, attr_cols: List[str] = [],
                          expression=True, add_reverse=True) \
            -> Tuple[Union[HeteroData, Any], Union[List[str], Series, np.array], Dict[str, List[str]]]:
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

            hetero[metapath].edge_index = get_edge_index(nxgraph, self.nodes[head_type], self.nodes[tail_type])

        # Add node attributes
        node_attr_cols = self.all_annotations.columns.drop([target, "omic", SEQUENCE_COL], errors="ignore")
        node_attr_cols = node_attr_cols.intersection(attr_cols if attr_cols is not None else [])

        for ntype in node_types:
            annotations = self.multiomics[ntype].annotations.loc[self.nodes[ntype]]

            node_feats = []
            for col in node_attr_cols:
                if col in self.feature_transformer:
                    feat_filtered = filter_multilabel(annotations[col], min_count=None,
                                                      dropna=False, delimiter=self.delimiter)
                    feat: np.ndarray = self.feature_transformer[col].transform(feat_filtered)
                    # data[ntype][col] = feat
                    print(ntype, col)
                    node_feats.append(torch.tensor(feat, dtype=torch.float))

            if expression:
                node_expression = self.multiomics[ntype].expressions.T.loc[self.nodes[ntype]].values
                node_feats.append(torch.tensor(node_expression, dtype=torch.float))

            if node_feats:
                hetero[ntype].x = torch.cat(node_feats, dim=1)
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
                if target not in self.multiomics[ntype].annotations.columns: continue
                y_label = filter_multilabel(self.multiomics[ntype].annotations.loc[self.nodes[ntype], target],
                                            min_count=None, label_subset=classes, dropna=False,
                                            delimiter=self.delimiter)
                y_dict[ntype] = self.feature_transformer[target].transform(y_label)
                y_dict[ntype] = torch.tensor(y_dict[ntype])

                hetero[ntype]["y"] = y_dict[ntype]
        else:
            classes = None

        # Add reverse metapaths to allow reverse message passing for directed edges
        if add_reverse:
            transform = T.ToUndirected(merge=True)
            hetero = transform(hetero)

        # Train test split (from previously saved node train/test split)
        train_idx = {ntype: ntype_nids.get_indexer_for(ntype_nids.intersection(self.training.node_list)) \
                     for ntype, ntype_nids in self.nodes.items()}
        valid_idx = {ntype: ntype_nids.get_indexer_for(ntype_nids.intersection(self.validation.node_list)) \
                     for ntype, ntype_nids in self.nodes.items()}
        test_idx = {ntype: ntype_nids.get_indexer_for(ntype_nids.intersection(self.testing.node_list)) \
                    for ntype, ntype_nids in self.nodes.items()}

        for ntype in node_types:
            hetero[ntype].num_nodes = len(self.nodes[ntype])

            if ntype in train_idx:
                mask = torch.zeros(hetero[ntype].num_nodes, dtype=torch.bool)
                mask[train_idx[ntype]] = 1
                hetero[ntype].train_mask = mask

            if ntype in valid_idx:
                mask = torch.zeros(hetero[ntype].num_nodes, dtype=torch.bool)
                mask[valid_idx[ntype]] = 1
                hetero[ntype].valid_mask = mask

            if ntype in test_idx:
                mask = torch.zeros(hetero[ntype].num_nodes, dtype=torch.bool)
                mask[test_idx[ntype]] = 1
                hetero[ntype].test_mask = mask

        return hetero, classes, self.nodes.to_dict()
