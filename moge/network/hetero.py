from argparse import Namespace

import dgl
import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Union, List

import torch
from torch_geometric.data import HeteroData

from moge.network.base import SEQUENCE_COL
from moge.network.attributed import AttributedNetwork, MODALITY_COL, filter_multilabel
from moge.network.train_test_split import TrainTestSplit, stratify_train_test

from openomics.utils.df import concat_uniques
from openomics import MultiOmics


class HeteroNetwork(AttributedNetwork, TrainTestSplit):
    def __init__(self, multiomics: MultiOmics, node_types: list, layers: Dict[Tuple[str], nx.Graph],
                 annotations=True, ) -> None:
        """
        :param multiomics: MultiOmics object containing annotations
        :param node_types: Node types
        :param layers: A dict of edge types tuple and networkx.Graph/Digraph containing heterogeneous edges
        :param annotations: Whether to process annotation data, default True
        """
        self.multiomics: MultiOmics = multiomics
        self.node_types = node_types
        self.networks: Dict[Tuple, nx.Graph] = {}

        networks = {}
        for src_etype_dst, GraphClass in layers.items():
            networks[src_etype_dst] = GraphClass()

        super(HeteroNetwork, self).__init__(networks=networks, multiomics=multiomics,
                                            annotations=annotations)

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
        self.nodes = pd.Series(self.nodes)
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

    def add_edges(self, edgelist: List[Tuple], etype: Tuple[str], database: str, **kwargs):
        source = etype[0]
        target = etype[-1]
        self.networks[etype].add_edges_from(edgelist, source=source, target=target, database=database, etype=etype,
                                            **kwargs)
        print(len(edgelist), "edges added to self.networks[{}]".format(etype))

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

    def split_stratified(self, stratify_label: str, stratify_omic=True, n_splits=5,
                         dropna=False, seed=42, verbose=False):
        y_label = filter_multilabel(df=self.all_annotations, column=stratify_label, min_count=n_splits, dropna=dropna,
                                    delimiter=self.delimiter)
        if stratify_omic:
            omic_type_col = self.all_annotations.loc[y_label.index, MODALITY_COL].str.split("\||:")
            y_label = y_label + omic_type_col

        train_val, test = next(stratify_train_test(y_label=y_label, n_splits=n_splits, seed=seed))

        if not hasattr(self, "training"):
            self.training = Namespace()
        if not hasattr(self, "validation"):
            self.validation = Namespace()
        if not hasattr(self, "testing"):
            self.testing = Namespace()

        train, valid = next(stratify_train_test(y_label=y_label[train_val], n_splits=int(1 // 0.1), seed=seed))

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

    def to_pyg_heterodata(self, label_col="go_id", min_count=10, label_subset=None, sequence=False):
        # Filter node that doesn't have a sequence
        if sequence:
            self.filter_sequence_nodes()

        G = HeteroData()

        # Edge index
        edge_index_dict = {}
        for relation, nxgraph in self.networks.items():
            biadj = nx.bipartite.biadjacency_matrix(nxgraph,
                                                    row_order=self.nodes[relation[0]],
                                                    column_order=self.nodes[relation[-1]],
                                                    format="coo")
            G[relation].edge_index = torch.stack([torch.from_numpy(biadj.row),
                                                  torch.from_numpy(biadj.col)])

        # Add node attributes
        for ntype in self.node_types:
            annotations = self.multiomics[ntype].annotations.loc[self.nodes[ntype]]

            node_feats = []
            for col in self.all_annotations.columns.drop([label_col, "omic", SEQUENCE_COL]):
                if col in self.feature_transformer:
                    feat_filtered = filter_multilabel(df=annotations,
                                                      column=col, min_count=None,
                                                      dropna=False, delimiter=self.delimiter)

                    feat = self.feature_transformer[col].transform(feat_filtered)
                    print(col, feat.shape)
                    node_feats.append(feat)

            G.nodes[ntype].x = torch.from_numpy(np.stack(node_feats, axis=1))

        return G

    def to_dgl_heterograph(self, label_col="go_id", min_count=10, label_subset=None, sequence=False):
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

            for col in self.all_annotations.columns.drop([label_col, "omic", SEQUENCE_COL]):
                if col in self.feature_transformer:
                    feat_filtered = filter_multilabel(df=annotations,
                                                      column=col, min_count=None,
                                                      dropna=False, delimiter=self.delimiter)

                    feat = self.feature_transformer[col].transform(feat_filtered)
                    G.nodes[ntype].data[col] = torch.from_numpy(feat)

            # DNA/RNA sequence
            if sequence and "sequence" in annotations:
                assert hasattr(self, "tokenizer")
                padded_encoding, seq_lens = self.tokenizer.one_hot_encode(ntype,
                                                                          sequences=annotations[SEQUENCE_COL])
                print(f"Added sequences ({padded_encoding.shape}) to {ntype}")
                G.nodes[ntype].data[SEQUENCE_COL] = padded_encoding
                G.nodes[ntype].data["seq_len"] = seq_lens

        # Labels
        self.process_feature_tranformer(filter_label=label_col, min_count=min_count)
        if label_subset is not None:
            self.feature_transformer[label_col].classes_ = np.intersect1d(self.feature_transformer[label_col].classes_,
                                                                          label_subset, assume_unique=True)

        labels = {}
        for ntype in G.ntypes:
            if label_col not in self.multiomics[ntype].annotations.columns: continue
            y_label = filter_multilabel(df=self.multiomics[ntype].annotations.loc[self.nodes[ntype]],
                                        column=label_col, min_count=min_count,
                                        label_subset=label_subset, dropna=False, delimiter=self.delimiter)
            labels[ntype] = self.feature_transformer[label_col].transform(y_label)
            labels[ntype] = torch.tensor(labels[ntype])

            G.nodes[ntype].data["label"] = labels[ntype]
            num_classes = labels[ntype].shape[1]

        # Train test split
        training_idx = {ntype: ntype_nids.get_indexer_for(ntype_nids.intersection(self.training.node_list)) \
                        for ntype, ntype_nids in self.nodes.to_dict().items()}
        validation_idx = {ntype: ntype_nids.get_indexer_for(ntype_nids.intersection(self.validation.node_list)) \
                          for ntype, ntype_nids in self.nodes.to_dict().items()}
        testing_idx = {ntype: ntype_nids.get_indexer_for(ntype_nids.intersection(self.testing.node_list)) \
                       for ntype, ntype_nids in self.nodes.to_dict().items()}

        return G, labels, num_classes, training_idx, validation_idx, testing_idx
