from pprint import pprint
from typing import List, Tuple, Union, Dict, Optional

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pandas import DataFrame, Series
from torch import Tensor
from torch.utils.data import DataLoader
from torch_geometric.data import HeteroData
from torch_sparse.tensor import SparseTensor
from umap import UMAP

from moge.dataset.PyG.neighbor_sampler import NeighborLoader, HGTLoader
from moge.dataset.PyG.triplet_generator import TripletDataset
from moge.dataset.graph import HeteroGraphDataset
from moge.dataset.sequences import SequenceTokenizers
from moge.dataset.utils import get_edge_index, edge_index_to_adjs, to_scipy_adjacency, gather_node_dict
from moge.model.PyG.utils import num_edges, convert_to_nx_edgelist, is_negative
from moge.network.hetero import HeteroNetwork


def reverse_metapath_name(metapath: Tuple[str, str, str]) -> Tuple[str, str, str]:
    rev_metapath = tuple(reversed(["rev_" + type if i % 2 == 1 else type \
                                   for i, type in enumerate(metapath)]))
    return rev_metapath


class HeteroNodeClfDataset(HeteroGraphDataset):
    def __init__(self, dataset: HeteroData,
                 seq_tokenizer: SequenceTokenizers = None,
                 neighbor_loader: str = "NeighborLoader",
                 neighbor_sizes: Union[List[int], Dict[str, List[int]]] = [128, 128],
                 node_types: List[str] = None, metapaths: List[Tuple[str, str, str]] = None, head_node_type: str = None,
                 edge_dir: str = "in", reshuffle_train: float = None, add_reverse_metapaths: bool = False,
                 inductive: bool = False, **kwargs):
        super().__init__(dataset, node_types, metapaths, head_node_type, edge_dir, reshuffle_train,
                         add_reverse_metapaths, inductive, **kwargs)
        if seq_tokenizer:
            self.seq_tokenizer = seq_tokenizer

        self.neighbor_loader = neighbor_loader
        self.neighbor_sizes = neighbor_sizes

    @classmethod
    def from_pyg_heterodata(cls, network: HeteroNetwork, target: str = None, min_count: int = None,
                            attr_cols: List[str] = None,
                            expression=False, sequence=False, add_reverse=True, label_subset=None, **kwargs):
        hetero, classes, nodes = \
            network.to_pyg_heterodata(target=target, min_count=min_count,
                                      attr_cols=attr_cols, expression=expression, sequence=sequence,
                                      add_reverse=add_reverse, label_subset=label_subset)

        self = cls(dataset=hetero, metapaths=hetero.edge_types, add_reverse_metapaths=False, edge_dir="in", **kwargs)
        self.network = network
        self.classes = classes
        self.nodes = nodes
        self._name = ""
        self.use_reverse = True if any("rev_" in metapath[1] for metapath in hetero.edge_types) else False

        return self

    def process_pyg_heterodata(self, hetero: HeteroData):
        self.G = hetero
        self.x_dict = hetero.x_dict
        self.node_types = hetero.node_types
        self.num_nodes_dict = {ntype: hetero[ntype].num_nodes for ntype in hetero.node_types}
        self.y_dict = {ntype: hetero[ntype].y for ntype in hetero.node_types if hasattr(hetero[ntype], "y")}

        self.metapaths = hetero.edge_types
        self.edge_index_dict = {etype: etype_dict["edge_index"] \
                                for etype, etype_dict in zip(hetero.edge_types, hetero.edge_stores)}

    def add_ontology_edges(self, ontology, train_date='2017-06-15', valid_date='2017-11-15', test_date='2021-12-31',
                           go_ntype="GO_term", **kwargs):
        all_go = set(ontology.network.nodes).intersection(ontology.data.index)
        self.go_ntype = go_ntype

        # Order nodes with classes nodes first
        if self.classes is not None:
            go_nodes = np.concatenate([self.classes, np.array(list(set(all_go) - set(self.classes)))])
        else:
            go_nodes = np.array(list(all_go))

        # Edges between GO terms
        edge_types = {e for u, v, e in ontology.network.edges}
        edge_index_dict = to_scipy_adjacency(ontology.network, nodes=go_nodes, edge_types=edge_types,
                                             reverse=not self.use_reverse,
                                             format="pyg", d_ntype=go_ntype)
        for metapath, edge_index in edge_index_dict.items():
            if edge_index.size(1) < 200: continue
            self.G[metapath].edge_index = edge_index
            self.edge_index_dict[metapath] = edge_index
            self.metapaths.append(metapath)

        # Cls node attrs
        for attr, values in ontology.data.loc[go_nodes][["name", "namespace", "def"]].iteritems():
            self.G[go_ntype][attr] = values.to_numpy()

        self.G[go_ntype]['nid'] = torch.arange(len(go_nodes), dtype=torch.long)
        self.G[go_ntype].num_nodes = len(go_nodes)
        self.num_nodes_dict[go_ntype] = len(go_nodes)
        self.node_types.append(go_ntype)

        self.nodes[go_ntype] = pd.Index(go_nodes)

        # Edges between RNA nodes and GO terms
        train_go_ann, valid_go_ann, test_go_ann = ontology.annotation_train_val_test_split(
            train_date=train_date, valid_date=valid_date, test_date=test_date, groupby=["gene_name"], **kwargs)
        go_ann = pd.concat([train_go_ann, valid_go_ann, test_go_ann], axis=0)
        nx_graph = nx.from_pandas_edgelist(go_ann["go_id"].explode().to_frame().reset_index().dropna(),
                                           source="gene_name", target="go_id", create_using=nx.DiGraph)
        metapath = (self.head_node_type, "associated", go_ntype)
        self.G[metapath].edge_index = get_edge_index(nx_graph,
                                                     nodes_A=self.nodes[metapath[0]], nodes_B=go_nodes)

        self.set_train_test_split(ontology, train_date, valid_date, test_date, **kwargs)

    def set_train_test_split(self, ontology, train_date='2017-06-15', valid_date='2017-11-15',
                             test_date='2021-12-31', **kwargs) -> None:
        # Edges between RNA nodes and GO terms
        train_go_ann, valid_go_ann, test_go_ann = ontology.annotation_train_val_test_split(
            train_date=train_date, valid_date=valid_date, test_date=test_date, groupby=["gene_name"], **kwargs)

        # Set test nodes as new nodes in annotations
        train_node_list = train_go_ann.index
        valid_node_list = valid_go_ann.index.drop(train_go_ann.index, errors="ignore")
        test_node_list = test_go_ann.index.drop(train_go_ann.index, errors="ignore") \
            .drop(valid_go_ann.index, errors="ignore")

        # Set train test split on hetero graph
        train_idx = {ntype: ntype_nids.get_indexer_for(ntype_nids.intersection(train_node_list)) \
                     for ntype, ntype_nids in self.nodes.items()}
        valid_idx = {ntype: ntype_nids.get_indexer_for(ntype_nids.intersection(valid_node_list)) \
                     for ntype, ntype_nids in self.nodes.items()}
        test_idx = {ntype: ntype_nids.get_indexer_for(ntype_nids.intersection(test_node_list)) \
                    for ntype, ntype_nids in self.nodes.items()}

        for ntype in self.node_types:
            if self.G[ntype].num_nodes is None:
                self.G[ntype].num_nodes = len(self.nodes[ntype])

            if ntype in train_idx:
                mask = torch.zeros(self.G[ntype].num_nodes, dtype=torch.bool)
                mask[train_idx[ntype]] = 1
                self.G[ntype].train_mask = mask

            if ntype in valid_idx:
                mask = torch.zeros(self.G[ntype].num_nodes, dtype=torch.bool)
                mask[valid_idx[ntype]] = 1
                self.G[ntype].valid_mask = mask

            if ntype in test_idx:
                mask = torch.zeros(self.G[ntype].num_nodes, dtype=torch.bool)
                mask[test_idx[ntype]] = 1
                self.G[ntype].test_mask = mask

            print(ntype, "train", train_idx[ntype].size, "valid", valid_idx[ntype].size, "test", test_idx[ntype].size)

    def create_graph_sampler(self, graph: HeteroData, batch_size: int,
                             node_mask: Tensor, transform_fn=None, num_workers=10, verbose=False, **kwargs):
        min_expansion_size = min(self.neighbor_sizes)
        # max_expansion_size = self.num_nodes_dict[self.go_ntype]
        max_expansion_size = 100

        if self.neighbor_loader == "NeighborLoader":
            self.num_neighbors = {
                metapath: self.neighbor_sizes \
                # if hasattr(self, "go_ntype") and metapath[0] != self.go_ntype and metapath[-1] != self.go_ntype \
                #     else [max_expansion_size, ] * len(self.neighbor_sizes)
                for metapath in self.metapaths}

        elif self.neighbor_loader == "HGTLoader":
            self.num_neighbors = {
                ntype: self.neighbor_sizes \
                # if hasattr(self, "go_ntype") and ntype != self.go_ntype \
                #     else [max_expansion_size, ] * len(self.neighbor_sizes)
                for ntype in self.node_types}

        print(f"{self.neighbor_loader} neighbor_sizes:") if verbose else None
        pprint(self.num_neighbors, width=300) if verbose else None

        if self.neighbor_loader == "NeighborLoader":
            graph_loader_cls = NeighborLoader
        elif self.neighbor_loader == "HGTLoader":
            graph_loader_cls = HGTLoader

        dataset = graph_loader_cls(graph,
                                   num_neighbors=self.num_neighbors,
                                   batch_size=batch_size,
                                   # directed=True,
                                   transform=transform_fn,
                                   input_nodes=(self.head_node_type, node_mask),
                                   shuffle=True,
                                   num_workers=num_workers,
                                   **kwargs)
        return dataset

    def transform_heterograph(self, batch: HeteroData):
        X = {}
        X["x_dict"] = {ntype: x for ntype, x in batch.x_dict.items() if x.size(0)}
        X["edge_index_dict"] = {metapath: edge_index for metapath, edge_index in batch.edge_index_dict.items() \
                                # if "associated" not in metapath[1]
                                }
        X["global_node_index"] = {ntype: nid for ntype, nid in batch.nid_dict.items() if nid.numel()}
        X['sizes'] = {ntype: size for ntype, size in batch.num_nodes_dict.items() if size}
        X['batch_size'] = batch.batch_size_dict

        if hasattr(batch, "sequence_dict") and hasattr(self, "seq_tokenizer"):
            X["sequences"] = {}
            for ntype in X["global_node_index"]:
                if not hasattr(batch[ntype], "sequence") or ntype not in self.seq_tokenizer.tokenizers: continue
                X["sequences"][ntype] = self.seq_tokenizer.encode_sequences(batch, ntype=ntype, max_length=None)

        y_dict = {ntype: y for ntype, y in batch.y_dict.items() if y.size(0)}
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
        else:
            weights = None

        return X, y_dict, weights

    def full_batch(self):
        return self.transform_heterograph(self.G)

    def get_projection_pos(self, X: Dict[str, Dict[str, Tensor]], embeddings: Dict[str, Tensor],
                           weights: Optional[Dict[str, Series]] = None,
                           losses: Dict[str, Tensor] = None, return_all=False,
                           ) -> DataFrame:
        """
        Collect node metadata for all nodes in X["global_node_index"]
        Args:
            X (): a batch's dict of data
            embeddings (): Embeddings of nodes in the `X` batch
            weights (Optional[Dict[str, Series]]): A Dict of ntype and a Pandas Series same same as number of nodes
                where entries > 0 are returned.

        Returns:
            node_metadata (DataFrame)
        """
        if return_all and hasattr(self, "node_metadata"):
            return self.node_metadata

        global_node_index = {ntype: nids.numpy() for ntype, nids in X["global_node_index"].items() \
                             if ntype in embeddings}

        # Concatenated list of node embeddings and other metadata
        nodes_emb = {ntype: embeddings[ntype].detach().numpy() for ntype in embeddings}
        nodes_emb = np.concatenate([nodes_emb[ntype] for ntype in global_node_index])

        nodes_train_valid_test = np.vstack([
            np.concatenate([self.G[ntype].train_mask[nids].numpy() for ntype, nids in global_node_index.items()]),
            np.concatenate([self.G[ntype].valid_mask[nids].numpy() for ntype, nids in global_node_index.items()]),
            np.concatenate([self.G[ntype].test_mask[nids].numpy() for ntype, nids in global_node_index.items()])],
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


        # Rename Gene Ontology namespace for GO_term ntypes
        if hasattr(self, "go_namespace") and hasattr(self, "go_ntype") and self.go_ntype in self.nodes:
            go_namespace = {k: v for k, v in zip(self.nodes[self.go_ntype], self.go_namespace)}
            rename_ntype = pd.Series(df["node"].map(go_namespace), index=df.index).dropna()

            df["ntype"].update(rename_ntype)
            df["ntype"] = df["ntype"].replace(
                {"biological_process": "BP", "molecular_function": "MF", "cellular_component": "CC", })

        tsne = UMAP(n_components=2, n_jobs=-1)
        # tsne = MulticoreTSNE.MulticoreTSNE(n_components=2, n_jobs=-1)
        nodes_pos = tsne.fit_transform(nodes_emb)
        nodes_pos = {node_name: pos for node_name, pos in zip(df.index, nodes_pos)}
        df[['pos1', 'pos2']] = np.vstack(df.index.map(nodes_pos))

        # Reset index
        df = df.reset_index().set_index(["ntype", "nid"])

        # Update all nodes embeddings
        if not hasattr(self, "node_metadata"):
            self.node_metadata = df
        else:
            self.node_metadata.update(df)

        # return only nodes that have > 0 weights (used for visualization of node clf models)
        if weights is not None:
            nodes_weight = {ntype: weights[ntype].detach().numpy() \
                if isinstance(weights[ntype], Tensor) else weights[ntype] for ntype in weights}
            nodes_weight = np.concatenate([nodes_weight[ntype] for ntype in global_node_index]).astype(bool)

            return df.loc[nodes_weight]

        return df

    def to_networkx(self, nodes: Dict[str, Union[List[str], List[int]]] = None,
                    edge_index_dict: Dict[Tuple[str, str, str], Tensor] = [],
                    global_node_idx: Dict[str, Tensor] = None) -> nx.MultiDiGraph:
        G = nx.MultiDiGraph()

        if not edge_index_dict:
            edge_index_dict = self.G.edge_index_dict
        elif isinstance(edge_index_dict, list):
            edge_index_dict = {m: eidx for m, eidx in self.G.edge_index_dict if
                               m in edge_index_dict or m[1] in edge_index_dict}

        edge_list = convert_to_nx_edgelist(nodes=self.nodes, edge_index_dict=edge_index_dict,
                                           global_node_idx=global_node_idx)
        for etype, edges in edge_list.items():
            G.add_edges_from(edges, etype=etype)

        if nodes:
            filter_nodes = []

            for ntype, node_list in nodes.items():
                if isinstance(node_list, Tensor):
                    node_list = node_list.detach().numpy().tolist()

                if all(isinstance(node, str) for node in node_list):
                    select_nodes = ntype + "-" + pd.Index(node_list)
                elif all(isinstance(node, int) for node in node_list):
                    select_nodes = ntype + "-" + self.nodes[ntype][nodes[ntype]]
                else:
                    print([type(node) for node in node_list])
                    select_nodes = []

                if G.number_of_nodes() > 0 and set(select_nodes).issubset(set(G.nodes())):
                    filter_nodes.extend(select_nodes)
                else:
                    G.add_nodes_from(select_nodes)

            if len(filter_nodes):
                H = G.copy()
                H = G.subgraph(filter_nodes)
        else:
            H = G.copy()
            H.remove_nodes_from(list(nx.isolates(H)))

        return H

    # @property
    # def training_idx(self):
    #     return torch.arange(self.G[self.head_node_type].num_nodes)[self.G[self.head_node_type].train_mask]
    #
    # @property
    # def validation_idx(self):
    #     return torch.arange(self.G[self.head_node_type].num_nodes)[self.G[self.head_node_type].valid_mask]
    #
    # @property
    # def testing_idx(self):
    #     return torch.arange(self.G[self.head_node_type].num_nodes)[self.G[self.head_node_type].train_mask]

    def train_dataloader(self, collate_fn=None, batch_size=128, num_workers=0, **kwargs):
        dataset = self.create_graph_sampler(self.G, batch_size, node_mask=self.G[self.head_node_type].train_mask,
                                            transform_fn=self.transform_heterograph, num_workers=num_workers)

        return dataset

    def valid_dataloader(self, collate_fn=None, batch_size=128, num_workers=0, **kwargs):
        dataset = self.create_graph_sampler(self.G, batch_size, node_mask=self.G[self.head_node_type].valid_mask,
                                            transform_fn=self.transform_heterograph, num_workers=num_workers)

        return dataset

    def test_dataloader(self, collate_fn=None, batch_size=128, num_workers=0, **kwargs):
        dataset = self.create_graph_sampler(self.G, batch_size, node_mask=self.G[self.head_node_type].test_mask,
                                            transform_fn=self.transform_heterograph, num_workers=num_workers)

        return dataset

    def split_labels_by_go_namespace(self, y: Union[Tensor, Dict[str, Tensor], np.ndarray]):
        assert hasattr(self, 'go_namespace')
        go_namespaces = self.go_namespace[self.classes]

        y_dict = {}
        for namespace in np.unique(go_namespaces):
            mask = go_namespaces == namespace

            if isinstance(y, (Tensor, np.ndarray, DataFrame)):
                y_dict[namespace] = y[:, mask]
            elif isinstance(y, dict):
                for ntype, labels in y.items():
                    y_dict.setdefault(ntype, {})[namespace] = labels[:, mask]

        return y_dict

    def split_edge_index_by_go_namespace(self, edge_index_dict: Dict[Tuple[str, str, str], Tensor],
                                         batch_to_global: Dict[str, Tensor] = None,
                                         edge_pos: Dict[Tuple[str, str, str], Tensor] = None,
                                         mode=None) -> Dict[Tuple[str, str, str], Tensor]:
        if not hasattr(self, "go_namespace"):
            return edge_index_dict

        out_edge_index_dict = {}

        for metapath, edge_index in edge_index_dict.items():
            tail_type = metapath[-1]

            # Pos or neg edges
            if edge_index.size(0) == 2:
                if batch_to_global is not None:
                    go_terms = batch_to_global[tail_type][edge_index[1]]
                else:
                    go_terms = edge_index[1]
                go_namespaces = self.go_namespace[go_terms]

                for namespace in np.unique(go_namespaces):
                    mask = go_namespaces == namespace
                    new_metapath = metapath[:-1] + (namespace,)

                    if not isinstance(mask, bool):
                        out_edge_index_dict[new_metapath] = edge_index[:, mask]
                    else:
                        out_edge_index_dict[new_metapath] = edge_index

            # Sampled head_batch or tail_batch
            elif mode == "head_batch":
                if batch_to_global is not None:
                    go_terms = batch_to_global[tail_type][edge_pos[metapath][1]]
                else:
                    go_terms = edge_pos[metapath][1]

                go_namespaces = self.go_namespace[go_terms]

                for namespace in np.unique(go_namespaces):
                    mask = go_namespaces == namespace
                    new_metapath = metapath[:-1] + (namespace,)

                    if not isinstance(mask, bool):
                        out_edge_index_dict[new_metapath] = edge_index[mask]
                    else:
                        out_edge_index_dict[new_metapath] = edge_index

        return out_edge_index_dict





class HeteroLinkPredDataset(HeteroNodeClfDataset):
    def __init__(self, dataset: HeteroData,
                 negative_sampling_size=1000,
                 seq_tokenizer: SequenceTokenizers = None,
                 neighbor_loader: str = "NeighborLoader",
                 neighbor_sizes: Union[List[int], Dict[str, List[int]]] = [128, 128], node_types: List[str] = None,
                 metapaths: List[Tuple[str, str, str]] = None, head_node_type: str = None, edge_dir: str = "in",
                 reshuffle_train: float = None, add_reverse_metapaths: bool = True, inductive: bool = False, **kwargs):
        super().__init__(dataset, seq_tokenizer, neighbor_loader, neighbor_sizes, node_types, metapaths, head_node_type,
                         edge_dir, reshuffle_train, add_reverse_metapaths, inductive, **kwargs)
        self.negative_sampling_size = negative_sampling_size
        self.eval_negative_sampling_size = 1000

        self.neighbor_sizes = neighbor_sizes
        self.multilabel = False

        self.graph_sampler = self.create_graph_sampler(self.G, batch_size=1,
                                                       node_mask=torch.ones(self.G[self.head_node_type].num_nodes),
                                                       transform_fn=super().transform_heterograph,
                                                       num_workers=0)

    def add_ontology_edges(self, ontology,
                           train_date='2017-06-15', valid_date='2017-11-15', test_date='2021-12-31',
                           go_ntype="GO_term",
                           metapaths: List[Tuple[str, str, str]] = None,
                           add_annotation_as_edges=True):

        self._name = f"{self.head_node_type}-{go_ntype}-{train_date}"
        if ontology.data.index.name != "go_id":
            ontology.data.set_index("go_id", inplace=True)

        all_go = set(ontology.network.nodes).intersection(ontology.data.index)
        go_nodes = np.array(list(all_go))
        self.go_ntype = go_ntype

        # Edges between GO terms
        edge_types = {e for u, v, e in ontology.network.edges}
        edge_index_dict = ontology.to_scipy_adjacency(nodes=go_nodes, edge_types=edge_types,
                                                      reverse=not self.use_reverse,
                                                      format="pyg", d_ntype=go_ntype)
        for metapath, edge_index in edge_index_dict.items():
            if edge_index.size(1) < 100 or (
                    metapaths and metapath not in metapaths and metapath[1] not in metapaths): continue
            self.metapaths.append(metapath)
            self.G[metapath].edge_index = edge_index
            self.edge_index_dict[metapath] = edge_index

            if self.use_reverse:
                rev_metapath = reverse_metapath_name(metapath)
                self.metapaths.append(rev_metapath)
                self.G[rev_metapath].edge_index = edge_index[[1, 0], :]
                self.edge_index_dict[rev_metapath] = edge_index[[1, 0], :]

        # Cls node attrs
        for attr, values in ontology.data.loc[go_nodes][["name", "namespace", "def"]].iteritems():
            self.G[go_ntype][attr] = values.to_numpy()

        self.G[go_ntype]['nid'] = torch.arange(len(go_nodes), dtype=torch.long)
        self.G[go_ntype].num_nodes = len(go_nodes)
        self.num_nodes_dict[go_ntype] = len(go_nodes)
        self.node_types.append(go_ntype)
        self.go_namespace = self.G[go_ntype].namespace

        # Set sequence
        self.nodes[go_ntype] = pd.Index(go_nodes)
        self.G[go_ntype]["sequence"] = pd.Series(self.G[go_ntype]["name"] + " : " + self.G[go_ntype]["def"],
                                                 index=self.nodes[go_ntype])

        # Edges between RNA nodes and GO terms
        if hasattr(ontology, "gaf_annotations"):
            self.load_annotation_edges(ontology, go_nodes=go_nodes, train_date=train_date, valid_date=valid_date,
                                       test_date=test_date)

        # Add the training pos edges to hetero graph
        if add_annotation_as_edges:
            edge_index_dict, _ = TripletDataset.get_relabled_edge_index(
                triples={k: tensor[self.training_idx] for k, tensor in self.triples.items() if not is_negative(k)},
                global_node_index={ntype: torch.arange(len(nodelist)) for ntype, nodelist in self.nodes.items()},
                metapaths=self.pred_metapaths, relation_ids_all=self.triples["relation"].unique())

            for metapath, edge_index in edge_index_dict.items():
                self.G[metapath].edge_index = edge_index
                self.metapaths.append(metapath)

                if self.use_reverse:
                    rev_metapath = reverse_metapath_name(metapath)
                    self.metapaths.append(rev_metapath)
                    self.G[rev_metapath].edge_index = edge_index[[1, 0], :]

        if add_annotation_as_edges and hasattr(self, "network"):
            for ntype in self.network.annotations.index.drop(["MessengerRNA", "Protein"], errors="ignore"):
                annotations = self.network.annotations[ntype]["go_id"].dropna()
                source_ntype = annotations.index.name
                nx_graph = nx.from_pandas_edgelist(annotations.explode().to_frame().reset_index().dropna(),
                                                   source=source_ntype, target="go_id", create_using=nx.Graph)
                metapath = (ntype, "associated", go_ntype)
                self.metapaths.append(metapath)

                edge_index = get_edge_index(nx_graph, nodes_A=self.nodes[metapath[0]], nodes_B=go_nodes)
                self.G[metapath].edge_index = edge_index
                print(metapath, nx_graph.number_of_edges())

                if self.use_reverse:
                    rev_metapath = reverse_metapath_name(metapath)
                    self.metapaths.append(rev_metapath)

                    self.G[rev_metapath].edge_index = edge_index[[1, 0], :]
                    print(rev_metapath, nx_graph.number_of_edges())

        # Reinstantiate graph sampler since hetero graph was modified
        self.graph_sampler = self.create_graph_sampler(self.G, batch_size=1,
                                                       node_mask=torch.ones(self.G[self.head_node_type].num_nodes),
                                                       transform_fn=super().transform_heterograph,
                                                       num_workers=0, verbose=True)

    def load_annotation_edges(self, ontology, go_nodes: List[str],
                              train_date: str, valid_date: str, test_date: str):
        train_go_ann, valid_go_ann, test_go_ann = ontology.annotation_train_val_test_split(
            train_date=train_date, valid_date=valid_date, test_date=test_date,
            groupby=["gene_name", "Qualifier"], filter_go_id=go_nodes)
        self.triples = {}
        self.triples_neg = {}
        pos_train_valid_test_sizes = []
        neg_train_valid_test_sizes = []
        self.pred_metapaths = [(self.head_node_type, etype, self.go_ntype) \
                               for etype in train_go_ann.reset_index()["Qualifier"].unique()]

        # Process train, validation, and test annotations
        for go_ann in [train_go_ann, valid_go_ann, test_go_ann]:
            # True Positive links (undirected)
            edges_df = go_ann["go_id"].dropna().explode().to_frame().reset_index()
            nx_graph = nx.from_pandas_edgelist(edges_df,
                                               source="gene_name", target="go_id", edge_attr=True,
                                               edge_key="Qualifier", create_using=nx.MultiGraph)

            metapaths = {(self.head_node_type, e, self.go_ntype) for u, v, e in nx_graph.edges}
            edge_index_dict = to_scipy_adjacency(nx_graph, nodes=self.nodes,
                                                 edge_types=metapaths.intersection(self.pred_metapaths),
                                                 reverse=None, format="pyg")
            for metapath, edge_index in edge_index_dict.items():
                if metapath not in self.pred_metapaths: continue
                relation_ids = torch.tensor([self.pred_metapaths.index(metapath)] * edge_index.size(1))

                self.triples.setdefault("head", []).append(edge_index[0])
                self.triples.setdefault("relation", []).append(relation_ids)
                self.triples.setdefault("tail", []).append(edge_index[1])

            pos_train_valid_test_sizes.append(num_edges(edge_index_dict))

            # True Negative links (undirected)
            nx_graph = nx.from_pandas_edgelist(go_ann["neg_go_id"].dropna().explode().to_frame().reset_index(),
                                               source="gene_name", target="neg_go_id", edge_attr=True,
                                               edge_key="Qualifier", create_using=nx.MultiGraph)
            metapaths = {(self.head_node_type, e, self.go_ntype) for u, v, e in nx_graph.edges}
            neg_edge_index_dict = to_scipy_adjacency(nx_graph, nodes=self.nodes,
                                                     edge_types=metapaths.intersection(self.pred_metapaths),
                                                     reverse=None, format="pyg")
            for metapath, edge_index in neg_edge_index_dict.items():
                if metapath not in self.pred_metapaths: continue
                relation_ids = torch.tensor([self.pred_metapaths.index(metapath)] * edge_index.size(1))

                self.triples.setdefault("head_neg", []).append(edge_index[0])
                self.triples.setdefault("relation_neg", []).append(relation_ids)
                self.triples.setdefault("tail_neg", []).append(edge_index[1])

            neg_train_valid_test_sizes.append(num_edges(neg_edge_index_dict))

        print("pos_train_valid_test_sizes", pos_train_valid_test_sizes)
        print("neg_train_valid_test_sizes", neg_train_valid_test_sizes)

        # Collect all edges
        self.triples = {key: torch.cat(li) for key, li in self.triples.items()}
        self.global_node_index = {ntype: torch.arange(len(nodelist)) for ntype, nodelist in self.nodes.items()}

        # Adjacency of pos edges (for neg sampling)
        edge_index_dict, edge_neg_dict = TripletDataset.get_relabled_edge_index(
            triples=self.triples,
            global_node_index={ntype: torch.arange(len(nodelist)) for ntype, nodelist in self.nodes.items()},
            metapaths=self.pred_metapaths,
            relation_ids_all=self.triples["relation"].unique())

        self.triples_adj: Dict[Tuple[str, str, str], SparseTensor] = edge_index_to_adjs(edge_index_dict,
                                                                                        nodes=self.nodes)

        # Train/valid/test positive edges
        self.training_idx = torch.arange(0, pos_train_valid_test_sizes[0])
        self.validation_idx = torch.arange(pos_train_valid_test_sizes[0],
                                           pos_train_valid_test_sizes[0] + pos_train_valid_test_sizes[1])
        self.testing_idx = torch.arange(pos_train_valid_test_sizes[0] + pos_train_valid_test_sizes[1],
                                        pos_train_valid_test_sizes[0] + pos_train_valid_test_sizes[1] + \
                                        pos_train_valid_test_sizes[2])
        # Train/valid/test positive edges
        self.training_idx_neg = torch.arange(0, neg_train_valid_test_sizes[0])
        self.validation_idx_neg = torch.arange(neg_train_valid_test_sizes[0],
                                               neg_train_valid_test_sizes[0] + neg_train_valid_test_sizes[1])
        self.testing_idx_neg = torch.arange(neg_train_valid_test_sizes[0] + neg_train_valid_test_sizes[1],
                                            neg_train_valid_test_sizes[0] + neg_train_valid_test_sizes[1] + \
                                            neg_train_valid_test_sizes[2])

        # Set node train_mask, valid_mask, and test_mask based on train/valid/test edges
        train_nodes = gather_node_dict(self.get_edge_index_dict_from_triples(self.training_idx)[0])
        valid_nodes = gather_node_dict(self.get_edge_index_dict_from_triples(self.validation_idx)[0])
        test_nodes = gather_node_dict(self.get_edge_index_dict_from_triples(self.testing_idx)[0])

        for ntype in train_nodes:
            self.G[ntype].train_mask = F.one_hot(train_nodes[ntype],
                                                 num_classes=self.G[ntype].num_nodes).sum(0).bool()
            self.G[ntype].valid_mask = F.one_hot(valid_nodes[ntype],
                                                 num_classes=self.G[ntype].num_nodes).sum(0).bool()
            self.G[ntype].test_mask = F.one_hot(test_nodes[ntype],
                                                num_classes=self.G[ntype].num_nodes).sum(0).bool()

    def get_prior(self) -> Tensor:
        return torch.tensor(1) / (1 + self.negative_sampling_size)

    @staticmethod
    def get_relabled_edge_index(edge_index_dict: Dict[Tuple[str, str, str], Tensor],
                                global_node_index: Dict[str, Tensor],
                                global2batch: Dict[str, Tensor] = None,
                                mode=None) -> Dict[Tuple[str, str, str], Tensor]:
        if global2batch is None:
            global2batch = {
                node_type: dict(zip(global_node_index[node_type].numpy(), range(len(global_node_index[node_type])))) \
                for node_type in global_node_index}

        renamed_edge_index_dict = {}
        for metapath, edge_index in edge_index_dict.items():
            head_type, edge_type, tail_type = metapath

            if edge_index.size(0) == 2 and mode is None:
                sources = edge_index[0].apply_(global2batch[head_type].get)
                targets = edge_index[-1].apply_(global2batch[tail_type].get)

                renamed_edge_index_dict[metapath] = torch.stack([sources, targets], dim=1).t()

            elif mode == "head_batch":
                renamed_edge_index_dict[metapath] = edge_index.apply_(global2batch[head_type].get)
            elif mode == "tail_batch":
                renamed_edge_index_dict[metapath] = edge_index.apply_(global2batch[tail_type].get)
            else:
                raise Exception("Must provide an edge_index with shape (2, N) or pass 'head_batch' or 'tail_batch' "
                                "in `mode`.")

        return renamed_edge_index_dict

    def transform(self, edge_idx: List[int], mode=None):
        if not isinstance(edge_idx, torch.LongTensor):
            edge_idx = torch.LongTensor(edge_idx)

        # Get train/valid/test indices
        if edge_idx[0] in self.training_idx:
            mode = "train" if mode is None else mode
            max_negative_sampling_size = self.negative_sampling_size
            all_neg_edge_idx = self.training_idx_neg
        elif edge_idx[0] in self.validation_idx:
            mode = "valid" if mode is None else mode
            max_negative_sampling_size = self.eval_negative_sampling_size
            all_neg_edge_idx = self.validation_idx_neg
        elif edge_idx[0] in self.testing_idx:
            mode = "test" if mode is None else mode
            max_negative_sampling_size = self.eval_negative_sampling_size
            all_neg_edge_idx = self.testing_idx_neg

        # True pos edges
        edge_pos, edge_neg = self.get_edge_index_dict_from_triples(edge_idx, neg_edge_idx=all_neg_edge_idx)

        # Get all nodes induced by sampled edges
        if num_edges(edge_neg):
            query_edges = {
                metapath: torch.cat([edge_pos[metapath] if metapath in edge_pos else torch.tensor([], dtype=torch.long),
                                     edge_neg[metapath] if metapath in edge_neg else torch.tensor([], dtype=torch.long)
                                     ], dim=1) \
                for metapath in set(edge_pos).union(set(edge_neg))}
        else:
            query_edges = edge_pos

        query_nodes = gather_node_dict(query_edges)

        # Add random GO term nodes for negative sampling
        go_nodes_proba = 1 - F.one_hot(query_nodes[self.go_ntype], num_classes=self.num_nodes_dict[self.go_ntype]) \
            .sum(axis=0).to(torch.float)
        go_nids = torch.multinomial(go_nodes_proba, num_samples=self.negative_sampling_size, replacement=False)
        query_nodes[self.go_ntype] = torch.cat([query_nodes[self.go_ntype], go_nids])

        # Get subgraph induced by neighborhood hopping from the query nodes
        X, _, _ = self.graph_sampler.transform_fn(self.graph_sampler.collate_fn(query_nodes))

        # Edge_pos must be global index, not batch index
        edge_pos_split = self.split_edge_index_by_go_namespace(edge_pos, batch_to_global=None)
        head_batch, tail_batch = self.generate_negative_sampling(edge_pos_split,
                                                                 global_node_index=X["global_node_index"],
                                                                 max_negative_sampling_size=max_negative_sampling_size,
                                                                 mode=mode)

        # Rename node index from global to batch
        global2batch = {ntype: dict(zip(
            X["global_node_index"][ntype].numpy(),
            range(len(X["global_node_index"][ntype])))
        ) for ntype in X["global_node_index"]}

        edge_pos = self.get_relabled_edge_index(edge_pos, global_node_index=X["global_node_index"],
                                                global2batch=global2batch)
        edge_pos = self.split_edge_index_by_go_namespace(edge_pos, batch_to_global=X["global_node_index"])

        y = {"edge_pos": edge_pos, }
        if num_edges(edge_neg):
            edge_neg = self.get_relabled_edge_index(edge_neg, global_node_index=X["global_node_index"],
                                                    global2batch=global2batch, )
            y['edge_neg'] = self.split_edge_index_by_go_namespace(edge_neg, batch_to_global=X["global_node_index"])

        # Negative sampling
        y.update({"head_batch": head_batch, "tail_batch": tail_batch, })

        edge_weights = None
        return X, y, edge_weights

    def get_edge_index_dict_from_triples(self, edge_idx, neg_edge_idx=None):
        triples = {k: v[edge_idx] for k, v in self.triples.items() if not is_negative(k)}

        # If ensures same number of true neg edges to true pos edges
        if neg_edge_idx is not None:
            neg_edge_idx = np.random.choice(neg_edge_idx, size=min(edge_idx.numel(), neg_edge_idx.numel()),
                                            replace=False)
            triples_neg = {k: v[neg_edge_idx] for k, v in self.triples.items() if is_negative(k)}
            triples.update(triples_neg)

        # Get edge_index_dict from triplets
        edge_pos, edge_neg = TripletDataset.get_relabled_edge_index(triples=triples,
                                                                    global_node_index=self.global_node_index,
                                                                    metapaths=self.pred_metapaths)
        return edge_pos, edge_neg

    def generate_negative_sampling(self, edge_pos: Dict[Tuple[str, str, str], Tensor],
                                   global_node_index: Dict[str, Tensor], max_negative_sampling_size: int,
                                   mode: str = None) -> \
            Tuple[Dict[Tuple[str, str, str], Tensor], Dict[Tuple[str, str, str], Tensor]]:
        head_batch = {}
        tail_batch = {}

        sampling_size = self.get_neg_sampling_size(edge_pos, global_node_index=global_node_index,
                                                   max_samples=max_negative_sampling_size, mode=mode)

        # Perform negative sampling
        for metapath, edge_index in edge_pos.items():
            head_type, tail_type = metapath[0], metapath[-1]
            adj: SparseTensor = self.triples_adj[metapath] \
                if metapath in self.triples_adj \
                else self.triples_adj[metapath[:-1] + (self.go_ntype,)]

            # head noise distribution
            head_neg_nodes = global_node_index[head_type]
            head_prob_dist = 1 - adj[head_neg_nodes, edge_index[1]].to_dense().T

            head_batch[metapath] = torch.multinomial(head_prob_dist, num_samples=sampling_size,
                                                     replacement=True)

            # Tail noise distribution
            tail_neg_nodes = global_node_index[tail_type if tail_type in global_node_index else self.go_ntype]
            tail_prob_dist = 1 - adj[edge_index[0], tail_neg_nodes].to_dense()

            # Only generate negative tail_batch within BPO, CCO, or MFO terms of the positive edge's tail go_type
            for go_type in ['biological_process', 'cellular_component', 'molecular_function']:
                if go_type != tail_type: continue

                go_terms_mask = self.go_namespace[tail_neg_nodes] != go_type
                tail_prob_dist[:, go_terms_mask] = 0

            # print(tail_prob_dist.sum(1), "num_samples", sampling_size, "tail_neg_nodes", tail_neg_nodes.shape)
            tail_batch[metapath] = torch.multinomial(tail_prob_dist, num_samples=sampling_size,
                                                     replacement=True)

        return head_batch, tail_batch

    def get_neg_sampling_size(self, edge_index_dict: Dict[Tuple[str, str, str], Tensor],
                              global_node_index: Dict[str, Tensor],
                              max_samples: int, mode: str) -> int:
        # Choose neg sampling size by twice (minimum) the count of head or tail nodes, if it's less than self.negative_sampling_size
        if mode == "train":
            head_tail_ntypes = [ntype for metapath in edge_index_dict for ntype in [metapath[0], metapath[-1]]]
            min_num_node = min(
                global_node_index[ntype if ntype in global_node_index else self.ntype_mapping[ntype]].size(0) \
                for ntype in head_tail_ntypes)

            sampling_size = min(min_num_node, max_samples // 2)

        else:
            sampling_size = max_samples // 2

        sampling_size = int(max(sampling_size, 1))

        return sampling_size

    def full_batch(self):
        return self.transform(edge_idx=torch.cat([self.training_idx, self.validation_idx, self.testing_idx]),
                              mode="test")

    def to_networkx(self, nodes: Dict[str, Union[List[str], List[int]]] = None,
                    metapaths: Union[Dict[Tuple[str, str, str], Tensor], List[Tuple[str, str, str]]] = [],
                    global_node_idx: Dict[str, Tensor] = None,
                    pos_edges: Dict[Tuple[str, str, str], Tensor] = None, ) -> nx.MultiDiGraph:
        G = super().to_networkx(nodes=nodes, metapaths=metapaths, global_node_idx=global_node_idx)

        if pos_edges is not None:
            edge_list = convert_to_nx_edgelist(nodes=self.nodes, edge_index_dict=pos_edges,
                                               global_node_idx=global_node_idx)
        else:
            edge_list = {}

        for etype, edges in edge_list.items():
            G.add_edges_from(edges, etype=etype)

        return G

    def train_dataloader(self, collate_fn=None, batch_size=128, num_workers=0, shuffle=True, **kwargs):
        dataset = DataLoader(self.training_idx, batch_size=batch_size,
                             collate_fn=self.transform,
                             shuffle=shuffle,
                             num_workers=num_workers, **kwargs)
        return dataset

    def valid_dataloader(self, collate_fn=None, batch_size=128, num_workers=0, shuffle=False, **kwargs):
        dataset = DataLoader(self.validation_idx, batch_size=batch_size,
                             collate_fn=self.transform,
                             shuffle=shuffle,
                             num_workers=num_workers, **kwargs)
        return dataset

    def test_dataloader(self, collate_fn=None, batch_size=128, num_workers=0, shuffle=False, **kwargs):
        dataset = DataLoader(self.testing_idx, batch_size=batch_size,
                             collate_fn=self.transform,
                             shuffle=shuffle,
                             num_workers=num_workers, **kwargs)
        return dataset

    def dataloader(self, edge_idx, batch_size=128, num_workers=0):
        dataset = DataLoader(edge_idx, batch_size=batch_size,
                             collate_fn=self.transform,
                             shuffle=False,
                             num_workers=num_workers)
        return dataset
