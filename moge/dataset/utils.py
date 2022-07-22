import copy
from typing import List, Union, Dict, Tuple, Set

import networkx as nx
import numpy as np
import pandas as pd
import torch
from moge.model.PyG import is_negative
from torch import Tensor
from torch_geometric.utils import is_undirected
from torch_sparse import SparseTensor, transpose


def gather_node_dict(edge_index_dict: Dict[Tuple[str, str, str], Tensor]) -> Dict[str, Tensor]:
    nodes = {}
    for metapath, edge_index in edge_index_dict.items():
        nodes.setdefault(metapath[0], []).append(edge_index[0])
        nodes.setdefault(metapath[-1], []).append(edge_index[1])

    nodes = {ntype: torch.unique(torch.cat(nids, dim=0)) for ntype, nids in nodes.items()}

    return nodes


def to_edge_index_dict(graph: Tuple[nx.DiGraph, nx.MultiGraph], nodes: Union[List[str], Dict[str, List[str]]],
                       edge_types: Union[List[str], Tuple[str, str, str], Set[Tuple[str, str, str]]] = None,
                       reverse=False,
                       format="coo", d_ntype="_N") -> Dict[Tuple[str, str, str], Tensor]:
    if reverse:
        graph = graph.reverse(copy=True)

    if not isinstance(graph, nx.MultiGraph):
        raise NotImplementedError

    if edge_types is None:
        edge_types = {e for u, v, e in graph.edges}
    if not isinstance(edge_types, (list, tuple, set)) and isinstance(graph, nx.Graph):
        edge_types = ["_E"]

    edge_index_dict = {}
    for etype in edge_types:
        if isinstance(graph, nx.MultiGraph) and isinstance(etype, str):
            assert not isinstance(nodes, dict)
            edge_subgraph = graph.edge_subgraph([(u, v, e) for u, v, e in graph.edges if e == etype])
            nodes_A = nodes
            nodes_B = nodes
            metapath = (d_ntype, etype, d_ntype)

        elif isinstance(graph, nx.MultiGraph) and isinstance(etype, tuple):
            assert isinstance(nodes, dict)
            metapath: Tuple[str, str, str] = etype
            head, etype, tail = metapath
            edge_subgraph = graph.edge_subgraph([(u, v, e) for u, v, e in graph.edges if e == etype])

            nodes_A = nodes[head]
            nodes_B = nodes[tail]

        elif etype == "_E":
            assert not isinstance(nodes, dict)
            edge_subgraph = graph.edges
            nodes_A = nodes
            nodes_B = nodes
            metapath = (d_ntype, etype, d_ntype)
        else:
            raise Exception(f"Edge types `{edge_types}` are ill formed.")

        try:
            biadj = nx.bipartite.biadjacency_matrix(edge_subgraph, row_order=nodes_A, column_order=nodes_B, weight=None,
                                                    format="coo")
        except Exception as e:
            print(etype, e)
            continue

        if format == "coo":
            edge_index_dict[metapath] = (biadj.row, biadj.col)
        elif format == "pyg":
            import torch
            edge_index_dict[metapath] = torch.stack(
                [torch.tensor(biadj.row, dtype=torch.long),
                 torch.tensor(biadj.col, dtype=torch.long)])
        elif format == "dgl":
            import torch
            edge_index_dict[metapath] = (torch.tensor(biadj.row, dtype=torch.long),
                                         torch.tensor(biadj.col, dtype=torch.long))
        else:
            edge_index_dict[metapath] = biadj

    return edge_index_dict


def one_hot_encoder(idx: Tensor, embed_dim=None):
    """
    Get one hot embedding of the input tensor.
    Args:
        idx: torch.Tensor, input 1-D tensor.
    Returns:
        one_hot: torch.Tensor, one-hot embedding of x.
    """
    ids = idx.unique()
    id_dict = dict(list(zip(ids.numpy(), np.arange(len(ids)))))
    one_hot = torch.zeros((len(idx), len(ids)))
    for i, u in enumerate(idx):
        if id_dict[u.item()] == 4:
            pass
        else:
            one_hot[i][id_dict[u.item()]] = 1

    return one_hot


def edge_dict_sizes(edge_index_dict):
    return {k: v.shape[1] for k, v in edge_index_dict.items()}


def edge_dict_intersection(edge_index_dict_A, edge_index_dict_B):
    inters = {}
    for metapath, edge_index in edge_index_dict_A.items():
        if metapath not in edge_index_dict_B:
            inters[metapath] = 0
            continue

        inters[metapath] = edge_intersection(edge_index, edge_index_dict_B[metapath])

    return inters


def edge_intersection(edge_index_A, edge_index_B, remove_duplicates=True):
    A = pd.DataFrame(edge_index_A.T.numpy(), columns=["source", "target"])
    B = pd.DataFrame(edge_index_B.T.numpy(), columns=["source", "target"])
    int_df = pd.merge(A, B, how='inner', on=["source", "target"], sort=True)
    if remove_duplicates:
        int_df = int_df[~int_df.duplicated()]
    return torch.tensor(int_df.to_numpy().T, dtype=torch.long)


def nonduplicate_indices(edge_index):
    edge_df = pd.DataFrame(edge_index.t().numpy())  # shape: (n_edges, 2)
    return ~edge_df.duplicated(subset=[0, 1])


def edge_index_to_adjs(edge_index_dict: Dict[Tuple[str, str, str], Tensor], nodes=Dict[str, List[str]]) \
        -> Dict[Tuple[str, str, str], SparseTensor]:
    """

    Args:
        edge_index_dict ():
        nodes (): A Dict of ntype and a list of all nodes.

    Returns:

    """
    adj_dict = {}

    for metapath, edge_index in edge_index_dict.items():
        head_type, tail_type = metapath[0], metapath[-1]

        if isinstance(edge_index, (tuple, list)):
            edge_index, edge_values = edge_index
        else:
            edge_values = None

        adj = SparseTensor.from_edge_index(edge_index,
                                           edge_attr=edge_values,
                                           sparse_sizes=(len(nodes[head_type]), len(nodes[tail_type])))

        adj_dict[metapath] = adj

    return adj_dict

def merge_node_index(old_node_index, new_node_index):
    merged = {k: [v] for k, v in old_node_index.items()}

    for ntype, new_nodes in new_node_index.items():
        if ntype not in old_node_index:
            merged.setdefault(ntype, []).append(new_nodes)
        else:
            merged.setdefault(ntype, []).append(old_node_index[ntype])
            new_nodes_mask = np.isin(new_nodes, old_node_index[ntype], invert=True)
            merged[ntype].append(new_nodes[new_nodes_mask])

        merged[ntype] = torch.cat(merged[ntype], dim=0)
    return merged


def get_edge_index(nx_graph: nx.Graph, nodes_A: Union[List[str], np.array],
                   nodes_B: Union[List[str], np.array]) -> torch.LongTensor:
    biadj = nx.bipartite.biadjacency_matrix(nx_graph,
                                            row_order=nodes_A,
                                            column_order=nodes_B,
                                            format="coo")
    edge_index = torch.stack([torch.tensor(biadj.row, dtype=torch.long),
                              torch.tensor(biadj.col, dtype=torch.long)])

    return edge_index


def add_reverse_edge_index(edge_index_dict: Dict[Tuple[str], Tensor], num_nodes_dict) -> None:
    reverse_edge_index_dict = {}
    for metapath, edge_index in edge_index_dict.items():
        if is_negative(metapath) or edge_index_dict[metapath] == None: continue

        reverse_metapath = reverse_metapath_name(metapath)

        if metapath[0] == metapath[-1] and isinstance(edge_index, Tensor) and is_undirected(edge_index):
            print(f"skipping reverse {metapath} because edges are symmetrical")
            continue

        print("Reversing", metapath, "to", reverse_metapath)
        reverse_edge_index_dict[reverse_metapath] = transpose(index=edge_index_dict[metapath], value=None,
                                                              m=num_nodes_dict[metapath[0]],
                                                              n=num_nodes_dict[metapath[-1]])[0]
    edge_index_dict.update(reverse_edge_index_dict)


def get_reverse_metapaths(metapaths) -> List[Tuple[str]]:
    reverse_metapaths = []
    for metapath in metapaths:
        reverse = reverse_metapath_name(metapath)
        reverse_metapaths.append(reverse)
    return reverse_metapaths


def reverse_metapath_name(metapath: Tuple[str]) -> Tuple[str]:
    if isinstance(metapath, tuple):
        tokens = []
        for i, token in enumerate(reversed(copy.deepcopy(metapath))):
            if i == 1:
                if len(token) == 2:
                    reverse_etype = token[::-1]
                else:
                    reverse_etype = token + "_"
                tokens.append(reverse_etype)
            else:
                tokens.append(token)

        reverse_metapath = tuple(tokens)

    elif isinstance(metapath, str):
        reverse_metapath = "".join(reversed(metapath))

    elif isinstance(metapath, (int, np.int)):
        reverse_metapath = str(metapath) + "_"
    else:
        raise NotImplementedError(f"{metapath} not supported")
    return reverse_metapath
