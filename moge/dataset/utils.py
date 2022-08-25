import copy
import itertools
from typing import List, Union, Dict, Tuple, Set, Optional, Any

import networkx as nx
import numpy as np
import pandas as pd
import torch
from logzero import logger
from pandas import Series
from torch import Tensor
from torch_geometric.utils import is_undirected
from torch_sparse import SparseTensor, transpose


def is_sorted(arr: Tensor):
    return torch.all(arr[:-1] <= arr[1:])


def split_edge_index_by_namespace(nodes_namespace: Dict[str, Union[np.ndarray, Series]],
                                  edge_index_dict: Dict[Tuple[str, str, str], Tuple[Tensor, Tensor]],
                                  edge_values: Dict[Tuple[str, str, str], Tensor]) \
        -> Tuple[Dict[Tuple[str, str, str], Tuple[Tensor, Tensor]], Dict[Tuple[str, str, str], Tensor]]:
    """
    Split edges in `edge_index_dict` and values in `edge_values` by the node namespace given in `nodes_namespace`.
    When `nodes_namespace` is given as a Dict of ntype keys and the node's namespace values, each metapath will
    split into multiple metapaths on either its src and/or dst nodes if the src_type and dst_type matches the
    `nodes_namespace`'s ntypes.

    Args:
        nodes_namespace (): A Dict of ntype keys and the node's namespace values, where node's namespace values is
            of the same length as all nodes in self.nodes[ntype].
        edge_index_dict (): A Dict of metapath keys and edge index tuple, where the edge index uses global index.
        edge_values (): A Dict of metapath keys and edge values, where the edge values's number of elements equals
            the number of edges in `edge_index_dict[metapath]`

    Returns:
        split_edge_index_dict, split_edge_values
    """
    split_edge_index_dict = {}
    split_edge_values = {}

    for metapath, (src, dst) in edge_index_dict.items():
        if edge_values and metapath not in edge_values: continue
        # assert src.size(0) == edge_values[
        #     metapath].numel(), f"num edges {src.size(0)} != values {edge_values[metapath].numel()}"
        head_type, tail_type = metapath[0], metapath[-1]

        unique_heads = np.unique(nodes_namespace[head_type]) if head_type in nodes_namespace else [None]
        unique_tails = np.unique(nodes_namespace[tail_type]) if tail_type in nodes_namespace else [None]

        for head_namespace, tail_namespace in itertools.product(unique_heads, unique_tails):
            if head_namespace is not None and tail_namespace is not None:
                new_metapath = (head_namespace,) + metapath[1] + (tail_namespace,)
                mask = (nodes_namespace[head_type][src.detach().cpu().numpy()] == head_namespace) & \
                       (nodes_namespace[tail_type][dst.detach().cpu().numpy()] == tail_namespace)
            elif tail_namespace is not None:
                new_metapath = metapath[:-1] + (tail_namespace,)
                mask = nodes_namespace[tail_type][dst.detach().cpu().numpy()] == tail_namespace
            elif head_namespace is not None:
                new_metapath = (head_namespace,) + metapath[1:]
                mask = nodes_namespace[head_type][src.detach().cpu().numpy()] == head_namespace
            else:
                continue

            if isinstance(mask, np.ndarray) and mask.sum():
                split_edge_index_dict[new_metapath] = (src[mask], dst[mask])
                if edge_values:
                    split_edge_values[new_metapath] = edge_values[metapath].view(-1)[mask]
            else:
                split_edge_index_dict[new_metapath] = (src, dst)
                if edge_values:
                    split_edge_values[new_metapath] = edge_values[metapath].view(-1)

    return split_edge_index_dict, split_edge_values if split_edge_values else {}


def gather_node_dict(edge_index_dict: Dict[Tuple[str, str, str], Tensor]) -> Dict[str, Tensor]:
    nodes = {}
    for metapath, edge_index in edge_index_dict.items():
        nodes.setdefault(metapath[0], []).append(edge_index[0])
        nodes.setdefault(metapath[-1], []).append(edge_index[1])

    nodes = {ntype: torch.unique(torch.cat(nids, dim=0)) for ntype, nids in nodes.items()}

    return nodes


def get_relabled_edge_index(triples: Dict[str, Tensor], global_node_index: Dict[str, Tensor],
                            metapaths: List[Tuple[str, str, str]],
                            relation_ids_all: Optional[Tensor] = None,
                            local2batch: Optional[Dict[str, Dict[int, int]]] = None,
                            format: str = "pyg") \
        -> Tuple[Dict[Tuple[str, str, str], Tensor], Dict[Tuple[str, str, str], Tensor]]:
    edges_pos = {}
    edges_neg = {}

    if local2batch is None and not all(is_sorted(node_ids) for ntype, node_ids in global_node_index.items()):
        local2batch = {
            node_type: dict(zip(
                global_node_index[node_type].numpy(),
                range(len(global_node_index[node_type])))
            ) for node_type in global_node_index}

    if relation_ids_all is None:
        relation_ids_all = triples["relation" if "relation" in triples else "relation_neg"].unique()
        assert len(relation_ids_all) == len(metapaths)

    # Get edge_index with batch id
    for relation_id in relation_ids_all:
        metapath = metapaths[relation_id]
        head_type, tail_type = metapath[0], metapath[-1]

        if "relation" in triples:
            mask = triples["relation"] == relation_id
            src = triples["head"][mask]
            dst = triples["tail"][mask]
            if local2batch:
                src, dst = src.apply_(local2batch[head_type].get), dst.apply_(local2batch[tail_type].get)

            if format == "pyg":
                edges_pos[metapath] = torch.stack([src, dst], dim=1).t()
            elif format == "dgl":
                edges_pos[metapath] = (src, dst)

        # Negative edges
        if any(["neg" in k for k in triples.keys()]) and "relation_neg" in triples:
            mask = triples["relation_neg"] == relation_id
            src_neg = triples["head_neg"][mask]
            dst_neg = triples["tail_neg"][mask]
            if local2batch:
                src_neg, dst_neg = src_neg.apply_(local2batch[head_type].get), dst_neg.apply_(
                    local2batch[tail_type].get)

            if format == "pyg":
                edges_neg[metapath] = torch.stack([src_neg, dst_neg], dim=1).t()
            elif format == "dgl":
                edges_neg[metapath] = (src_neg, dst_neg)

        # Negative sampled edges
        if any(["neg" in k for k in triples.keys()]) and "relation" in triples and "relation_neg" not in triples:
            mask = triples["relation"] == relation_id
            src_neg = triples["head_neg"][mask]
            dst_neg = triples["tail_neg"][mask]
            if local2batch:
                src_neg, dst_neg = src_neg.apply_(local2batch[head_type].get), dst_neg.apply_(
                    local2batch[tail_type].get)
            head_batch = torch.stack([src_neg.view(-1), dst.repeat(src_neg.size(1))])
            tail_batch = torch.stack([src.repeat(dst_neg.size(1)), dst_neg.view(-1)])

            if format == "pyg":
                edges_neg[metapath] = torch.cat([head_batch, tail_batch], dim=1)
            elif format == "dgl":
                edge_index = torch.cat([head_batch, tail_batch], dim=1)
                edges_neg[metapath] = (edge_index[0], edge_index[1])

    return edges_pos, edges_neg


def get_edge_index_dict(graph: Tuple[nx.Graph, nx.MultiGraph], nodes: Union[List[str], Dict[str, List[str]]],
                        metapaths: Union[List[str], Tuple[str, str, str], Set[Tuple[str, str, str]]] = None,
                        format="coo", d_ntype="_N") -> Dict[Tuple[str, str, str], Tensor]:
    if metapaths is None and isinstance(graph, nx.MultiGraph):
        metapaths = {e for u, v, e in graph.edges}
    if isinstance(graph, nx.Graph):
        if isinstance(metapaths, (tuple, str)):
            metapaths = [metapaths]
        elif metapaths is None:
            metapaths = ["_E"]

    assert isinstance(metapaths, (list, set)) and isinstance(list(metapaths)[0], (tuple, str))

    edge_index_dict = {}
    for etype in metapaths:
        if isinstance(graph, nx.MultiGraph) and isinstance(etype, str):
            assert not isinstance(nodes, (dict, pd.Series))
            subgraph = graph.edge_subgraph([(u, v, e) for u, v, e in graph.edges if e == etype])
            nodes_A = nodes
            nodes_B = nodes
            metapath = (d_ntype, etype, d_ntype)

        elif isinstance(graph, nx.MultiGraph) and isinstance(etype, tuple):
            assert isinstance(nodes, (dict, pd.Series))
            metapath: Tuple[str, str, str] = etype
            head, etype, tail = metapath
            subgraph = graph.edge_subgraph([(u, v, e) for u, v, e in graph.edges if e == etype])

            nodes_A, nodes_B = nodes[head], nodes[tail]

        elif isinstance(graph, nx.Graph):
            subgraph = graph

            if isinstance(etype, str):
                assert not isinstance(nodes, (dict, pd.Series))
                nodes_A = nodes_B = nodes
                metapath = (d_ntype, etype, d_ntype)
            elif isinstance(etype, tuple):
                head, etype, tail = metapath = etype
                nodes_A, nodes_B = nodes[head], nodes[tail]

        else:
            raise Exception(f"Edge types `{metapaths}` are ill formed.")

        try:
            biadj = nx.bipartite.biadjacency_matrix(subgraph, row_order=nodes_A, column_order=nodes_B, weight=None,
                                                    format="coo")
        except Exception as e:
            logger.error(f"{e.__class__.__name__}:{e} \n"
                         f"{metapath}, {subgraph}")
            # "\n", subgraph, list(subgraph.edges())[:2],
            # '\n', f"nodes_A {len(nodes_A)} {nodes_A[:5]}", f"nodes_A {len(nodes_A)} {nodes_A[:5]}"
            # traceback.print_exc()
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
        elif format == "nx":
            edge_index_dict[metapath] = [(nodes_A[u], nodes_B[v]) for u, v in zip(biadj.row, biadj.col)]

        elif format == "csr":
            edge_index_dict[metapath] = biadj

    return edge_index_dict


def get_edge_attr_keys(nx_graph) -> Set[str]:
    u, v, d = next(itertools.islice(nx_graph.edges(data=True), 1))
    return d.keys()


def get_edge_index_values(nx_graph: nx.Graph, nodes_A: Union[List[str], np.array],
                          nodes_B: Union[List[str], np.array], edge_attrs: List[str] = None, format="pyg") \
        -> Tuple[Union[torch.LongTensor, Tuple[Tensor]], Optional[Dict[str, Tensor]]]:
    values = {}

    if edge_attrs is None or len(edge_attrs) == 0:
        edge_attrs = [None]

    for edge_attr in edge_attrs:
        biadj = nx.bipartite.biadjacency_matrix(nx_graph, row_order=nodes_A, column_order=nodes_B, weight=edge_attr,
                                                format="coo")
        if hasattr(biadj, 'data') and isinstance(biadj.data, np.ndarray) and edge_attr in get_edge_attr_keys(nx_graph):
            values[edge_attr] = biadj.data

    if format == "pyg":
        import torch
        edge_index = torch.stack([torch.tensor(biadj.row, dtype=torch.long), torch.tensor(biadj.col, dtype=torch.long)])
        values = {edge_attr: torch.tensor(edge_value) for edge_attr, edge_value in values.items()}

    elif format == "dgl":
        import torch
        edge_index = (torch.tensor(biadj.row, dtype=torch.int64), torch.tensor(biadj.col, dtype=torch.int64))
        values = {edge_attr: torch.tensor(edge_value) for edge_attr, edge_value in values.items()}

    return edge_index, values


def one_hot_encoder(idx: Tensor):
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


def edge_intersection(edge_index_A: Tensor, edge_index_B: Tensor, remove_duplicates=True):
    A = pd.DataFrame(edge_index_A.T.numpy(), columns=["source", "target"])
    B = pd.DataFrame(edge_index_B.T.numpy(), columns=["source", "target"])
    int_df = pd.merge(A, B, how='inner', on=["source", "target"], sort=True)

    if remove_duplicates:
        int_df = int_df[~int_df.duplicated()]
    return torch.tensor(int_df.to_numpy().T, dtype=torch.long)


def nonduplicate_indices(edge_index):
    edge_df = pd.DataFrame(edge_index.t().numpy())  # shape: (n_edges, 2)
    return ~edge_df.duplicated(subset=[0, 1])

def edge_index_to_adjs(edge_index_dict: Dict[Tuple[str, str, str], Tuple[Tensor, Tensor]],
                       nodes=Dict[str, List[str]]) \
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


def add_reverse_edge_index(edge_index_dict: Dict[Tuple[str], Tensor], num_nodes_dict) -> None:
    reverse_edge_index_dict = {}
    for metapath, edge_index in edge_index_dict.items():
        if is_negative(metapath) or edge_index_dict[metapath] == None: continue

        reverse_metapath = reverse_metapath(metapath)

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
        reverse = reverse_metapath(metapath)
        reverse_metapaths.append(reverse)
    return reverse_metapaths


def reverse_metapath(metapath: Union[Tuple[str, str, str], List[Tuple], Dict[Tuple, Any]]) \
        -> Union[Tuple[str, str, str], List[Tuple], Dict[Tuple, Any]]:
    if isinstance(metapath, list):
        return [reverse_metapath(m) for m in metapath]

    elif isinstance(metapath, dict):
        return {reverse_metapath(m): eid for m, eid in metapath.items()}

    if isinstance(metapath, tuple):
        tokens = []
        for i, token in enumerate(reversed(copy.deepcopy(metapath))):
            if i == 1:
                if len(token) == 2:  # 2 letter string etype
                    rev_etype = token[::-1]
                else:
                    rev_etype = "rev_" + token
                tokens.append(rev_etype)
            else:
                tokens.append(token)

        rev_metapath = tuple(tokens)

        return rev_metapath

    elif isinstance(metapath, str):
        rev_metapath = "".join(reversed(metapath))

    elif isinstance(metapath, (int, np.int)):
        rev_metapath = str(metapath) + "_"
    else:
        raise NotImplementedError(f"{metapath} not supported")


def unreverse_metapath(metapath: Union[Tuple[str, str, str], List[Tuple], Dict[Tuple, Any]]) \
        -> Union[Tuple[str, str, str], List[Tuple], Dict[Tuple, Any]]:
    if isinstance(metapath, list):
        return [unreverse_metapath(m) for m in metapath]

    elif isinstance(metapath, dict):
        return {unreverse_metapath(m): eid for m, eid in metapath.items()}

    if isinstance(metapath, tuple):
        tokens = []
        for i, token in enumerate(reversed(copy.deepcopy(metapath))):
            if i == 1:
                if len(token) == 2:  # 2 letter string etype
                    rev_etype = token[::-1]
                else:
                    rev_etype = token.removeprefix("rev_")
                tokens.append(rev_etype)
            else:
                tokens.append(token)

        rev_metapath = tuple(tokens)
        return rev_metapath

    else:
        raise NotImplementedError(f"{metapath} not supported")


def is_reversed(metapath):
    if isinstance(metapath, tuple):
        return any("rev_" in token for token in metapath)
    elif isinstance(metapath, str):
        return "rev" in metapath


def tag_negative_metapath(metapath: Union[Tuple[str, str, str], List[Tuple], Dict[Tuple, Any]]) \
        -> Union[Tuple[str, str, str], List[Tuple], Dict[Tuple, Any]]:
    if isinstance(metapath, list):
        return [tag_negative_metapath(m) for m in metapath]

    elif isinstance(metapath, dict):
        return {tag_negative_metapath(m): eid for m, eid in metapath.items()}

    elif isinstance(metapath, tuple):
        tokens = []
        for i, token in enumerate(copy.deepcopy(metapath)):
            if i == 1:
                if len(token) == 2:  # 2 letter string etype
                    rev_etype = token[::-1]
                else:
                    rev_etype = f"neg_{token}"
                tokens.append(rev_etype)
            else:
                tokens.append(token)

        rev_metapath = tuple(tokens)

        return rev_metapath
    else:
        raise NotImplementedError(f"{metapath} not supported")


def untag_negative_metapath(metapath: Union[Tuple[str, str, str], List[Tuple], Dict[Tuple, Any]]) \
        -> Union[Tuple[str, str, str], List[Tuple], Dict[Tuple, Any]]:
    if isinstance(metapath, list):
        return [untag_negative_metapath(m) for m in metapath]

    elif isinstance(metapath, dict):
        return {untag_negative_metapath(m): eid for m, eid in metapath.items()}

    elif isinstance(metapath, tuple):
        tokens = []
        for i, token in enumerate(copy.deepcopy(metapath)):
            if i == 1:
                rev_etype = token.removeprefix("neg_")
                tokens.append(rev_etype)
            else:
                tokens.append(token)

        rev_metapath = tuple(tokens)

        return rev_metapath
    else:
        raise NotImplementedError(f"{metapath} not supported")


def is_negative(metapath: Union[Tuple[str, str, str], str]):
    if isinstance(metapath, tuple):
        return any("neg" in token for token in metapath)
    elif isinstance(metapath, str):
        return "neg" in metapath
