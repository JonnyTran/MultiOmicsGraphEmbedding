from collections import OrderedDict
from typing import Union, Tuple, Iterable, List

import torch
from torch_sparse import SparseTensor


def tag_negative(metapath):
    if isinstance(metapath, tuple):
        return metapath + ("neg",)
    elif isinstance(metapath, str):
        return metapath + "_neg"
    else:
        return "neg"


def untag_negative(metapath):
    if isinstance(metapath, tuple) and metapath[-1] == "neg":
        return metapath[:-1]
    elif isinstance(metapath, str):
        return metapath.strip("_neg")
    else:
        return metapath


def is_negative(metapath):
    if isinstance(metapath, tuple) and "neg" in metapath:
        return True
    elif isinstance(metapath, str) and "_neg" in metapath:
        return True
    else:
        return False


def adamic_adar(indexA, valueA, indexB, valueB, m, k, n, coalesced=False, sampling=True):
    A = SparseTensor(row=indexA[0], col=indexA[1], value=valueA,
                     sparse_sizes=(m, k), is_sorted=not coalesced)
    B = SparseTensor(row=indexB[0], col=indexB[1], value=valueB,
                     sparse_sizes=(k, n), is_sorted=not coalesced)

    deg_A = A.storage.colcount()
    deg_B = B.storage.rowcount()
    deg_normalized = 1.0 / (deg_A + deg_B).to(torch.float)
    deg_normalized[deg_normalized == float('inf')] = 0.0

    D = SparseTensor(row=torch.arange(deg_normalized.size(0), device=valueA.device),
                     col=torch.arange(deg_normalized.size(0), device=valueA.device),
                     value=deg_normalized.type_as(valueA),
                     sparse_sizes=(deg_normalized.size(0), deg_normalized.size(0)))

    out = A @ D @ B
    row, col, values = out.coo()

    num_samples = min(int(valueA.numel()), int(valueB.numel()), values.numel())
    if sampling and values.numel() > num_samples:
        idx = torch.multinomial(values, num_samples=num_samples,
                                replacement=False)
        row, col, values = row[idx], col[idx], values[idx]

    return torch.stack([row, col], dim=0), values


def join_metapaths(metapath_A, metapath_B):
    output_metapaths = []

    for relation_b in metapath_B:
        for relation_a in metapath_A:
            if relation_a[-1] == relation_b[0]:
                new_relation = relation_a + relation_b[1:]
                output_metapaths.append(new_relation)
    return output_metapaths


def filter_metapaths(metapaths: List[Tuple[str]],
                     order: Union[int, Iterable] = None,
                     head_type: str = None,
                     tail_type: str = None,
                     remove_duplicates=True):
    def filter_func(metapath: Tuple[str]):
        condition = True

        if order and isinstance(order, int):
            condition = condition & (len(metapath[1::2]) == order)
        elif order and isinstance(order, Iterable):
            condition = condition & (len(metapath[1::2]) in order)

        if head_type:
            condition = condition & (metapath[0] == head_type)
        if tail_type:
            condition = condition & (metapath[-1] == tail_type)

        return condition

    return [m for m in sorted(OrderedDict.fromkeys(metapaths)) if filter_func(m)]


def get_edge_index_values(edge_index_tup: Union[tuple, torch.Tensor], filter_edge=False, threshold=0.5):
    if isinstance(edge_index_tup, tuple):
        edge_index, edge_values = edge_index_tup

        if filter_edge:
            mask = edge_values >= threshold
            # print("edge_values", edge_values.shape, edge_values[:5], "filtered", (~mask).sum().item())

            if mask.sum(0) == 0:
                mask[torch.argmax(edge_values)] = True

            edge_index = edge_index[:, mask]
            edge_values = edge_values[mask]

    elif isinstance(edge_index_tup, torch.Tensor) and edge_index_tup.size(1) > 0:
        edge_index = edge_index_tup
        edge_values = torch.ones(edge_index_tup.size(1), dtype=torch.float64, device=edge_index_tup.device)
    else:
        return None, None

    if edge_values.dtype != torch.float:
        edge_values = edge_values.to(torch.float)

    return edge_index, edge_values


def join_edge_indexes(edge_index_dict_A, edge_index_dict_B, global_node_idx, metapaths=None, edge_sampling=False):
    output_edge_index = {}
    for metapath_a, edge_index_a in edge_index_dict_A.items():
        if is_negative(metapath_a): continue
        edge_index_a, values_a = get_edge_index_values(edge_index_a, filter_edge=False)
        if edge_index_a is None: continue

        for metapath_b, edge_index_b in edge_index_dict_B.items():
            if metapath_a[-1] != metapath_b[0] or is_negative(metapath_b): continue

            new_metapath = metapath_a + metapath_b[1:]
            if metapaths and new_metapath not in metapaths: continue

            edge_index_b, values_b = get_edge_index_values(edge_index_b, filter_edge=False)
            if edge_index_b is None: continue

            try:
                new_edge_index, new_values = adamic_adar(indexA=edge_index_a, valueA=values_a,
                                                         indexB=edge_index_b, valueB=values_b,
                                                         m=global_node_idx[metapath_a[0]].size(0),
                                                         k=global_node_idx[metapath_b[0]].size(0),
                                                         n=global_node_idx[metapath_b[-1]].size(0),
                                                         coalesced=True,
                                                         sampling=edge_sampling)
                if new_edge_index.size(1) == 0: continue
                output_edge_index[new_metapath] = (new_edge_index, new_values)

            except Exception as e:
                print(f"{e} \n {metapath_a}: {edge_index_a.size(1)}, {metapath_b}: {edge_index_b.size(1)}")
                print("\t", {"m": global_node_idx[metapath_a[0]].size(0),
                             "k": global_node_idx[metapath_a[-1]].size(0),
                             "n": global_node_idx[metapath_b[-1]].size(0), })
                continue

        if metapaths and metapath_a in metapaths:
            # In the current LATTE layer that calls this method, a metapath is repeated (i.e. not higher-order), so we return the edges to it again.
            output_edge_index[metapath_a] = (edge_index_a, values_a)

    return output_edge_index
