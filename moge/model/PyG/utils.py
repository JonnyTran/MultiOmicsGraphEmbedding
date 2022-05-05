from collections import OrderedDict
from typing import Union, Tuple, Iterable, List, Dict, Optional

import pandas as pd
import torch
from torch import Tensor
from torch_sparse import SparseTensor, spspmm


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


def join_metapaths(metapath_A, metapath_B):
    output_metapaths = []

    for relation_b in metapath_B:
        for relation_a in metapath_A:
            if relation_a[-1] == relation_b[0]:
                new_relation = relation_a + relation_b[1:]
                output_metapaths.append(new_relation)
    return output_metapaths


def filter_metapaths(metapaths: List[Tuple[str]],
                     order: Union[int, List[int]] = None,
                     head_type: Union[str, List[str]] = None,
                     tail_type: Union[str, List[str]] = None):
    def filter_func(metapath: Tuple[str]):
        condition = True

        if order is not None and isinstance(order, int):
            condition = condition & (len(metapath[1::2]) == order)
        elif order is not None and isinstance(order, Iterable):
            condition = condition & (len(metapath[1::2]) in order)

        if head_type:
            condition = condition & (metapath[0] in head_type)
        if tail_type:
            condition = condition & (metapath[-1] in tail_type)

        return condition

    return [m for m in sorted(OrderedDict.fromkeys(metapaths)) if filter_func(m)]


def get_edge_index_values(edge_index_tup: Union[Tuple[Tensor, Tensor], Tensor],
                          filter_edge=False, threshold=0.5):
    if isinstance(edge_index_tup, tuple):
        edge_index, edge_values = edge_index_tup

        if filter_edge and threshold > 0.0:
            mask = edge_values >= threshold
            if mask.dim() > 1:
                mask = mask.any(dim=1)

            edge_index = edge_index[:, mask]
            edge_values = edge_values[mask]

    elif isinstance(edge_index_tup, Tensor) and edge_index_tup.size(1) > 0:
        edge_index = edge_index_tup
        edge_values = None
        # edge_values = torch.ones(edge_index_tup.size(1), dtype=torch.float, device=edge_index_tup.device)
    else:
        return None, None

    # if edge_values.dtype != torch.float:
    #     edge_values = edge_values.to(torch.float)

    return edge_index, edge_values


def join_edge_indexes(edge_index_dict_A: Dict[Tuple[str, str, str], Union[Tensor, Tuple[Tensor, Tensor]]],
                      edge_index_dict_B: Dict[Tuple[str, str, str], Union[Tensor, Tuple[Tensor, Tensor]]],
                      sizes: Union[Dict[str, int], List[Dict[str, Tuple[int]]]],
                      layer: Optional[int] = None,
                      metapaths: List[Tuple[str, str, str]] = None,
                      edge_threshold: Optional[float] = None,
                      edge_sampling: bool = False) -> Dict[Tuple[str, str, str], Tuple[Tensor, Tensor]]:
    """
    Return a cartesian product from two set of adjacency matrices, such that the output adjacency matrices are
    relation-matching.
    """
    output_edge_index = {}
    if not edge_index_dict_A or not edge_index_dict_B:
        return {}

    for metapath_b, edge_index_b in edge_index_dict_B.items():
        edge_index_b, values_b = get_edge_index_values(edge_index_b, filter_edge=False)
        if edge_index_b is None or edge_index_b.size(1) < 1: continue

        # In the current LATTE layer that calls this method, a metapath is not higher-order
        if metapaths is not None and metapath_b in metapaths:
            output_edge_index[metapath_b] = (edge_index_b, values_b)

        for metapath_a, edge_index_a in edge_index_dict_A.items():
            if metapath_a[-1] != metapath_b[0]: continue

            new_metapath = metapath_a + metapath_b[1:]
            if metapaths is not None and new_metapath not in metapaths:
                continue

            edge_index_a, values_a = get_edge_index_values(edge_index_a,
                                                           filter_edge=True if edge_threshold else False,
                                                           threshold=edge_threshold)
            if edge_index_a is None or edge_index_a.size(1) < 1 or is_negative(metapath_a): continue

            head, middle, tail = metapath_a[0], metapath_a[-1], metapath_b[-1]
            a_order = len(metapath_a[1::2])

            if isinstance(sizes, list):
                m = sizes[layer - a_order][head][0]
                k = sizes[layer - a_order][middle][1]
                n = sizes[layer][tail][1]
            elif isinstance(sizes, dict):
                m = sizes[head]
                k = sizes[middle]
                n = sizes[tail]

            # try:
            if values_a is None or values_b is None:
                # if values_a.dim() > 1 and values_a.size(1) > 1:
                # new_values = []
                # for d in range(values_a.size(1)):
                #     new_edge_index, values = spspmm(indexA=edge_index_a, valueA=values_a[:, d],
                #                                     indexB=edge_index_b, valueB=values_b[:, d],
                #                                     m=m, k=k, n=n,
                #                                     # sampling=edge_sampling,
                #                                     coalesced=True)
                #     new_values.append(values)
                # new_values = torch.stack(new_values, dim=1)

                new_edge_index, new_values = spspmm(indexA=edge_index_a, valueA=None,
                                                    indexB=edge_index_b, valueB=None,
                                                    m=m, k=k, n=n,
                                                    # sampling=edge_sampling,
                                                    coalesced=True)

            else:
                if values_a.dim() > 1 and values_a.size(1) == 1:
                    values_a = values_a.squeeze(-1)
                if values_b.dim() > 1 and values_b.size(1) == 1:
                    values_b = values_b.squeeze(-1)

                new_edge_index, new_values = spspmm(indexA=edge_index_a, valueA=values_a,
                                                    indexB=edge_index_b, valueB=values_b,
                                                    m=m, k=k, n=n,
                                                    # sampling=edge_sampling,
                                                    coalesced=True)

            if new_edge_index.size(1) == 0: continue
            output_edge_index[new_metapath] = (new_edge_index, new_values)

            # except Exception as e:
            #     print(f"{e} \n {metapath_a}: {edge_index_a.max(1).values, values_a.shape if values_a is not None else values_a}, "
            #           f"{metapath_b}: {edge_index_b.max(1).values, values_b.shape if values_b is not None else values_a}")
            #     print("sizes: ", {"m": m, "k": k, "n": n, })
            #     # raise e
            #     continue

    return output_edge_index


def adamic_adar(indexA, valueA, indexB, valueB, m, k, n, coalesced=True, sampling=False):
    assert valueA.dim() == 1
    assert valueB.dim() == 1

    A = SparseTensor(row=indexA[0], col=indexA[1], value=valueA,
                     sparse_sizes=(m, k), is_sorted=not coalesced)
    B = SparseTensor(row=indexB[0], col=indexB[1], value=valueB,
                     sparse_sizes=(k, n), is_sorted=not coalesced)

    deg_A = A.sum(0)
    deg_B = B.sum(1)
    deg_normalized = 1.0 / (deg_A + deg_B)

    D = SparseTensor(row=torch.arange(deg_normalized.size(0), device=valueA.device),
                     col=torch.arange(deg_normalized.size(0), device=valueA.device),
                     value=deg_normalized.type_as(valueA),
                     sparse_sizes=(deg_normalized.size(0), deg_normalized.size(0)), is_sorted=True)

    # print("A", A.sizes(), "D", D.sizes(), "B", B.sizes(), )
    out = A @ D @ B
    row, col, values = out.coo()

    num_samples = min(int(valueA.numel()), int(valueB.numel()), values.numel())
    if sampling and values.numel() > num_samples:
        idx = torch.multinomial(values, num_samples=num_samples, replacement=False)
        row, col, values = row[idx], col[idx], values[idx]

    return torch.stack([row, col], dim=0), values


def edge_index2matrix(edge_index_dict, metapath, describe=True):
    edge_idx, edge_values = get_edge_index_values(edge_index_dict[metapath])
    if edge_values.dim() > 1:
        edge_values = edge_values.mean(1)

    st = SparseTensor.from_edge_index(edge_index=edge_idx,
                                      edge_attr=edge_values)

    if describe:
        print(f"sizes {st.sizes()}, max-min: {(st.storage.value().max(), st.storage.value().min())}")
        axis = 0
        row_sum = st.sum(axis)  # [st.sum(axis).nonzero()]
        if row_sum.dim() > 2:
            row_sum = row_sum.max(-1).values
        print(f"sum on {axis}-axis", pd.Series(row_sum.detach().numpy().flatten()).describe().to_dict())

    mtx = st.detach().to_dense().numpy()
    return mtx


def spspmm_outer_norm(indexA, valueA, indexB, valueB, m, k, n, coalesced=True, sampling=False):
    A = SparseTensor(row=indexA[0], col=indexA[1], value=valueA,
                     sparse_sizes=(m, k), is_sorted=not coalesced)
    B = SparseTensor(row=indexB[0], col=indexB[1], value=valueB,
                     sparse_sizes=(k, n), is_sorted=not coalesced)

    deg_A = torch.pow(A.sum(1), -1)
    deg_B = torch.pow(B.sum(1), -1)

    diag_A = SparseTensor(row=torch.arange(deg_A.size(0), device=deg_A.device),
                          col=torch.arange(deg_A.size(0), device=deg_A.device),
                          value=deg_A,
                          sparse_sizes=(deg_A.size(0), deg_A.size(0)), is_sorted=True)

    diag_B = SparseTensor(row=torch.arange(deg_B.size(0), device=deg_B.device),
                          col=torch.arange(deg_B.size(0), device=deg_B.device),
                          value=deg_B,
                          sparse_sizes=(deg_B.size(0), deg_B.size(0)), is_sorted=True)

    out = A @ B
    row, col, values = out.coo()

    return torch.stack([row, col], dim=0), values
