import logging
import traceback
from collections import OrderedDict
from typing import Union, Tuple, List, Dict, Optional, Set

import pandas as pd
import torch
from logzero import logger
from moge.dataset.utils import is_negative
from torch import Tensor
from torch_sparse import SparseTensor, spspmm


def num_edges(edge_index_dict: Dict[Tuple[str, str, str], Union[Tensor, Tuple[Tensor, Tensor]]]):
    return sum(edge_index.size(1) if isinstance(edge_index, Tensor) else edge_index[0].size(1) \
               for m, edge_index in edge_index_dict.items())


def max_num_hops(metapaths: List[Tuple[str, str, str]]):
    return max(len(metapath[1::2]) for metapath in metapaths)


def convert_to_nx_edgelist(edge_index_dict: Dict[Tuple[str, str, str], Tensor], node_names: Dict[str, pd.Index],
                           global_node_idx: Dict[str, Tensor] = None) -> Dict[str, List[Tuple[str, str]]]:
    edge_list = {}
    for metapath, edge_index in edge_index_dict.items():
        head_type, etype, tail_type = metapath
        if "rev_" in metapath:
            continue

        head_nodes = edge_index[0] if not global_node_idx else global_node_idx[head_type][edge_index[0]]
        tail_nodes = edge_index[1] if not global_node_idx else global_node_idx[tail_type][edge_index[1]]

        if isinstance(head_nodes, Tensor):
            head_nodes = head_nodes.detach().numpy()
        if isinstance(tail_nodes, Tensor):
            tail_nodes = tail_nodes.detach().numpy()

        head_nodes = head_type + "-" + node_names[head_type][head_nodes]
        tail_nodes = tail_type + "-" + node_names[tail_type][tail_nodes]
        edge_list[etype] = [(u, v) for u, v in zip(head_nodes, tail_nodes)]

    return edge_list


def join_metapaths(metapaths_A: List[Tuple[str, str, str]], metapaths_B: List[Tuple[str, str, str]],
                   tail_types: List[str] = None, skip_undirected=False, return_dict=False, ) \
        -> Dict[str, List[Tuple[str, str, str]]]:
    """

    Args:
        metapaths_A ():
        metapaths_B ():
        tail_types ():
        skip_undirected ():
        return_dict (): If True, return in from of Dict of new etype as keys and chain of metapaths as values

    Returns:

    """
    if return_dict:
        output_metapaths = {}
    else:
        output_metapaths = []

    for metapath_b in metapaths_B:
        if tail_types and metapath_b[-1] not in tail_types: continue
        for metapath_a in metapaths_A:
            if metapath_a[-1] != metapath_b[0]: continue
            if skip_undirected and metapath_a[0] == metapath_a[-1] and metapath_b[0] == metapath_b[-1]: continue

            if return_dict:
                new_metapath = ".".join([metapath_a[1], metapath_b[1]])
                output_metapaths[new_metapath] = [metapath_a, metapath_b]
            else:
                output_metapaths.append(metapath_a + metapath_b[1:])

    return output_metapaths


def filter_metapaths(metapaths: List[Tuple[str, str, str]], order: Union[int, List[int]] = None,
                     head_type: Union[str, List[str]] = None, tail_type: Union[str, List[str]] = None) \
        -> List[Tuple[str, str, str]]:
    def filter_func(metapath: Tuple[str]):
        condition = True

        num_hops = len(metapath[1::2])

        if isinstance(order, int):
            condition = condition & (num_hops == order)
        elif isinstance(order, (list, set, tuple)):
            condition = condition & (num_hops in order)

        if head_type:
            condition = condition & (metapath[0] in head_type)
        if tail_type:
            condition = condition & (metapath[-1] in tail_type)

        return condition

    return [m for m in sorted(OrderedDict.fromkeys(metapaths)) if filter_func(m)]


def get_edge_index_values(edge_index_tup: Union[Tuple[Tensor, Tensor], Tensor],
                          filter_edge=False, threshold=0.5, drop_edge_values=False):
    if isinstance(edge_index_tup, tuple):
        edge_index, edge_values = edge_index_tup

        if filter_edge and isinstance(edge_values, Tensor) and threshold > 0.0:
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
    if drop_edge_values:
        edge_values = None

    return edge_index, edge_values


def join_edge_indexes(edge_index_dict_A: Dict[Tuple[str, str, str], Union[Tensor, Tuple[Tensor, Tensor]]],
                      edge_index_dict_B: Dict[Tuple[str, str, str], Union[Tensor, Tuple[Tensor, Tensor]]],
                      sizes: Union[Dict[str, int], List[Dict[str, Tuple[int]]]],
                      layer: Optional[int] = None,
                      filter_metapaths: Union[List[Tuple[str, str, str]], Set[Tuple[str, str, str]]] = None,
                      edge_threshold: Optional[float] = None,
                      device=None) \
        -> Dict[Tuple[str, str, str], Tuple[Tensor, Tensor]]:
    """
    Return a cartesian product from two set of adjacency matrices, such that each metapath_A have same tail node type
     as metapath_B's head node type.
    """
    joined_edge_index = {}
    if not edge_index_dict_A or not edge_index_dict_B:
        return {}

    for metapath_b, edge_index_b in edge_index_dict_B.items():
        edge_index_b, values_b = get_edge_index_values(edge_index_b, filter_edge=False, drop_edge_values=True)
        if edge_index_b is None or edge_index_b.size(1) < 1: continue

        # In the current LATTE layer that calls this method, a metapath is not higher-order
        if filter_metapaths is not None and metapath_b in filter_metapaths:
            joined_edge_index[metapath_b] = (edge_index_b, values_b)

        for metapath_a, edge_index_a in edge_index_dict_A.items():
            if metapath_a[-1] != metapath_b[0]: continue

            new_metapath = metapath_a + metapath_b[1:]
            if filter_metapaths is not None and new_metapath not in filter_metapaths: continue

            edge_index_a, values_a = get_edge_index_values(edge_index_a,
                                                           filter_edge=True if edge_threshold else False,
                                                           threshold=edge_threshold, drop_edge_values=True)
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

            orig_device = edge_index_a.device
            if device is None:
                device = orig_device

            try:
                if not isinstance(values_a, Tensor):
                    values_a = None
                elif values_a.dim() > 1 and values_a.size(1) == 1:
                    values_a = values_a.squeeze(-1).to(device)

                if not isinstance(values_b, Tensor):
                    values_b = None
                elif values_b.dim() > 1 and values_b.size(1) == 1:
                    values_b = values_b.squeeze(-1).to(device)

                # elif values_a.dim() > 1 and values_a.size(1) > 1:
                # new_values = []
                # for d in range(values_a.size(1)):
                #     new_edge_index, values = spspmm(indexA=edge_index_a, valueA=values_a[:, d],
                #                                     indexB=edge_index_b, valueB=values_b[:, d],
                #                                     m=m, k=k, n=n,
                #                                     coalesced=True)
                #     new_values.append(values)
                # new_values = torch.stack(new_values, dim=1)

                new_edge_index, new_values = spspmm(indexA=edge_index_a.to(device), valueA=values_a,
                                                    indexB=edge_index_b.to(device), valueB=values_b,
                                                    m=m, k=k, n=n)

                if new_edge_index.size(1):
                    joined_edge_index[new_metapath] = (
                        new_edge_index.to(orig_device),
                        new_values.to(orig_device) if isinstance(new_values, Tensor) else None)
                else:
                    joined_edge_index[new_metapath] = None

            except RuntimeError as re:
                logging.error(re)
                traceback.print_exc()
                # When CUDA out of memory, perform spspmm in cpu
                new_edge_index, new_values = spspmm(indexA=edge_index_a.cpu(),
                                                    valueA=values_a.cpu() if isinstance(values_a, Tensor) else None,
                                                    indexB=edge_index_b.cpu(),
                                                    valueB=values_b.cpu() if isinstance(values_b, Tensor) else None,
                                                    m=m, k=k, n=n)

                if new_edge_index.size(1):
                    joined_edge_index[new_metapath] = (
                        new_edge_index.to(orig_device),
                        new_values.to(orig_device) if isinstance(new_values, Tensor) else None)
                else:
                    joined_edge_index[new_metapath] = None

            except Exception as e:
                logger.error(f"\n{e} "
                             f"\n{new_metapath} sizes: {dict(m=m, k=k, n=n)}"
                             f"\n {metapath_a}: "
                             f"{edge_index_a.max(1).values.tolist(), values_a.shape if values_a is not None else values_a}, "
                             f"\n {metapath_b}: "
                             f"{edge_index_b.max(1).values.tolist(), values_b.shape if values_b is not None else values_a}"
                             )
                joined_edge_index[new_metapath] = None

    return joined_edge_index


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


def get_edge_index_from_neg_batch(neg_batch: Dict[Tuple[str, str, str], Tensor],
                                  edge_pos: Dict[Tuple[str, str, str], Tensor],
                                  mode: str) -> Tuple[Dict[Tuple[str, str, str], Tensor], int]:
    """

    Args:
        neg_batch (Dict[Tuple[str, str, str], Tensor]): The head_batch or tail_batch, a dict with values
            of shape [num_pos_edges, num_neg_sampling].
        edge_pos (Dict[Tuple[str, str, str], Tensor]): An edge_index_dict with values of shape [2, num_pos_edges].
        mode (str): either "head_batch" or "tail_batch"

    Returns:
        edge_index_dict, neg_samp_size
    """
    edge_index_dict = {}

    for metapath, edge_index in edge_pos.items():
        num_edges, neg_samp_size = neg_batch[metapath].shape

        if mode == "head_batch":
            head_nodes = neg_batch[metapath].view(-1)
            tail_nodes = edge_pos[metapath][1].repeat_interleave(neg_samp_size)
            edge_index_dict[metapath] = torch.stack([head_nodes, tail_nodes], dim=0)

        elif mode == "tail_batch":
            head_nodes = edge_pos[metapath][0].repeat_interleave(neg_samp_size)
            tail_nodes = neg_batch[metapath].view(-1)
            edge_index_dict[metapath] = torch.stack([head_nodes, tail_nodes], dim=0)

    return edge_index_dict, neg_samp_size


def batch2global_edge_index(batch_edge_index: Tensor, metapath: Tuple[str, str, str],
                            global_node_index: Dict[str, Tensor], ntype_mapping: Dict[str, str] = None) -> Tensor:
    head_type, tail_type = metapath[0], metapath[-1]
    if ntype_mapping is not None:
        head_type = ntype_mapping[head_type] if head_type not in global_node_index else head_type
        tail_type = ntype_mapping[tail_type] if tail_type not in global_node_index else tail_type

    sources = global_node_index[head_type][batch_edge_index[0]]
    targets = global_node_index[tail_type][batch_edge_index[-1]]

    global_edge_index = torch.stack([sources, targets], dim=1).t()
    return global_edge_index
