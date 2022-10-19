from collections import OrderedDict
from collections.abc import MutableMapping
from typing import Dict, Tuple, Optional, Union, List

import dgl
import numpy as np
import pandas as pd
import torch
from dgl._deprecate.graph import DGLGraph
from dgl.heterograph import DGLBlock, DGLHeteroGraph
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_sparse import SparseTensor


def to_device(obj: Union[Tensor, Dict, List, Tuple], device: str):
    if torch.is_tensor(obj):
        return obj.to(device)

    elif isinstance(obj, SparseTensor):
        return obj.to(device)

    elif isinstance(obj, torch.nn.Module):
        return obj.to(device)

    elif isinstance(obj, (dgl.DGLHeteroGraph, dgl.DGLGraph, DGLBlock)):
        return obj.to(device)

    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = to_device(v, device)
        return res

    elif isinstance(obj, (list, tuple)):
        res = []
        for v in obj:
            res.append(to_device(v, device))
        return res

    elif obj is None or isinstance(obj, (int, float, str, np.ndarray, pd.DataFrame, type(None))):
        return obj

    else:
        raise TypeError("Invalid type for move_to", type(obj), "\n", obj)


def tensor_sizes(input=None, **kwargs) -> ...:
    """
    A very useful method to inspect the sizes of tensors in object containing Tensors
    Args:
        input ():
        **kwargs ():

    Returns:

    """
    if kwargs:
        return tensor_sizes(kwargs)

    if isinstance(input, (dict, MutableMapping)):
        return {key: tensor_sizes(v) \
                for key, v in input.items()}

    elif isinstance(input, tuple):
        return tuple(tensor_sizes(v) for v in input)
    elif isinstance(input, list):
        if len(input) and isinstance(input[0], str):
            return len(input)
        return [tensor_sizes(v) for v in input]
    elif isinstance(input, set):
        if len(input) and isinstance(list(input)[0], str):
            return len(input)
        return {tensor_sizes(v) for v in input}

    elif isinstance(input, (DGLGraph, DGLBlock, DGLHeteroGraph)):
        return {ntype: (input.num_src_nodes(ntype), input.num_dst_nodes(ntype)) \
            if isinstance(input, DGLBlock) else (input.num_nodes(ntype),) for ntype in input.ntypes} | \
               {etype: input.num_edges(etype=etype) for etype in input.canonical_etypes if input.num_edges(etype=etype)}

    elif isinstance(input, HeteroData):
        return {ntype: (input[ntype].num_nodes,) for ntype in input.node_types} | \
               {etype: input[etype].num_edges for etype in input.edge_types if input[etype].num_edges}

    elif isinstance(input, SparseTensor):
        return input.storage.sparse_sizes()

    else:
        if input is not None and hasattr(input, "shape"):
            if isinstance(input, Tensor) and input.dim() == 0:
                return input.item()

            return list(input.shape)
        else:
            return input


def activation(y_pred: Tensor, loss_type: str):
    # Apply softmax/sigmoid activation if needed
    if "LOGITS" in loss_type or "FOCAL" in loss_type:
        if "SOFTMAX" in loss_type:
            y_pred = torch.softmax(y_pred, dim=1)
        else:
            y_pred = torch.sigmoid(y_pred)

    elif "NEGATIVE_LOG_LIKELIHOOD" == loss_type or "SOFTMAX_CROSS_ENTROPY" in loss_type:
        y_pred = torch.softmax(y_pred, dim=1)

    return y_pred


def filter_samples(Y_hat: Tensor, Y: Tensor, weights: Tensor):
    if weights is None or not hasattr(weights, 'shape') or weights.shape == None or weights.numel() == 0:
        return Y_hat, Y

    if not isinstance(weights, Tensor):
        weights = torch.tensor(weights)

    idx = torch.nonzero(weights).view(-1)

    if Y.dim() > 1:
        Y = Y[idx, :]
    else:
        Y = Y[idx]

    if Y_hat.dim() > 1:
        Y_hat = Y_hat[idx, :]
    else:
        Y_hat = Y_hat[idx]

    return Y_hat, Y


def filter_samples_weights(y_pred: Tensor, y_true: Tensor, weights: Optional[Tensor] = None, return_index=False):
    if weights is None or not isinstance(weights, (Tensor, np.ndarray)) or weights.shape == None:
        return y_pred, y_true, None
    else:
        assert weights.dim() == 1, f"`weights` must be a 1D vector of size n_samples. Current size: {weights.shape}"

    if isinstance(weights, Tensor) and weights.numel():
        idx = torch.nonzero(weights).ravel()
    else:
        idx = torch.tensor(np.nonzero(weights)[0])

    if return_index:
        return idx

    if y_true.dim() > 1:
        y_true_out = y_true[idx, :]
    else:
        y_true_out = y_true[idx]

    if y_pred.dim() > 1:
        y_pred_out = y_pred[idx, :]
    else:
        y_pred_out = y_pred[idx]

    return y_pred_out, y_true_out, weights[idx]


def concat_dict_batch(batch_size: Dict[str, int], y_pred: Dict[str, Tensor], y_true: Dict[str, Tensor],
                      weights: Optional[Dict[str, Tensor]] = None) \
        -> Tuple[Tensor, Tensor, Tensor]:
    # Filter out node types which have no labels and ensure same order of ntypes
    batch_size = OrderedDict({ntype: size for ntype, size in batch_size.items()})

    if isinstance(y_true, dict):
        y_trues = [y_true[ntype][:size] for ntype, size in batch_size.items()]
        y_true_concat = torch.cat(y_trues, dim=0) if len(y_trues) > 1 else y_trues[0]
    elif isinstance(y_true, Tensor):
        size = list(batch_size.values())[0]
        y_true_concat = y_true[:size]
    else:
        raise Exception(f"Check `y_true` type: {y_true}")

    if isinstance(y_pred, dict):
        y_preds = [y_pred[ntype][:size] for ntype, size in batch_size.items()]
        y_pred_concat = torch.cat(y_preds, dim=0) if len(y_preds) > 1 else y_preds[0]
    elif isinstance(y_pred, Tensor):
        size = list(batch_size.values())[0]
        y_pred_concat = y_pred[:size]
    else:
        raise Exception()

    if isinstance(weights, dict):
        weights = torch.cat([weights[ntype][:size] for ntype, size in batch_size.items()], dim=0)
    elif isinstance(weights, (np.ndarray, pd.Series, pd.DataFrame, Tensor)):
        size = list(batch_size.values())[0]
        weights = weights[:size]
    else:
        raise Exception()

    return y_pred_concat, y_true_concat, weights


def stack_tensor_dicts(y_pred: Dict[str, Tensor], y_true: Dict[str, Tensor],
                       weights: Optional[Dict[str, Tensor]] = None) \
        -> Tuple[Tensor, Tensor, Tensor]:
    """
    Returns y_pred, y_true, weights as Tensors and ensure they all have the same batch_size in dim 0.

    Args:
        y_pred ():
        y_true ():
        weights ():

    Returns:

    """
    if isinstance(y_true, dict) and isinstance(y_pred, dict):
        ntypes = list(y_pred.keys())
        # Filter node types which have no data
        ntypes = [ntype for ntype in ntypes if y_true[ntype].sum() > 0]

        y_true = torch.cat([y_true[ntype] for ntype in ntypes], dim=0)
        y_pred = torch.cat([y_pred[ntype] for ntype in ntypes], dim=0)
        if isinstance(weights, dict):
            weights = torch.cat([weights[ntype] for ntype in ntypes], dim=0)

    elif isinstance(y_true, dict) and isinstance(y_pred, Tensor):
        head_node_type = list({ntype for ntype, label in y_true.items() if label.numel() > 0}).pop()
        y_true = y_true[head_node_type]
        if isinstance(weights, dict):
            weights = weights[head_node_type]

    elif isinstance(y_true, Tensor) and isinstance(y_pred, dict):
        head_node_type = list(y_pred.keys()).pop()
        y_pred = y_pred[head_node_type]
        if isinstance(weights, dict):
            weights = weights[head_node_type]

    return y_pred, y_true, weights

def edge_index_sizes(edge_index_dict):
    output = {}
    for m, edge_index in edge_index_dict.items():
        if isinstance(edge_index, tuple):
            edge_index, values_a = edge_index
        else:
            values_a = None

        if edge_index.size(1) == 0:
            output[m] = None
        else:
            sizes = edge_index.max(1).values.data.tolist()
            if values_a is not None:
                output[m] = (sizes, values_a.shape)
            else:
                output[m] = sizes

    return output


def preprocess_input(input, device, dtype=None, half=False):
    if isinstance(input, dict):
        input = {k: preprocess_input(v, device, dtype, half) for k, v in input.items()}
    elif isinstance(input, tuple):
        input = tuple(preprocess_input(v, device, dtype, half) for v in input)
    elif isinstance(input, list):
        input = [preprocess_input(v, device, dtype, half) for v in input]
    else:
        input = process_tensor(input, device=device, dtype=dtype, half=half)
    return input


def process_tensor(input, device=None, dtype=None, half=False):
    if input is None:
        return input

    if not isinstance(input, Tensor):
        input = torch.tensor(input)

    if dtype:
        input = input.type(dtype)
    if half:
        input = input.half()
    if device:
        input = input.to(device)

    return input


def pad_tensors(sequences):
    num = len(sequences)
    max_len = max([s.size(-1) for s in sequences])
    out_dims = (num, 2, max_len)
    out_tensor = sequences[0].dataset.new(*out_dims).fill_(0)
    #     mask = sequences[0].data.new(*out_dims).fill_(0)
    for i, tensor in enumerate(sequences):
        length = tensor.size(-1)
        out_tensor[i, :, :length] = tensor
    #         mask[i, :length] = 1
    return out_tensor


def collate_fn(batch):
    protein_seqs_all, physical, genetic, correlation, y_all, idx_all = [], [], [], [], [], []
    for X, y, idx in batch:
        protein_seqs_all.append(torch.tensor(X["Protein_seqs"]))
        physical.append(torch.tensor(X["Protein-Protein-physical"]))
        genetic.append(torch.tensor(X["Protein-Protein-genetic"]))
        correlation.append(torch.tensor(X["Protein-Protein-correlation"]))
        y_all.append(torch.tensor(y))
        idx_all.append(torch.tensor(idx))

    X_all = {"Protein_seqs": torch.cat(protein_seqs_all, dim=0),
             "Protein-Protein-physical": pad_tensors(physical),
             "Protein-Protein-genetic": pad_tensors(genetic),
             "Protein-Protein-correlation": pad_tensors(correlation), }
    return X_all, torch.cat(y_all, dim=0), torch.cat(idx_all, dim=0)


def get_multiplex_collate_fn(node_types, layers):
    def multiplex_collate_fn(batch):
        y_all, idx_all = [], []
        node_type_concat = dict()
        layer_concat = dict()
        for node_type in node_types:
            node_type_concat[node_type] = []
        for layer in layers:
            layer_concat[layer] = []

        for X, y, idx in batch:
            for node_type in node_types:
                node_type_concat[node_type].append(torch.tensor(X[node_type]))
            for layer in layers:
                layer_concat[layer].append(torch.tensor(X[layer]))
            y_all.append(torch.tensor(y))
            idx_all.append(torch.tensor(idx))

        X_all = {}
        for node_type in node_types:
            X_all[node_type] = torch.cat(node_type_concat[node_type])
        for layer in layers:
            X_all[layer] = pad_tensors(layer_concat[layer])

        return X_all, torch.cat(y_all), torch.cat(idx_all)

    return multiplex_collate_fn
