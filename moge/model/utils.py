from typing import Dict

import numpy as np
import torch
from torch import Tensor


def activation(y_pred, loss_type):
    # Apply softmax/sigmoid activation if needed
    if "LOGITS" in loss_type or "FOCAL" in loss_type:
        if "SOFTMAX" in loss_type:
            y_pred = torch.softmax(y_pred, dim=1)
        else:
            y_pred = torch.sigmoid(y_pred)
    elif "NEGATIVE_LOG_LIKELIHOOD" == loss_type or "SOFTMAX_CROSS_ENTROPY" in loss_type:
        y_pred = torch.softmax(y_pred, dim=1)
    return y_pred


def filter_samples(Y_hat: Tensor, Y: Tensor, weights: Tensor, max_mode=False):
    if weights is None or weights.shape == None or weights.numel() == 0:
        return Y_hat, Y

    if not isinstance(weights, Tensor):
        weights = torch.tensor(weights)

    if max_mode:
        idx = weights == weights.max()
    else:
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


def filter_samples_weights(Y_hat: Tensor, Y: Tensor, weights, return_index=False):
    if weights is None or \
            (isinstance(weights, (Tensor, np.ndarray)) and weights.shape == None):
        return Y_hat, Y, None

    if isinstance(weights, Tensor):
        idx = torch.nonzero(weights).view(-1)
    else:
        idx = torch.tensor(np.nonzero(weights)[0])

    if return_index:
        return idx

    if Y.dim() > 1:
        Y = Y[idx, :]
    else:
        Y = Y[idx]

    if Y_hat.dim() > 1:
        Y_hat = Y_hat[idx, :]
    else:
        Y_hat = Y_hat[idx]

    return Y_hat, Y, weights[idx]


def select_batch(batch_size: Dict[str, int], y_pred: Dict[str, Tensor], y_true: Dict[str, Tensor], weights=None):
    # Filter node types which have no data
    batch_size = {ntype: size for ntype, size in batch_size.items() if y_true[ntype].sum() > 0}

    if isinstance(y_true, dict):
        y_true = torch.cat([y_true[ntype][:size] for ntype, size in batch_size.items()], dim=0)
    if isinstance(y_pred, dict):
        y_pred = torch.cat([y_pred[ntype][:size] for ntype, size in batch_size.items()], dim=0)
    if isinstance(weights, dict):
        weights = torch.cat([weights[ntype][:size] for ntype, size in batch_size.items()], dim=0)

    return y_pred, y_true, weights


def process_tensor_dicts(y_pred: Dict[str, Tensor], y_true: Dict[str, Tensor], weights: Dict[str, Tensor] = None):
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


def tensor_sizes(input):
    if isinstance(input, dict):
        return {metapath if not isinstance(metapath, tuple) else \
                    ".".join(metapath): tensor_sizes(v) \
                for metapath, v in input.items()}
    elif isinstance(input, tuple):
        return tuple(tensor_sizes(v) for v in input)
    elif isinstance(input, list):
        return [tensor_sizes(v) for v in input]
    else:
        if input is not None and hasattr(input, "shape"):
            if isinstance(input, Tensor) and input.dim() == 0:
                return input.item()

            return list(input.shape)
        else:
            return input


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
