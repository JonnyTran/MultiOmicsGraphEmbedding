from collections import Iterable

import numpy as np
import torch


def filter_samples(Y_hat: torch.Tensor, Y: torch.Tensor, weights: torch.Tensor, max_mode=False):
    if weights is None or weights.shape == None or weights.numel() == 0:
        return Y_hat, Y

    if not isinstance(weights, torch.Tensor):
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


def filter_samples_weights(Y_hat: torch.Tensor, Y: torch.Tensor, weights):
    if weights is None or weights.shape == None:
        return Y_hat, Y, None

    if isinstance(weights, torch.Tensor):
        idx = torch.nonzero(weights).view(-1)
    else:
        idx = torch.tensor(np.nonzero(weights)[0])

    if Y.dim() > 1:
        Y = Y[idx, :]
    else:
        Y = Y[idx]

    if Y_hat.dim() > 1:
        Y_hat = Y_hat[idx, :]
    else:
        Y_hat = Y_hat[idx]

    return Y_hat, Y, weights[idx]


def tensor_sizes(input):
    if isinstance(input, dict):
        return {metapath if not isinstance(metapath, tuple) else \
                    ".".join([type[0].upper() if i % 2 == 0 else type[0].lower() for i, type in
                              enumerate(metapath)]): tensor_sizes(v) \
                for metapath, v in input.items()}
    elif isinstance(input, tuple):
        return tuple(tensor_sizes(v) for v in input)
    elif isinstance(input, list):
        return [tensor_sizes(v) for v in input]
    else:
        if input is not None and hasattr(input, "shape"):
            if isinstance(input, torch.Tensor) and input.dim() == 0:
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

    if not isinstance(input, torch.Tensor):
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


def process_multi_ntypes(y_pred, y_true, weights):
    if isinstance(y_true, dict):
        ntypes = list(y_pred.keys())

        y_pred = torch.cat([y_pred[ntype] for ntype in ntypes], dim=0)
        y_true = torch.cat([y_true[ntype] for ntype in ntypes], dim=0)
        if isinstance(weights, dict):
            weights = torch.cat([weights[ntype] for ntype in ntypes], dim=0)

    return y_pred, y_true, weights
