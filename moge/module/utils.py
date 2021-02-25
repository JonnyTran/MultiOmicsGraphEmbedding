import numpy as np
import torch


def filter_samples(Y_hat: torch.Tensor, Y: torch.Tensor, weights):
    if weights is None:
        return Y_hat, Y

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

    return Y_hat, Y


def tensor_sizes(input):
    if isinstance(input, dict):
        return {k if not isinstance(k, tuple) else k[-2]: tensor_sizes(v) for k, v in input.items()}
    elif isinstance(input, tuple):
        return tuple(tensor_sizes(v) for v in input)
    elif isinstance(input, list):
        return [tensor_sizes(v) for v in input]
    else:
        return input.shape if input is not None and hasattr(input, "shape") else None


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