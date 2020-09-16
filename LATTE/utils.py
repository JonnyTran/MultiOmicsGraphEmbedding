import itertools

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
        return {k: tensor_sizes(v) for k, v in input.items()}
    elif isinstance(input, tuple):
        return tuple(tensor_sizes(v) for v in input)
    elif isinstance(input, list):
        return [tensor_sizes(v) for v in input]
    else:
        return input.shape


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
    if not isinstance(input, torch.Tensor):
        input = torch.tensor(input)

    if dtype:
        input = input.type(dtype)
    if half:
        input = input.half()
    if device:
        input = input.to(device)

    return input
