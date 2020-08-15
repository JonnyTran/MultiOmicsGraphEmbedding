import random

import numpy as np
import torch


def negative_sample(edge_index, M: int, N: int, num_neg_samples: int):
    num_neg_samples = int(min(num_neg_samples,
                              M * N - edge_index.size(1)))
    rng = range(M * N)
    idx = (edge_index[0] * N + edge_index[1]).to('cpu')  # idx = N * i + j

    perm = torch.tensor(random.sample(rng, num_neg_samples))
    mask = torch.from_numpy(np.isin(perm, idx)).to(torch.bool)
    rest = mask.nonzero().view(-1)
    while rest.numel() > 0:  # pragma: no cover
        tmp = torch.tensor(random.sample(rng, rest.size(0)))
        mask = torch.from_numpy(np.isin(tmp, idx)).to(torch.bool)
        perm[rest] = tmp
        rest = rest[mask.nonzero().view(-1)]

    row = perm // N
    col = perm % N
    neg_edge_index = torch.stack([row, col], dim=0).long()

    return neg_edge_index.to(edge_index.device)

def negative_sample_head_tail(edge_index, M: int, N: int, num_neg_samples: int):
    num_neg_samples = int(min(num_neg_samples,
                              M * N - edge_index.size(1)))
