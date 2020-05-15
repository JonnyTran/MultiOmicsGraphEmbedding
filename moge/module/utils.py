import torch


def preprocess_input(X, cuda=True, half=False):
    if isinstance(X, dict):
        X = {k: _preprocess_input(v, cuda=cuda, half=half) for k, v in X.items()}
    else:
        X = _preprocess_input(X, cuda=cuda, half=half)

    return X


def _preprocess_input(X, cuda=True, half=False):
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X)

    if cuda:
        X = X.cuda()
    else:
        X = X.cpu()

    if half:
        X = X.half()

    return X
