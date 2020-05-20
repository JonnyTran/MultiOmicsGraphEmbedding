import torch


def pad_tensors(sequences):
    num = len(sequences)
    max_len = max([s.size(-1) for s in sequences])
    out_dims = (num, 2, max_len)
    out_tensor = sequences[0].data.new(*out_dims).fill_(0)
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

    X_all = {"Protein_seqs": torch.cat(protein_seqs_all),
             "Protein-Protein-physical": pad_tensors(physical),
             "Protein-Protein-genetic": pad_tensors(genetic),
             "Protein-Protein-correlation": pad_tensors(correlation), }
    return X_all, torch.cat(y_all), torch.cat(idx_all)


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
