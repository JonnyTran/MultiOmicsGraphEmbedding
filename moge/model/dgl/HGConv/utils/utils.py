import numpy as np
import random
import torch
import torch.nn as nn
import dgl
from dgl.data.utils import load_graphs
from ogb.nodeproppred import Evaluator


# convert the inputs from cpu to gpu, accelerate the running speed
def convert_to_gpu(*data, device: str):
    res = []
    for item in data:
        item = item.to(device)
        res.append(item)
    if len(res) > 1:
        res = tuple(res)
    else:
        res = res[0]
    return res


def set_random_seed(seed: int = 0):
    """
    set random seed.
    :param seed: int, random seed to use
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    dgl.random.seed(0)


def load_model(model: nn.Module, model_path: str):
    """Load the model.
    :param model: model
    :param model_path: model path
    """
    print(f"load model {model_path}")
    model.load_state_dict(torch.load(model_path))


def get_n_params(model: nn.Module):
    """
    get parameter size of trainable parameters in model
    :param model: model
    :return: int
    """
    return sum(p.numel() for p in model.parameters())


def load_dataset(data_path: str, predict_category: str, data_split_idx_path: str = None):
    """
    load dataset
    :param data_path: data file path
    :param predict_category: predict node category
    :param data_split_idx_path: split index file path
    :return:
    """
    graph_list, labels = load_graphs(data_path)

    graph = graph_list[0]

    labels = labels[predict_category].squeeze(dim=-1)

    num_classes = len(labels.unique())

    split_idx = torch.load(data_split_idx_path)
    train_idx, valid_idx, test_idx = split_idx['train'][predict_category], split_idx['valid'][predict_category], \
                                     split_idx['test'][predict_category]

    return graph, labels, num_classes, train_idx, valid_idx, test_idx


def get_node_data_loader(node_neighbors_min_num: int, n_layers: int,
                         graph: dgl.DGLGraph, batch_size: int, sampled_node_type: str,
                         train_idx: torch.Tensor, valid_idx: torch.Tensor, test_idx: torch.Tensor,
                         shuffle: bool = True, drop_last: bool = False, num_workers: int = 4):
    """
    get graph node data loader, including train_loader, val_loader and test_loader
    :return:
    """
    # list of neighbors to sample per edge type for each GNN layer
    sample_nodes_num = []
    for layer in range(n_layers):
        sample_nodes_num.append({etype: node_neighbors_min_num + layer for etype in graph.canonical_etypes})

    # neighbor sampler
    sampler = dgl.dataloading.MultiLayerNeighborSampler(sample_nodes_num)

    train_loader = dgl.dataloading.NodeDataLoader(
        graph, {sampled_node_type: train_idx}, sampler,
        batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    val_loader = dgl.dataloading.NodeDataLoader(
        graph, {sampled_node_type: valid_idx}, sampler,
        batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    test_loader = dgl.dataloading.NodeDataLoader(
        graph, {sampled_node_type: test_idx}, sampler,
        batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return train_loader, val_loader, test_loader


def get_optimizer_and_lr_scheduler(model: nn.Module, optimizer_name: str, learning_rate: float, weight_deacy: float,
                                   steps_per_epoch: int, epochs: int):
    """
    get optimizer and lr scheduler
    :param model:
    :param optimizer_name:
    :param learning_rate:
    :param weight_deacy:
    :param steps_per_epoch:
    :param epochs:
    :return:
    """
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_deacy)
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_deacy)
    else:
        raise ValueError(f"wrong value for optimizer {optimizer_name}!")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps_per_epoch * epochs,
                                                           eta_min=learning_rate / 100)

    return optimizer, scheduler


def get_ogb_evaluator(predicts: torch.Tensor, labels: torch.Tensor):
    """
    get evaluation metrics for ogb, calculate accuracy
    :param predicts: Tensor, shape (N, )
    :param labels: Tensor, shape (N, )
    :return:
    """
    evaluator = Evaluator(name='ogbn-mag')

    predictions = predicts.cpu().numpy()
    labels = labels.cpu().numpy()

    accuracy = evaluator.eval({
        "y_true": labels.reshape(-1, 1),
        "y_pred": predictions.reshape(-1, 1)
    })['acc']

    return accuracy
