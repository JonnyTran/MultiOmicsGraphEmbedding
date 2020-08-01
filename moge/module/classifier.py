from argparse import ArgumentParser
from collections import OrderedDict
from copy import deepcopy

import networkx as nx
import numpy as np
import torch
from torch import nn
from torch_geometric.nn.inits import glorot, zeros


class Dense(nn.Module):
    def __init__(self, hparams) -> None:
        super(Dense, self).__init__()

        # Classifier
        self.fc_classifier = nn.Sequential(OrderedDict([
            ("linear_1", nn.Linear(hparams.embedding_dim, hparams.nb_cls_dense_size)),
            ("relu", nn.ReLU()),
            ("dropout", nn.Dropout(p=hparams.nb_cls_dropout)),
            ("linear", nn.Linear(hparams.nb_cls_dense_size, hparams.n_classes))
        ]))
        if "LOGITS" in hparams.loss_type or "FOCAL" in hparams.loss_type:
            print("INFO: Output of `_classifier` is logits")

        elif "NEGATIVE_LOG_LIKELIHOOD" in hparams.loss_type:
            self.fc_classifier.add_module("pred_activation", nn.LogSoftmax(dim=1))
            print("INFO: Output of `_classifier` is logits")
            # print("INFO: Output of `_classifier` is Softmax")
        elif "SOFTMAX_CROSS_ENTROPY" in hparams.loss_type:
            # self.fc_classifier.add_module("pred_activation", nn.Sigmoid())
            print("INFO: Output of `_classifier` is linear")

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument('--nb_cls_dense_size', type=int, default=512)
        parser.add_argument('--nb_cls_dropout', type=float, default=0.2)
        parser.add_argument('--n_classes', type=int, default=128)
        return parser

    def forward(self, embeddings):
        return self.fc_classifier(embeddings)


class HierarchicalAWX(nn.Module):
    def __init__(self, hparams, bias=True) -> None:
        super(HierarchicalAWX, self).__init__()
        class_adj = hparams.class_adj
        self.n = hparams.awx_n_norm
        self.leaves = class_adj.sum(0) == 0
        units = sum(self.leaves)
        self.A = deepcopy(class_adj)

        print("leaves", units)
        R = np.zeros(class_adj.shape)
        R[self.leaves, self.leaves] = 1

        self.g = nx.DiGraph(class_adj)

        for i in np.where(self.leaves)[0]:
            ancestors = list(nx.descendants(self.g, i))
            if ancestors:
                R[i, ancestors] = 1

        print("Child units", units)
        self.linear = nn.Parameter(torch.Tensor(hparams.embedding_dim, units))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(units))
        else:
            self.register_parameter('bias', None)

        self.R = torch.tensor(R[self.leaves], requires_grad=False).type_as(self.linear)
        self.R_t = torch.tensor(R[self.leaves].T, requires_grad=False).type_as(self.linear)

        self.hparams = hparams
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.linear)
        zeros(self.bias)

    def forward(self, inputs):
        output = torch.sigmoid(torch.matmul(inputs, self.linear) + self.bias)

        if self.n > 1:
            output = self.n_norm(torch.mul(torch.unsqueeze(output, 1), self.R_t.type_as(self.linear)))
        elif self.n > 0:
            output = torch.min(torch.mul(torch.unsqueeze(output, 1), self.R_t.type_as(self.linear)) - 1,
                               other=torch.tensor(1 - 1e-4).type_as(self.bias))
        else:
            output = torch.max(torch.multiply(torch.unsqueeze(output, 1), self.R_t.type_as(self.linear)), -1)

        return output

    def n_norm(self, x, epsilon=1e-6):
        return torch.pow(torch.clamp(torch.sum(torch.pow(x, self.n), -1), min=epsilon, max=1 - epsilon), 1. / self.n)
