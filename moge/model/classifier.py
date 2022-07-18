from argparse import Namespace
from collections import OrderedDict
from copy import deepcopy

import dgl
import networkx as nx
import numpy as np
import torch
from torch import nn, Tensor
from torch_geometric.nn.inits import glorot, zeros
from transformers import BertForSequenceClassification, BertConfig

from moge.model.dgl.HGT import Hgt


class LabelGraphNodeClassifier(nn.Module):
    def __init__(self, hparams: Namespace):
        super().__init__()
        self.n_classes = hparams.n_classes
        self.classes = hparams.classes
        self.n_heads = hparams.attn_heads

        if hparams.layer_pooling == "concat":
            self.n_heads = self.n_heads * hparams.n_layers
        elif hparams.layer_pooling == "order_concat":
            self.n_heads = self.n_heads * hparams.t_order

        self.out_channels = hparams.embedding_dim // self.n_heads

        self.dropout = nn.Dropout(p=hparams.nb_cls_dropout)
        self.attn_kernels = nn.Parameter(torch.rand((hparams.embedding_dim)), requires_grad=True)

        assert isinstance(hparams.cls_graph, dgl.DGLGraph)
        self.g: dgl.DGLHeteroGraph = hparams.cls_graph
        if "input_ids" in self.g.ndata and "cls_encoder" in hparams:
            go_encoder = self.create_encoder(hparams)
        else:
            go_encoder = None

        # self.embedder = HeteroRGCN(self.g, in_size=hparams.embedding_dim, hidden_size=128,
        #                            out_size=hparams.embedding_dim, encoder=go_encoder)
        self.embeddings = nn.ParameterDict(
            {ntype: nn.Embedding(self.g.num_nodes(ntype), embedding_dim=hparams.embedding_dim) for ntype in
             self.g.ntypes})

        self.embedder = Hgt(node_dict={ntype: i for i, ntype in enumerate(self.g.ntypes)},
                            edge_dict={etype: i for i, etype in enumerate(self.g.etypes)},
                            n_inp=hparams.embedding_dim,
                            n_hid=hparams.embedding_dim, n_out=hparams.embedding_dim,
                            n_layers=2,
                            n_heads=self.n_heads,
                            use_norm=True)

        self.embedder.cls_graph_nodes = hparams.classes

    def forward(self, embeddings: Tensor, classes=None) -> Tensor:
        if self.g.device != embeddings.device:
            self.g = self.g.to(embeddings.device)

        cls_emb = {ntype: self.embeddings[ntype].weight for ntype in self.g.ntypes}
        cls_emb = self.embedder(blocks=self.g, feat_dict=cls_emb)["_N"]
        cls_emb = self.dropout(cls_emb)

        if classes is None:
            cls_emb = cls_emb[:self.n_classes]
        else:
            mask = np.isin(self.embedder.cls_graph_nodes, classes, )
            cls_emb = cls_emb[mask]

        # logits = embeddings @ cls_emb.T
        side_A = (embeddings * self.attn_kernels).unsqueeze(1)  # (n_edges, 1, emb_dim)
        emb_B = cls_emb.unsqueeze(2)  # (n_edges, emb_dim, 1)
        logits = side_A[:, None, :] @ emb_B[None, :, :]
        logits = logits.squeeze(-1).squeeze(-1)

        return logits

    def create_encoder(self, hparams):
        if isinstance(hparams.cls_encoder, str):
            go_encoder = BertForSequenceClassification.from_pretrained("dmis-lab/biobert-base-cased-v1.2",
                                                                       num_hidden_layers=1,
                                                                       num_labels=hparams.embedding_dim)
        elif isinstance(hparams.cls_encoder, BertConfig):
            hparams.cls_encoder.num_labels = hparams.embedding_dim
            go_encoder = BertForSequenceClassification(hparams.cls_encoder)

        elif isinstance(hparams.cls_encoder, BertForSequenceClassification):
            go_encoder = hparams.cls_encoder

        return go_encoder


class DenseClassification(nn.Module):
    def __init__(self, hparams) -> None:
        super(DenseClassification, self).__init__()
        # Classifier
        if hparams.nb_cls_dense_size > 0:
            self.fc_classifier = nn.Sequential(OrderedDict([
                ("linear_1", nn.Linear(hparams.embedding_dim, hparams.nb_cls_dense_size)),
                ("relu", nn.ReLU()),
                ("dropout", nn.Dropout(p=hparams.nb_cls_dropout)),
                ("linear", nn.Linear(hparams.nb_cls_dense_size, hparams.n_classes))
            ]))
        else:
            self.fc_classifier = nn.Sequential(OrderedDict([
                ("linear", nn.Linear(hparams.embedding_dim, hparams.n_classes))
            ]))

        # Activation
        if "LOGITS" in hparams.loss_type or "FOCAL" in hparams.loss_type:
            print("INFO: Output of `_classifier` is logits")
        elif "NEGATIVE_LOG_LIKELIHOOD" == hparams.loss_type:
            print("INFO: Output of `_classifier` is LogSoftmax")

            self.fc_classifier.add_module("pred_activation", nn.LogSoftmax(dim=1))
        elif "SOFTMAX_CROSS_ENTROPY" == hparams.loss_type:
            print("INFO: Output of `_classifier` is logits")

        elif "BCE" == hparams.loss_type:
            print("INFO: Output of `_classifier` is sigmoid probabilities")
            self.fc_classifier.add_module("pred_activation", nn.Sigmoid())
        else:
            print("INFO: [Else Case] Output of `_classifier` is logits")
        self.reset_parameters()

    def forward(self, embeddings):
        return self.fc_classifier(embeddings)

    def reset_parameters(self):
        for linear in self.fc_classifier:
            if hasattr(linear, "weight"):
                glorot(linear.weight)


class MulticlassClassification(nn.Module):
    def __init__(self, num_feature, num_class, loss_type):
        super(MulticlassClassification, self).__init__()

        # Classifier
        self.fc_classifier = nn.Sequential(OrderedDict([
            ("layer_1", nn.Linear(num_feature, 512)),
            ("batchnorm1", nn.BatchNorm1d(512)),
            ("relu", nn.ReLU()),
            ("layer_2", nn.Linear(512, 128)),
            ("batchnorm2", nn.BatchNorm1d(128)),
            ("relu", nn.ReLU()),
            ("dropout", nn.Dropout(p=0.2)),
            ("layer_3", nn.Linear(128, 64)),
            ("batchnorm3", nn.BatchNorm1d(64)),
            ("relu", nn.ReLU()),
            ("dropout", nn.Dropout(p=0.2)),
            ("layer_out", nn.Linear(64, num_class)),
        ]))

        if "NEGATIVE_LOG_LIKELIHOOD" == loss_type:
            print("INFO: Output of `_classifier` is LogSoftmax")
            self.fc_classifier.add_module("pred_activation", nn.LogSoftmax(dim=1))
        elif "BCE" == loss_type:
            print("INFO: Output of `_classifier` is sigmoid probabilities")
            self.fc_classifier.add_module("pred_activation", nn.Sigmoid())
        else:
            print("INFO: Output of `_classifier` is logits")

    def forward(self, x):
        return self.fc_classifier.forward(x)


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
