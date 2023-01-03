from argparse import Namespace
from collections import OrderedDict
from copy import deepcopy
from typing import List, Optional, Dict, Mapping, Any

import networkx as nx
import numpy as np
import torch
from torch import nn, Tensor
from torch_geometric.nn.inits import glorot, zeros
from transformers import BertForSequenceClassification, BertConfig

import dgl
from moge.dataset.PyG.hetero_generator import HeteroNodeClfDataset
from moge.dataset.graph import HeteroGraphDataset
from moge.model.dgl.HGT import HGT


class LabelGraphNodeClassifier(nn.Module):
    def __init__(self, dataset: HeteroGraphDataset, hparams: Namespace):
        super().__init__()
        self.n_classes = hparams.n_classes
        self.classes = dataset.classes
        self.n_heads = hparams.attn_heads

        if hparams.layer_pooling == "concat":
            self.n_heads = self.n_heads * hparams.n_layers
        elif hparams.layer_pooling == "order_concat":
            self.n_heads = self.n_heads * hparams.t_order

        self.out_channels = hparams.embedding_dim // self.n_heads

        self.dropout = nn.Dropout(p=hparams.nb_cls_dropout)

        assert isinstance(hparams.cls_graph, dgl.DGLGraph)
        self.g: dgl.DGLHeteroGraph = hparams.cls_graph
        if "input_ids" in self.g.ndata and "cls_encoder" in hparams:
            go_encoder = self.create_encoder(hparams)
        else:
            go_encoder = None

        self.embeddings = nn.ParameterDict(
            {ntype: nn.Embedding(self.g.num_nodes(ntype), embedding_dim=hparams.embedding_dim) for ntype in
             self.g.ntypes})
        print("model.classifier.embeddings", self.embeddings)

        self.embedder = HGT(node_dict={ntype: i for i, ntype in enumerate(self.g.ntypes)},
                            edge_dict={etype: i for i, etype in enumerate(self.g.etypes)},
                            n_inp=hparams.embedding_dim,
                            n_hid=hparams.embedding_dim, n_out=hparams.embedding_dim,
                            n_layers=2,
                            n_heads=self.n_heads,
                            dropout=hparams.dropout,
                            use_norm=True)

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

    def forward(self, embeddings: Tensor, classes: Optional[List[str]] = None,
                cls_emb: Dict[str, Tensor] = None, **kwargs) -> Tensor:
        if self.g.device != embeddings.device:
            self.g = self.g.to(embeddings.device)

        if cls_emb is None:
            cls_emb = {ntype: self.embeddings[ntype].weight for ntype in self.g.ntypes}
        cls_emb = self.embedder.forward(G=self.g,
                                        feat=cls_emb)["_N"]
        cls_emb = self.dropout(cls_emb)

        if classes is None:
            cls_emb = cls_emb[:self.n_classes]
        else:
            mask = np.isin(self.classes, classes, )
            cls_emb = cls_emb[mask]

        logits = embeddings @ cls_emb.T
        # side_A = (embeddings * self.attn_kernels).unsqueeze(1)  # (n_edges, 1, emb_dim)
        # emb_B = cls_emb.unsqueeze(2)  # (n_edges, emb_dim, 1)
        # logits = side_A[:, None, :] @ emb_B[None, :, :]
        # logits = logits.squeeze(-1).squeeze(-1)

        return logits


class LabelNodeClassifer(nn.Module):
    def __init__(self, dataset: HeteroNodeClfDataset, hparams: Namespace):
        super().__init__()
        self.n_classes = hparams.n_classes
        self.classes = dataset.classes
        self.head_node_type = hparams.head_node_type

        self.pred_ntypes = dataset.pred_ntypes
        assert dataset.class_indices, f'dataset.class_indices ({dataset.class_indices}) must not be none'
        self.class_sizes = {ntype: ids.numel() \
                            for ntype, ids in dataset.class_indices.items()}

        # if hparams.embedding_dim
        self.embedding_dim = hparams.embedding_dim
        self.weight = nn.Parameter(torch.rand(hparams.embedding_dim))
        # self.bias = nn.Parameter(torch.zeros(self.n_classes))

        if hparams.loss_type == "BCE":
            self.activation = nn.Sigmoid()
        elif hparams.loss_type == "SOFTMAX_CROSS_ENTROPY":
            self.activation = nn.Softmax()

        # self.batchnorm = nn.BatchNorm1d(hparams.embedding_dim)
        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, 'weight') and self.weight.dim() > 1:
            nn.init.xavier_uniform_(self.weight)

    def forward(self, emb: Tensor, h_dict: Dict[str, Tensor], **kwargs) -> Tensor:
        for ntype in self.pred_ntypes:
            cls_emb = h_dict[ntype][:self.class_sizes[ntype]]

        assert cls_emb.shape[0] == self.n_classes, f"cls_emb.shape ({cls_emb.shape}) != n_classes ({self.n_classes})"
        logits = ((emb * self.weight) @ cls_emb.T)  # + self.bias

        if hasattr(self, 'activation'):
            logits = self.activation(logits)
        return logits


class DenseClassification(nn.Module):
    def __init__(self, hparams: Namespace):
        super().__init__()
        # Classifier
        if getattr(hparams, 'nb_cls_dense_size', 0) > 0:
            self.linears = nn.Sequential(OrderedDict([
                ("linear_1", nn.Linear(hparams.embedding_dim, hparams.nb_cls_dense_size)),
                ("relu", nn.ReLU()),
                ("dropout", nn.Dropout(p=getattr(hparams, 'nb_cls_dropout', 0.0))),
                ("linear", nn.Linear(hparams.nb_cls_dense_size, hparams.n_classes))
            ]))
        else:
            self.linears = nn.Sequential(OrderedDict([
                ("linear", nn.Linear(hparams.embedding_dim, hparams.n_classes))
            ]))

        # Activation
        self.loss_type = hparams.loss_type
        if "LOGITS" in self.loss_type or "FOCAL" in self.loss_type:
            print("INFO: Output of `classifier` is logits")

        elif "NEGATIVE_LOG_LIKELIHOOD" == self.loss_type:
            print("INFO: Output of `classifier` is LogSoftmax")
            self.linears.add_module("activation", nn.LogSoftmax(dim=1))

        elif "SOFTMAX_CROSS_ENTROPY" == self.loss_type:
            print("INFO: Output of `classifier` is logits")

        elif "BCE" == self.loss_type:
            print("INFO: Output of `classifier` is sigmoid probabilities")
            self.linears.add_module("activation", nn.Sigmoid())

        else:
            print("INFO: [Else Case] Output of `classifier` is logits")

        self.reset_parameters()

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        return super().load_state_dict(state_dict, strict)

    def reset_parameters(self):
        for linear in self.linears:
            if hasattr(linear, "weight"):
                nn.init.xavier_uniform_(linear.weight)

    def forward(self, h, **kwargs):
        h = self.linears(h)

        return h


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
