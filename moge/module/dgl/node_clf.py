import multiprocessing
import copy

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

import dgl
import dgl.function as fn
from dgl.heterograph import DGLHeteroGraph, DGLBlock
import dgl.nn.pytorch as dglnn
from dgl.udf import EdgeBatch, NodeBatch
from dgl.utils import expand_as_pair

from moge.data import DGLNodeSampler
from moge.module.classifier import DenseClassification
from moge.module.losses import ClassificationLoss
from moge.module.utils import filter_samples
from ..trainer import NodeClfTrainer, print_pred_class_counts

from moge.module.dgl.latte import LATTE
from ...module.utils import tensor_sizes
from .hgt import HGT, HGTLayer, HGTNodeClf

class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)  # (M, 1)
        beta = torch.softmax(w, dim=0)  # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)

        return (beta * z).sum(1)  # (N, D * K)


class StochasticTwoLayerRGCN(nn.Module):
    def __init__(self, in_feat, hidden_feat, out_feat, rel_names):
        super().__init__()
        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GATConv(in_feat, hidden_feat, num_heads=4)
            for rel in rel_names
        })
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GATConv(hidden_feat, out_feat, num_heads=4)
            for rel in rel_names
        })

    def forward(self, blocks, x):
        x = self.conv1(blocks[0], x)
        x = self.conv2(blocks[1], x)
        return x


class GAT(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers, ntypes, etypes):
        super().__init__()
        self.n_layers = n_layers

        self.linear_inp = nn.ModuleDict()
        for ntype in ntypes:
            self.linear_inp[ntype] = nn.Linear(in_dim, hid_dim)

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(GATLayer(hid_dim, out_dim, ntypes, etypes))

    def forward(self, blocks, feat_dict):
        h = {}
        for ntype in feat_dict:
            h[ntype] = F.gelu(self.linear_inp[ntype].forward(feat_dict[ntype]))

        for i in range(self.n_layers):
            h = self.layers[i].forward(blocks[i], h)
            # print(f"layer {i}", tensor_sizes(h))

        return h


class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, ntypes, etypes):
        super(GATLayer, self).__init__()
        self.ntypes = ntypes
        self.etypes = etypes

        self.W = nn.ModuleDict({
            ntype: nn.Linear(in_dim, out_dim, bias=False) for ntype in ntypes
        })

        self.attn = nn.ModuleDict({
            etype: nn.Linear(2 * out_dim, 1, bias=False) for etype in etypes
        })

        self.dropout = nn.Dropout(p=0.4)

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        for ntype in self.W.keys():
            nn.init.xavier_normal_(self.W[ntype].weight, gain=gain)

        for etype in self.attn.keys():
            nn.init.xavier_normal_(self.attn[etype].weight, gain=gain)

    def edge_attention(self, edges: EdgeBatch):
        srctype, etype, dsttype = edges.canonical_etype
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)

        a = self.attn[etype].forward(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges: EdgeBatch):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes: NodeBatch):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, g: DGLBlock, input_dict: dict):
        feat_dict = {ntype: self.W[ntype](ndata) for ntype, ndata in input_dict.items()}
        feat_src, feat_dst = expand_as_pair(input_=feat_dict, g=g)

        with g.local_scope():
            # print(g)
            for ntype in feat_dict:
                # print("ntype", "srcdata", g.srcnodes[ntype].data.keys(), "dstdata", g.srcnodes[ntype].data.keys())
                g.srcnodes[ntype].data['z'] = feat_src[ntype]
                g.dstnodes[ntype].data['z'] = feat_dst[ntype]

            funcs = {}
            for etype in self.etypes:
                if g.batch_num_edges(etype=etype).item() > 0:
                    g.apply_edges(self.edge_attention, etype=etype)
                    funcs[etype] = (self.message_func, self.reduce_func)

            g.multi_update_all(funcs, cross_reducer="mean")

            new_h = {}
            for ntype in g.ntypes:
                if "h" in g.dstnodes[ntype].data:
                    new_h[ntype] = self.dropout(g.dstnodes[ntype].data['h'])

            return new_h


class LATTENodeClassifier(NodeClfTrainer):
    def __init__(self, hparams, dataset: DGLNodeSampler, metrics=["accuracy"], collate_fn="neighbor_sampler") -> None:
        super(LATTENodeClassifier, self).__init__(hparams=hparams, dataset=dataset, metrics=metrics)
        self.head_node_type = dataset.head_node_type
        self.node_types = dataset.node_types
        self.dataset = dataset
        self.multilabel = dataset.multilabel
        self.y_types = list(dataset.y_dict.keys())
        self._name = f"DGL_LATTE-{hparams.t_order}"
        self.collate_fn = collate_fn

        # align the dimension of different types of nodes
        self.feature_projection = nn.ModuleDict({
            ntype: nn.Linear(in_channels, hparams.embedding_dim) \
            for ntype, in_channels in dataset.node_attr_shape.items()
        })

        self.embedder = LATTE(t_order=hparams.t_order, embedding_dim=hparams.embedding_dim,
                              num_nodes_dict=dataset.num_nodes_dict,
                              metapaths=dataset.get_metapaths(), activation=hparams.activation,
                              attn_heads=hparams.attn_heads, attn_activation=hparams.attn_activation,
                              attn_dropout=hparams.attn_dropout)

        if "batchnorm" in hparams and hparams.batchnorm:
            self.batchnorm = torch.nn.ModuleDict(
                {node_type: torch.nn.BatchNorm1d(hparams.embedding_dim) for node_type in
                 self.dataset.node_types})

        self.classifier = DenseClassification(hparams)

        self.criterion = ClassificationLoss(n_classes=dataset.n_classes, loss_type=hparams.loss_type,
                                            class_weight=dataset.class_weight if hasattr(dataset, "class_weight") and \
                                                                                 hparams.use_class_weights else None,
                                            multilabel=dataset.multilabel,
                                            reduction=hparams.reduction if hasattr(dataset, "reduction") else "mean")
        self.hparams.n_params = self.get_n_params()

    def forward(self, blocks, feat, **kwargs):
        h_dict = {}
        for ntype in self.node_types:
            if ntype in feat:
                h_dict[ntype] = self.feature_projection[ntype](feat[ntype])

        if hasattr(self, "batchnorm"):
            h_dict = {ntype: self.batchnorm[ntype](emb) \
                      for ntype, emb, in h_dict.items()}

        embeddings = self.embedder.forward(blocks, h_dict, **kwargs)

        y_pred = self.classifier(embeddings[self.head_node_type]) \
            if hasattr(self, "classifier") else embeddings[self.head_node_type]

        return y_pred

    def training_step(self, batch, batch_nb):
        input_nodes, seeds, blocks = batch

        for i, block in enumerate(blocks):
            blocks[i] = block.to(self.device)

        batch_inputs = blocks[0].srcdata['feat']
        if not isinstance(batch_inputs, dict):
            batch_inputs = {self.head_node_type: batch_inputs}
        y_true = blocks[-1].dstdata['labels'][self.head_node_type]

        y_pred = self.forward(blocks, batch_inputs)
        loss = self.criterion.forward(y_pred, y_true)

        self.train_metrics.update_metrics(y_pred, y_true, weights=None)

        self.log("loss", loss, logger=True, on_step=True)
        if batch_nb % 25 == 0:
            logs = self.train_metrics.compute_metrics()
            self.log_dict(logs, prog_bar=True, logger=True, on_step=True)

        return loss

    def validation_step(self, batch, batch_nb):
        input_nodes, seeds, blocks = batch

        for i, block in enumerate(blocks):
            blocks[i] = block.to(self.device)

        batch_inputs = blocks[0].srcdata['feat']
        if not isinstance(batch_inputs, dict):
            batch_inputs = {self.head_node_type: batch_inputs}
        y_true = blocks[-1].dstdata['labels'][self.head_node_type]

        y_pred = self.forward(blocks, batch_inputs)
        val_loss = self.criterion.forward(y_pred, y_true)

        self.valid_metrics.update_metrics(y_pred, y_true, weights=None)
        self.log("val_loss", val_loss, logger=True, on_step=True)
        return val_loss

    def test_step(self, batch, batch_nb):
        input_nodes, seeds, blocks = batch

        for i, block in enumerate(blocks):
            blocks[i] = block.to(self.device)

        batch_inputs = blocks[0].srcdata['feat']
        if not isinstance(batch_inputs, dict):
            batch_inputs = {self.head_node_type: batch_inputs}
        y_true = blocks[-1].dstdata['labels'][self.head_node_type]

        y_pred = self.forward(blocks, batch_inputs)
        test_loss = self.criterion.forward(y_pred, y_true)

        if batch_nb == 0:
            print_pred_class_counts(y_pred, y_true, multilabel=self.dataset.multilabel)

        self.test_metrics.update_metrics(y_pred, y_true, weights=None)
        self.log("test_loss", test_loss, logger=True, on_step=True)
        return test_loss

    def train_dataloader(self):
        return self.dataset.train_dataloader(collate_fn=None,
                                             batch_size=self.hparams.batch_size,
                                             num_workers=0)

    def val_dataloader(self, batch_size=None):
        return self.dataset.valid_dataloader(collate_fn=None,
                                             batch_size=self.hparams.batch_size,
                                             num_workers=0)

    def valtrain_dataloader(self):
        return self.dataset.valtrain_dataloader(collate_fn=None,
                                                batch_size=self.hparams.batch_size,
                                                num_workers=0)

    def test_dataloader(self, batch_size=None):
        return self.dataset.test_dataloader(collate_fn=None,
                                            batch_size=self.hparams.batch_size,
                                            num_workers=0)

    def configure_optimizers(self):
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'alpha_activation',
                    'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for name, p in param_optimizer if not any(key in name for key in no_decay)],
             'weight_decay': self.hparams.weight_decay},
            {'params': [p for name, p in param_optimizer if any(key in name for key in no_decay)],
             'weight_decay': 0.0}
        ]

        optimizer = torch.optim.Adam(optimizer_grouped_parameters,
                                     lr=self.hparams.lr)
        # scheduler = ReduceLROnPlateau(optimizer)

        return {"optimizer": optimizer,
                # "lr_scheduler": scheduler,
                "monitor": "val_loss"}
