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

from moge.generator import DGLNodeSampler
from moge.module.classifier import DenseClassification
from moge.module.losses import ClassificationLoss
from moge.module.utils import filter_samples
from ..trainer import NodeClfMetrics

from moge.module.dgl.latte import LATTE
from ...module.utils import tensor_sizes
from .hgt import HGT, HGTLayer

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


class HeteroRGCNLayer(nn.Module):
    def __init__(self, in_size, out_size, etypes):
        super(HeteroRGCNLayer, self).__init__()
        # W_r for each relation
        self.weight = nn.ModuleDict({
            name: nn.Linear(in_size, out_size) for name in etypes
        })

    def forward(self, G: DGLBlock, feat_dict):
        # The input is a dictionary of node features for each type
        funcs = {}
        for srctype, etype, dsttype in G.canonical_etypes:
            # Save it in graph for message passing
            G.nodes[srctype].data['Wh_%s' % etype] = self.weight[etype].forward(feat_dict[srctype])

            # Specify per-relation message passing functions: (message_func, reduce_func).
            # Note that the results are saved to the same destination feature 'h', which
            # hints the type wise reducer for aggregation.
            def message_func(edges: EdgeBatch):
                print(edges.src.keys(), edges.dst.keys())
                return {'m': edges.src[f'Wh_{etype}']}

            def reduce_func(nodes: NodeBatch):
                return {'h': torch.mean(nodes.mailbox['m'], dim=1)}

            funcs[etype] = (message_func, reduce_func)
            # funcs[etype] = (fn.copy_u('Wh_%s' % etype, 'm'), fn.mean('m', 'h'))

        G.multi_update_all(funcs, "sum")
        print("got here")
        # return the updated node feature dictionary
        return {ntype: G.nodes[ntype].data['h'] for ntype in G.ntypes}


class HeteroRGCN(nn.Module):
    def __init__(self, G, in_size, hidden_size, out_size):
        super(HeteroRGCN, self).__init__()
        # create layers
        self.layer1 = HeteroRGCNLayer(in_size, hidden_size, G.etypes)
        self.layer2 = HeteroRGCNLayer(hidden_size, out_size, G.etypes)

    def forward(self, blocks, feat_dict):
        print("input", tensor_sizes(feat_dict))

        h_dict = self.layer1.forward(blocks[0], feat_dict)
        h_dict = {k: F.leaky_relu(h) for k, h in h_dict.items()}
        print("interm", tensor_sizes(h_dict))

        h_dict = self.layer2.forward(blocks[-1], h_dict)

        print("output", tensor_sizes(h_dict))
        return h_dict


class StochasticTwoLayerRGCN(nn.Module):
    def __init__(self, in_feat, hidden_feat, out_feat, rel_names):
        super().__init__()
        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feat, hidden_feat, norm='right')
            for rel in rel_names
        })
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hidden_feat, out_feat, norm='right')
            for rel in rel_names
        })

    def forward(self, blocks, x):
        x = self.conv1(blocks[0], x)
        x = self.conv2(blocks[1], x)
        return x


class LATTENodeClassifier(NodeClfMetrics):
    def __init__(self, hparams, dataset: DGLNodeSampler, metrics=["accuracy"], collate_fn="neighbor_sampler") -> None:
        super(LATTENodeClassifier, self).__init__(hparams=hparams, dataset=dataset, metrics=metrics)
        self.head_node_type = dataset.head_node_type
        self.dataset = dataset
        self.multilabel = dataset.multilabel
        self.y_types = list(dataset.y_dict.keys())
        self._name = f"LATTE-{hparams.t_order}{' proximity' if hparams.use_proximity else ''}"
        self.collate_fn = collate_fn

        # self.latte = LATTE(t_order=hparams.t_order, embedding_dim=hparams.embedding_dim,
        #                    in_channels_dict=dataset.node_attr_shape, num_nodes_dict=dataset.num_nodes_dict,
        #                    metapaths=dataset.get_metapaths(), activation=hparams.activation,
        #                    attn_heads=hparams.attn_heads, attn_activation=hparams.attn_activation,
        #                    attn_dropout=hparams.attn_dropout, use_proximity=hparams.use_proximity,
        #                    neg_sampling_ratio=hparams.neg_sampling_ratio)
        # hparams.embedding_dim = hparams.embedding_dim * hparams.t_order

        # self.embedder = HeteroRGCN(self.dataset.G, in_size=self.dataset.node_attr_shape[self.head_node_type],
        #                            hidden_size=hparams.embedding_dim, out_size=hparams.embedding_dim)

        self.embedder = StochasticTwoLayerRGCN(in_feat=self.dataset.node_attr_shape[self.head_node_type],
                                               hidden_feat=hparams.embedding_dim, out_feat=hparams.embedding_dim,
                                               rel_names=self.dataset.G.etypes)

        self.embedder = HGT(node_dict={ntype: i for i, ntype in enumerate(dataset.node_types)},
                            edge_dict={metapath: i for i, metapath in enumerate(dataset.get_metapaths())},
                            n_inp=self.dataset.node_attr_shape[self.head_node_type],
                            n_hid=hparams.embedding_dim, n_out=hparams.embedding_dim,
                            n_layers=len(self.dataset.neighbor_sizes),
                            n_heads=hparams.attn_heads)

        self.classifier = DenseClassification(hparams)

        self.criterion = ClassificationLoss(n_classes=dataset.n_classes,
                                            class_weight=dataset.class_weight if hasattr(dataset, "class_weight") and \
                                                                                 hparams.use_class_weights else None,
                                            loss_type=hparams.loss_type,
                                            multilabel=dataset.multilabel)
        self.hparams.n_params = self.get_n_params()

    def forward(self, blocks, batch_inputs: dict, **kwargs):
        embeddings = self.embedder.forward(blocks, batch_inputs)

        y_hat = self.classifier.forward(embeddings[self.head_node_type])
        return y_hat

    def training_step(self, batch, batch_nb):
        input_nodes, seeds, blocks = batch
        batch_inputs = blocks[0].srcdata['feat']
        batch_labels = blocks[-1].dstdata['labels'][self.head_node_type]

        y_hat = self.forward(blocks, batch_inputs)
        loss = self.criterion.forward(y_hat, batch_labels)

        self.train_metrics.update_metrics(y_hat, batch_labels, weights=None)

        logs = None

        outputs = {'loss': loss}
        if logs is not None:
            outputs.update({'progress_bar': logs, "logs": logs})
        return outputs

    def validation_step(self, batch, batch_nb):
        input_nodes, seeds, blocks = batch
        batch_inputs = blocks[0].srcdata['feat']
        batch_labels = blocks[-1].dstdata['labels'][self.head_node_type]

        y_hat = self.forward(blocks, batch_inputs)

        val_loss = self.criterion.forward(y_hat, batch_labels)

        self.valid_metrics.update_metrics(y_hat, batch_labels, weights=None)

        return {"val_loss": val_loss}

    def test_step(self, batch, batch_nb):
        input_nodes, seeds, blocks = batch
        batch_inputs = blocks[0].srcdata['feat']
        batch_labels = blocks[-1].dstdata['labels'][self.head_node_type]

        y_hat = self.forward(blocks, batch_inputs, save_betas=True)

        test_loss = self.criterion.forward(y_hat, batch_labels)

        if batch_nb == 0:
            self.print_pred_class_counts(y_hat, batch_labels, multilabel=self.dataset.multilabel)

        self.test_metrics.update_metrics(y_hat, batch_labels, weights=None)

        return {"test_loss": test_loss}

    def train_dataloader(self):
        return self.dataset.train_dataloader(collate_fn=None,
                                             batch_size=self.hparams.batch_size,
                                             num_workers=int(0.4 * multiprocessing.cpu_count()))

    def val_dataloader(self, batch_size=None):
        return self.dataset.valid_dataloader(collate_fn=None,
                                             batch_size=self.hparams.batch_size,
                                             num_workers=max(1, int(0.1 * multiprocessing.cpu_count())))

    def valtrain_dataloader(self):
        return self.dataset.valtrain_dataloader(collate_fn=None,
                                                batch_size=self.hparams.batch_size,
                                                num_workers=max(1, int(0.1 * multiprocessing.cpu_count())))

    def test_dataloader(self, batch_size=None):
        return self.dataset.test_dataloader(collate_fn=None,
                                            batch_size=self.hparams.batch_size,
                                            num_workers=max(1, int(0.1 * multiprocessing.cpu_count())))

    def configure_optimizers(self):
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'alpha_activation']
        optimizer_grouped_parameters = [
            {'params': [p for name, p in param_optimizer if not any(key in name for key in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for name, p in param_optimizer if any(key in name for key in no_decay)], 'weight_decay': 0.0}
        ]

        # optimizer = torch.optim.AdamW(optimizer_grouped_parameters, eps=1e-06, lr=self.hparams.lr)

        optimizer = torch.optim.Adam(optimizer_grouped_parameters,
                                     lr=self.hparams.lr,  # momentum=self.hparams.momentum,
                                     weight_decay=self.hparams.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer)

        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
