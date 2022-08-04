import copy
from argparse import Namespace
from typing import Dict, List

import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F

from moge.dataset import DGLLinkSampler
from moge.model.dgl.HGT import HGT
from moge.model.trainer import LinkPredTrainer


class DotPredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            # Compute a new edge feature named 'score' by a dot-product between the
            # source node feature 'h' and destination node feature 'h'.
            g.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            return g.edata['score'][:, 0]


class MLPPredictor(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2, h_feats)
        self.W2 = nn.Linear(h_feats, 1)

    def apply_edges(self, edges):
        """
        Computes a scalar score for each edge of the given graph.

        Parameters
        ----------
        edges :
            Has three members ``src``, ``dst`` and ``data``, each of
            which is a dictionary representing the features of the
            source nodes, the destination nodes, and the edges
            themselves.

        Returns
        -------
        dict
            A dictionary of new edge features.
        """
        h = torch.cat([edges.src['h'], edges.dst['h']], 1)
        return {'score': self.W2(F.relu(self.W1(h))).squeeze(1)}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)
            return g.edata['score']


class HGConv(LinkPredTrainer):
    def __init__(self, args: Dict, dataset: DGLLinkSampler, metrics: List[str]):
        super().__init__(Namespace(**args), dataset, metrics)
        self.dataset = dataset

        if "node_neighbors_min_num" in args:
            fanouts = [args["node_neighbors_min_num"], ] * len(dataset.neighbor_sizes)
            self.set_fanouts(self.dataset, fanouts)

        hgconv = HGT(node_dict={ntype: i for i, ntype in enumerate(dataset.node_types)},
                     edge_dict={metapath[1]: i for i, metapath in enumerate(dataset.get_metapaths())},
                     n_inp=self.dataset.node_attr_shape[self.head_node_type if isinstance(self.head_node_type, str) \
                         else self.head_node_type[0]],
                     n_hid=args.embedding_dim,
                     n_out=args.embedding_dim,
                     n_layers=self.n_layers,
                     n_heads=args.attn_heads,
                     use_norm=args.use_norm)

        classifier = DotPredictor()

        self.model = nn.Sequential(hgconv, classifier)
        self.criterion = nn.BCEWithLogitsLoss() if dataset.multilabel else nn.CrossEntropyLoss()

        args["n_params"] = self.get_n_params()
        print(f'Model #Params: {self.get_n_params()}')

        print(f'configuration is {args}')
        self._set_hparams(args)

    def forward(self, blocks, input_features):
        nodes_representation = self.model[0](blocks, copy.deepcopy(input_features))
        train_y_predict = self.model[1](nodes_representation[self.hparams['head_node_type']])

        return train_y_predict

    def training_step(self, batch, batch_nb):
        input_nodes, seeds, blocks = batch
        input_features = {ntype: blocks[0].srcnodes[ntype].data['feat'] for ntype in input_nodes.keys()}
        y_true = blocks[-1].dstnodes[self.dataset.head_node_type].data["label"]

        for i, block in enumerate(blocks):
            blocks[i] = block.to(self.device)

        y_pred = self.forward(blocks, input_features)
        loss = self.criterion.forward(y_pred,
                                      y_true.type_as(y_pred) if self.dataset.multilabel else y_true)

        self.train_metrics.update_metrics(y_pred, y_true, weights=None)

        self.log("loss", loss, logger=True, on_step=True)
        if batch_nb % 25 == 0:
            logs = self.train_metrics.compute_metrics()
            self.log_dict(logs, prog_bar=True, logger=True, on_step=True)

        return loss

    def validation_step(self, batch, batch_nb):
        input_nodes, seeds, blocks = batch
        input_features = {ntype: blocks[0].srcnodes[ntype].data['feat'] for ntype in input_nodes.keys()}
        y_true = blocks[-1].dstnodes[self.dataset.head_node_type].data["label"]

        for i, block in enumerate(blocks):
            blocks[i] = block.to(self.device)

        y_pred = self.forward(blocks, input_features)
        val_loss = self.criterion.forward(y_pred,
                                          y_true.type_as(y_pred) if self.dataset.multilabel else y_true)

        self.valid_metrics.update_metrics(y_pred, y_true, weights=None)
        self.log("val_loss", val_loss, prog_bar=True, logger=True)
        return val_loss

    def test_step(self, batch, batch_nb):
        input_nodes, seeds, blocks = batch
        input_features = {ntype: blocks[0].srcnodes[ntype].data['feat'] for ntype in input_nodes.keys()}
        y_true = blocks[-1].dstnodes[self.dataset.head_node_type].data["label"]

        for i, block in enumerate(blocks):
            blocks[i] = block.to(self.device)

        y_pred = self.forward(blocks, input_features)
        test_loss = self.criterion.forward(y_pred,
                                           y_true.type_as(y_pred) if self.dataset.multilabel else y_true)

        if batch_nb == 0:
            print_pred_class_counts(y_pred, y_true, multilabel=self.dataset.multilabel)

        self.test_metrics.update_metrics(y_pred, y_true, weights=None)
        self.log("test_loss", test_loss, logger=True)
        return test_loss

    def train_dataloader(self):
        return self.dataset.train_dataloader(collate_fn=None, batch_size=self.hparams.batch_size, num_workers=0)

    def val_dataloader(self, batch_size=None):
        return self.dataset.valid_dataloader(collate_fn=None, batch_size=self.hparams.batch_size, num_workers=0)

    def test_dataloader(self, batch_size=None):
        return self.dataset.test_dataloader(collate_fn=None, batch_size=self.hparams.batch_size, num_workers=0)

    def configure_optimizers(self):
        if self.hparams['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams['learning_rate'],
                                         weight_decay=self.hparams['weight_decay'])
        elif self.hparams['optimizer'] == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams['learning_rate'],
                                        weight_decay=self.hparams['weight_decay'])
        else:
            raise ValueError(f"wrong value for optimizer {self.hparams['optimizer']}!")

        return {"optimizer": optimizer}
