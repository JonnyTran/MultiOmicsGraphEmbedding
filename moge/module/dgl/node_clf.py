import copy
import logging
import os
from argparse import Namespace
from typing import Dict, List

import dgl
import torch
from torch import nn
from torch.utils.data import DataLoader

import moge
from moge.data import DGLNodeSampler
from moge.module.classifier import DenseClassification
from moge.module.dgl.NARS import SIGN, WeightedAggregator, sample_relation_subsets, preprocess_features, \
    read_relation_subsets
from moge.module.dgl.RHGNN.model.R_HGNN import R_HGNN as RHGNN
from moge.module.dgl.latte import LATTE
from moge.module.losses import ClassificationLoss
from .hgt import Hgt
from ..sampling import sample_metapaths
from ..trainer import NodeClfTrainer, print_pred_class_counts
from ..utils import tensor_sizes
from ...data.dgl.node_generator import NARSDataLoader

from .conv import HAN as Han
from moge.data.dgl.node_generator import HANSampler


class LATTENodeClassifier(NodeClfTrainer):
    def __init__(self, hparams, dataset: DGLNodeSampler, metrics=["accuracy"], collate_fn="neighbor_sampler") -> None:
        super(LATTENodeClassifier, self).__init__(hparams=hparams, dataset=dataset, metrics=metrics)
        self.head_node_type = dataset.head_node_type
        self.node_types = dataset.node_types
        self.dataset = dataset
        self.multilabel = dataset.multilabel
        self.y_types = list(dataset.y_dict.keys())
        self._name = f"DGL_LATTE-{hparams.n_layers}"
        self.collate_fn = collate_fn

        if "fanouts" in hparams:
            self.dataset.neighbor_sizes = hparams.fanouts
            self.dataset.neighbor_sampler.fanouts = hparams.fanouts
            self.dataset.neighbor_sampler.num_layers = len(hparams.fanouts)

        # align the dimension of different types of nodes
        self.feature_projection = nn.ModuleDict({
            ntype: nn.Linear(in_channels, hparams.embedding_dim) \
            for ntype, in_channels in dataset.node_attr_shape.items()
        })

        self.embedder = LATTE(t_order=hparams.n_layers, embedding_dim=hparams.embedding_dim,
                              num_nodes_dict=dataset.num_nodes_dict,
                              metapaths=dataset.get_metapaths(),
                              batchnorm=hparams.batchnorm if "batchnorm" in hparams else False,
                              layernorm=hparams.layernorm if "layernorm" in hparams else False,
                              activation=hparams.activation,
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

        if isinstance(dataset.node_attr_shape, dict):
            non_attr_node_types = (dataset.num_nodes_dict.keys() - dataset.node_attr_shape.keys())
        else:
            non_attr_node_types = []

        if len(non_attr_node_types) > 0:
            print("Embedding.device = 'gpu'", "num_nodes_dict", dataset.num_nodes_dict)
            self.embeddings = nn.ModuleDict(
                {node_type: nn.Embedding(num_embeddings=dataset.num_nodes_dict[node_type],
                                         embedding_dim=hparams.embedding_dim,
                                         sparse=False) for node_type in non_attr_node_types})
        else:
            self.embeddings = None

    def forward(self, blocks, feat, **kwargs):
        h_dict = {}

        for ntype in self.node_types:
            if isinstance(feat, torch.Tensor) and ntype in self.feature_projection:
                h_dict[ntype] = self.feature_projection[ntype](feat)
            elif isinstance(feat, dict) and ntype in feat and ntype in self.feature_projection:
                h_dict[ntype] = self.feature_projection[ntype](feat[ntype])
            else:
                h_dict[ntype] = feat[ntype]

            if hasattr(self, "batchnorm") and ntype in self.feature_projection:
                h_dict[ntype] = self.batchnorm[ntype](h_dict[ntype])

        embeddings = self.embedder.forward(blocks, h_dict, **kwargs)

        y_pred = self.classifier(embeddings[self.head_node_type]) \
            if hasattr(self, "classifier") else embeddings[self.head_node_type]

        return y_pred

    def process_blocks(self, blocks):
        if self.embeddings is not None:
            batch_inputs = {ntype: self.embeddings[ntype].weight[blocks[0].ndata["_ID"][ntype]].to(self.device) \
                            for ntype in self.node_types}
        else:
            batch_inputs = blocks[0].srcdata['feat']

        if not isinstance(batch_inputs, dict):
            batch_inputs = {self.head_node_type: batch_inputs}

        y_true = blocks[-1].dstdata['labels']
        y_true = y_true[self.head_node_type] if isinstance(y_true, dict) else y_true
        return batch_inputs, y_true

    def training_step(self, batch, batch_nb):
        input_nodes, seeds, blocks = batch

        for i, block in enumerate(blocks):
            blocks[i] = block.to(self.device)

        batch_inputs, y_true = self.process_blocks(blocks)

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

        batch_inputs, y_true = self.process_blocks(blocks)

        y_pred = self.forward(blocks, batch_inputs)
        val_loss = self.criterion.forward(y_pred, y_true)

        # if batch_nb == 0:
        #     print_pred_class_counts(y_pred, y_true, multilabel=self.dataset.multilabel)

        self.valid_metrics.update_metrics(y_pred, y_true, weights=None)
        self.log("val_loss", val_loss, prog_bar=True, logger=True)
        return val_loss

    def test_step(self, batch, batch_nb):
        input_nodes, seeds, blocks = batch

        for i, block in enumerate(blocks):
            blocks[i] = block.to(self.device)

        batch_inputs, y_true = self.process_blocks(blocks)

        y_pred = self.forward(blocks, batch_inputs)
        test_loss = self.criterion.forward(y_pred, y_true)

        if batch_nb == 0:
            print_pred_class_counts(y_pred, y_true, multilabel=self.dataset.multilabel)

        self.test_metrics.update_metrics(y_pred, y_true, weights=None)
        self.log("test_loss", test_loss, logger=True)
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
        no_decay = ['bias', 'alpha_activation', 'embedding', 'batchnorm', 'layernorm']

        optimizer_grouped_parameters = [
            {'params': [p for name, p in param_optimizer if not any(key in name for key in no_decay)],
             'weight_decay': self.hparams.weight_decay},
            {'params': [p for name, p in param_optimizer if any(key in name for key in no_decay)],
             'weight_decay': 0.0}
        ]

        print("weight_decay", [name for name, p in param_optimizer if not any(key in name for key in no_decay)])

        optimizer = torch.optim.Adam(optimizer_grouped_parameters,
                                     lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_training_steps,
                                                               eta_min=self.hparams.lr / 100)

        return {"optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "val_loss"}


class HGConv(NodeClfTrainer):
    def __init__(self, args: Dict, dataset: DGLNodeSampler, metrics: List[str]):
        super().__init__(Namespace(**args), dataset, metrics)
        self.dataset = dataset

        self.hgconv = moge.module.dgl.HGConv.HGConv(graph=dataset.G,
                                                    input_dim_dict={ntype: dataset.G.nodes[ntype].data['feat'].shape[1]
                                                                    for ntype in dataset.G.ntypes},
                                                    hidden_dim=args['hidden_units'],
                                                    num_layers=len(dataset.neighbor_sizes),
                                                    n_heads=args['num_heads'],
                                                    dropout=args['dropout'],
                                                    residual=args['residual'])

        self.classifier = nn.Linear(args['hidden_units'] * args['num_heads'], dataset.n_classes)

        self.model = nn.Sequential(self.hgconv, self.classifier)
        self.criterion = nn.CrossEntropyLoss()

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
        loss = self.criterion.forward(y_pred, y_true)

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
        val_loss = self.criterion.forward(y_pred, y_true)

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
        test_loss = self.criterion.forward(y_pred, y_true)

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

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=len(self.train_dataloader()) * self.hparams[
                                                                   "epochs"],
                                                               eta_min=self.hparams['learning_rate'] / 100)

        return {"optimizer": optimizer,
                "scheduler": scheduler}



class R_HGNN(NodeClfTrainer):
    def __init__(self, args: Dict, dataset: DGLNodeSampler, metrics: List[str]):
        super(R_HGNN, self).__init__(Namespace(**args), dataset, metrics)
        self.dataset = dataset

        self.r_hgnn = RHGNN(graph=dataset.G,
                            input_dim_dict={ntype: dataset.G.nodes[ntype].data['feat'].shape[1]
                                            for ntype in dataset.G.ntypes},
                            hidden_dim=args['hidden_units'],
                            relation_input_dim=args['relation_hidden_units'],
                            relation_hidden_dim=args['relation_hidden_units'],
                            num_layers=len(dataset.neighbor_sizes),
                            n_heads=args['num_heads'],
                            dropout=args['dropout'],
                            residual=args['residual'])

        self.classifier = nn.Linear(args['hidden_units'] * args['num_heads'], dataset.n_classes)

        self.model = nn.Sequential(self.r_hgnn, self.classifier)
        self.criterion = nn.CrossEntropyLoss()

        args["n_params"] = self.get_n_params()
        print(f'Model #Params: {self.get_n_params()}')

        print(f'configuration is {args}')
        self._set_hparams(args)

    def forward(self, blocks, input_features):
        nodes_representation, _ = self.model[0](blocks, input_features)
        train_y_predict = self.model[1](nodes_representation[self.hparams['head_node_type']])

        return train_y_predict

    def training_step(self, batch, batch_nb):
        input_nodes, seeds, blocks = batch
        input_features = {(stype, etype, dtype): blocks[0].srcnodes[dtype].data['feat'] for stype, etype, dtype in
                          blocks[0].canonical_etypes}
        y_true = blocks[-1].dstnodes[self.dataset.head_node_type].data["label"]

        for i, block in enumerate(blocks):
            blocks[i] = block.to(self.device)

        y_pred = self.forward(blocks, input_features)
        loss = self.criterion.forward(y_pred, y_true)

        self.train_metrics.update_metrics(y_pred, y_true, weights=None)

        self.log("loss", loss, logger=True, on_step=True)
        if batch_nb % 25 == 0:
            logs = self.train_metrics.compute_metrics()
            self.log_dict(logs, prog_bar=True, logger=True, on_step=True)

        return loss

    def validation_step(self, batch, batch_nb):
        input_nodes, seeds, blocks = batch
        input_features = {(stype, etype, dtype): blocks[0].srcnodes[dtype].data['feat'] for stype, etype, dtype in
                          blocks[0].canonical_etypes}
        y_true = blocks[-1].dstnodes[self.dataset.head_node_type].data["label"]

        for i, block in enumerate(blocks):
            blocks[i] = block.to(self.device)

        y_pred = self.forward(blocks, input_features)
        val_loss = self.criterion.forward(y_pred, y_true)

        self.valid_metrics.update_metrics(y_pred, y_true, weights=None)
        self.log("val_loss", val_loss, prog_bar=True, logger=True)
        return val_loss

    def test_step(self, batch, batch_nb):
        input_nodes, seeds, blocks = batch
        input_features = {(stype, etype, dtype): blocks[0].srcnodes[dtype].data['feat'] for stype, etype, dtype in
                          blocks[0].canonical_etypes}
        y_true = blocks[-1].dstnodes[self.dataset.head_node_type].data["label"]

        for i, block in enumerate(blocks):
            blocks[i] = block.to(self.device)

        y_pred = self.forward(blocks, input_features)
        test_loss = self.criterion.forward(y_pred, y_true)

        if batch_nb == 0:
            print_pred_class_counts(y_pred, y_true, multilabel=self.dataset.multilabel)

        self.test_metrics.update_metrics(y_pred, y_true, weights=None)
        self.log("test_loss", test_loss, logger=True)
        return test_loss

    def train_dataloader(self):
        return self.dataset.train_dataloader(collate_fn=None,
                                             batch_size=self.hparams.batch_size,
                                             num_workers=0)

    def val_dataloader(self, batch_size=None):
        return self.dataset.valid_dataloader(collate_fn=None,
                                             batch_size=self.hparams.batch_size,
                                             num_workers=0)

    def test_dataloader(self, batch_size=None):
        return self.dataset.test_dataloader(collate_fn=None,
                                            batch_size=self.hparams.batch_size,
                                            num_workers=0)

    def configure_optimizers(self):
        if self.hparams['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams['learning_rate'],
                                         weight_decay=self.hparams['weight_decay'])
        elif self.hparams['optimizer'] == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams['learning_rate'],
                                        weight_decay=self.hparams['weight_decay'])
        else:
            raise ValueError(f"wrong value for optimizer {self.hparams['optimizer']}!")

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=len(self.train_dataloader()) * self.hparams[
                                                                   "epochs"],
                                                               eta_min=self.hparams['learning_rate'] / 100)

        return {"optimizer": optimizer,
                "scheduler": scheduler}


class NARS(NodeClfTrainer):
    def __init__(self, args: Namespace, dataset: DGLNodeSampler, metrics: List[str]):
        args.loss_type = "KL_DIVERGENCE" if dataset.multilabel else "NEGATIVE_LOG_LIKELIHOOD"
        super(NARS, self).__init__(args, dataset, metrics)

        self.dataset = dataset

        if "use_relation_subsets" in args and os.path.exists(args.use_relation_subsets):
            rel_subsets = read_relation_subsets(args.use_relation_subsets)
        else:
            rel_subsets = []
            subsets = sample_relation_subsets(self.dataset.G.metagraph(), args)
            for relation in set(subsets):
                etypes = []

                # only save subsets that touches target node type
                target_touched = False
                for u, v, e in relation:
                    etypes.append(e)
                    if u == args.head_node_type or v == args.head_node_type:
                        target_touched = True

                print(etypes, target_touched and "touched" or "not touched")
                if target_touched:
                    rel_subsets.append(etypes)

        print("rel_subsets", rel_subsets)
        self.rel_subsets = rel_subsets

        with torch.no_grad():
            feats = preprocess_features(self.dataset.G, self.rel_subsets, args)
            print("Done preprocessing")

            self.dataset.feats = feats
            self.dataset.labels = self.dataset.G.nodes[args.head_node_type].data["label"]
            print("feats", tensor_sizes(feats))
            print("labels", tensor_sizes(self.dataset.labels))

        _, num_feats, in_feats = feats[0].shape
        logging.info(f"new input size: {num_feats} {in_feats}")

        num_hops = args.R + 1  # include self feature hop 0
        self.model = nn.Sequential(
            WeightedAggregator(num_feats, in_feats, num_hops),
            SIGN(in_feats, args.num_hidden, dataset.n_classes, num_hops,
                 args.ff_layer, args.dropout, args.input_dropout)
        )

        self.criterion = nn.KLDivLoss(reduction='batchmean') if dataset.multilabel else nn.NLLLoss()

        args.n_params = self.get_n_params()
        print(f'Model #Params: {self.get_n_params()}')

        print(f'configuration is {args}')
        self._set_hparams(args)

    def forward(self, batch_feats):
        return self.model(batch_feats)

    def training_step(self, batch, batch_nb):
        input_features, y_true = batch

        y_pred = self.forward(input_features)
        loss = self.criterion.forward(y_pred, y_true)

        self.train_metrics.update_metrics(y_pred, y_true, weights=None)

        self.log("loss", loss, logger=True, on_step=True)
        if batch_nb % 25 == 0:
            logs = self.train_metrics.compute_metrics()
            self.log_dict(logs, prog_bar=True, logger=True, on_step=True)

        return loss

    def validation_step(self, batch, batch_nb):
        input_features, y_true = batch

        y_pred = self.forward(input_features)
        val_loss = self.criterion.forward(y_pred, y_true)

        self.valid_metrics.update_metrics(y_pred, y_true, weights=None)
        self.log("val_loss", val_loss, prog_bar=True, logger=True)
        return val_loss

    def test_step(self, batch, batch_nb):
        input_features, y_true = batch

        y_pred = self.forward(input_features)
        test_loss = self.criterion.forward(y_pred, y_true)

        if batch_nb == 0:
            print_pred_class_counts(y_pred, y_true, multilabel=self.dataset.multilabel)

        self.test_metrics.update_metrics(y_pred, y_true, weights=None)
        self.log("test_loss", test_loss, logger=True)
        return test_loss

    def train_dataloader(self):
        dataloader = NARSDataLoader(self.dataset.training_idx, batch_size=self.hparams.batch_size,
                                    feats=self.dataset.feats,
                                    labels=self.dataset.labels, shuffle=True)

        return dataloader

    def val_dataloader(self, batch_size=None):
        dataloader = NARSDataLoader(self.dataset.validation_idx, batch_size=self.hparams.batch_size,
                                    feats=self.dataset.feats,
                                    labels=self.dataset.labels, shuffle=False)

        return dataloader

    def test_dataloader(self, batch_size=None):
        dataloader = NARSDataLoader(self.dataset.testing_idx, batch_size=self.hparams.batch_size,
                                    feats=self.dataset.feats,
                                    labels=self.dataset.labels, shuffle=False)

        return dataloader

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.hparams['lr'],
                                     weight_decay=self.hparams[
                                         'weight_decay'] if "weight_decay" in self.hparams else 0.0)

        return optimizer


class HAN(NodeClfTrainer):
    def __init__(self, args: Dict, dataset: DGLNodeSampler, metrics: List[str]):
        super(HAN, self).__init__(Namespace(**args), dataset, metrics)
        self.dataset = dataset

        # if dataset.name() == "ACM":
        #     metapath_list = [['pa', 'ap'], ['pf', 'fp']]
        # else:
        edge_paths = sample_metapaths(metagraph=dataset.G.metagraph(),
                                      source=self.dataset.head_node_type,
                                      targets=self.dataset.head_node_type,
                                      cutoff=5)

        self.metapath_list = [[etype for srctype, dsttype, etype in metapaths] for metapaths in edge_paths]
        print("metapath_list", self.metapath_list)

        self.num_neighbors = args['num_neighbors']
        self.han_sampler = HANSampler(dataset.G, self.metapath_list, self.num_neighbors)

        self.features = dataset.G.nodes[dataset.head_node_type].data["feat"]
        self.labels = dataset.G.nodes[dataset.head_node_type].data["label"]
        self.model = Han(num_metapath=len(self.metapath_list),
                         in_size=set(dataset.node_attr_shape.values()).pop(),
                         hidden_size=args['hidden_units'],
                         out_size=dataset.n_classes,
                         num_heads=args['num_heads'],
                         dropout=args['dropout'])

        self.criterion = nn.CrossEntropyLoss()

        args["n_params"] = self.get_n_params()
        print(f'Model #Params: {self.get_n_params()}')

        print(f'configuration is {args}')
        self._set_hparams(args)

    def forward(self, blocks, input_features):
        return self.model(blocks, input_features)

    def load_subtensors(self, blocks, features):
        h_list = []
        for block in blocks:
            input_nodes = block.srcdata[dgl.NID]
            h_list.append(features[input_nodes])
        return h_list

    def training_step(self, batch, batch_nb):
        seeds, blocks = batch
        for i, block in enumerate(blocks):
            blocks[i] = block.to(self.device)
        y_true = self.labels[seeds].to(self.device)

        h_list = self.load_subtensors(blocks, self.features.to(self.device))

        y_pred = self.forward(blocks, h_list)
        loss = self.criterion.forward(y_pred, y_true)

        self.train_metrics.update_metrics(y_pred, y_true, weights=None)

        self.log("loss", loss, logger=True, on_step=True)
        if batch_nb % 25 == 0:
            logs = self.train_metrics.compute_metrics()
            self.log_dict(logs, prog_bar=True, logger=True, on_step=True)

        return loss

    def validation_step(self, batch, batch_nb):
        seeds, blocks = batch
        for i, block in enumerate(blocks):
            blocks[i] = block.to(self.device)
        y_true = self.labels[seeds].to(self.device)

        h_list = self.load_subtensors(blocks, self.features.to(self.device))

        y_pred = self.forward(blocks, h_list)
        val_loss = self.criterion.forward(y_pred, y_true)

        self.valid_metrics.update_metrics(y_pred, y_true, weights=None)
        self.log("val_loss", val_loss, prog_bar=True, logger=True)
        return val_loss

    def test_step(self, batch, batch_nb):
        seeds, blocks = batch
        for i, block in enumerate(blocks):
            blocks[i] = block.to(self.device)
        y_true = self.labels[seeds].to(self.device)

        h_list = self.load_subtensors(blocks, self.features.to(self.device))

        y_pred = self.forward(blocks, h_list)
        test_loss = self.criterion.forward(y_pred, y_true)

        if batch_nb == 0:
            print_pred_class_counts(y_pred, y_true, multilabel=self.dataset.multilabel)

        self.test_metrics.update_metrics(y_pred, y_true, weights=None)
        self.log("test_loss", test_loss, logger=True)
        return test_loss

    def train_dataloader(self):
        if self.dataset.inductive:
            han_sampler = HANSampler(self.dataset.get_training_subgraph(), self.metapath_list, self.num_neighbors)
        else:
            han_sampler = self.han_sampler

        return DataLoader(
            dataset=self.dataset.training_idx,
            batch_size=self.hparams['batch_size'], collate_fn=han_sampler.sample_blocks, shuffle=True, )

    def val_dataloader(self, batch_size=None):
        return DataLoader(
            dataset=self.dataset.validation_idx,
            batch_size=self.hparams['batch_size'], collate_fn=self.han_sampler.sample_blocks, shuffle=True,
        )

    def test_dataloader(self, batch_size=None):
        return DataLoader(
            dataset=self.dataset.testing_idx,
            batch_size=self.hparams['batch_size'], collate_fn=self.han_sampler.sample_blocks, shuffle=False,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams['lr'],
                                     weight_decay=self.hparams['weight_decay'])

        return optimizer


class HGT(NodeClfTrainer):
    def __init__(self, hparams, dataset: DGLNodeSampler, metrics=["accuracy"]) -> None:
        super(HGT, self).__init__(hparams=hparams, dataset=dataset, metrics=metrics)
        self.head_node_type = dataset.head_node_type
        self.dataset = dataset
        self.multilabel = dataset.multilabel
        self.y_types = list(dataset.y_dict.keys())

        if "fanouts" in hparams:
            self.dataset.neighbor_sizes = hparams.fanouts
            self.dataset.neighbor_sampler.fanouts = hparams.fanouts
            self.dataset.neighbor_sampler.num_layers = len(hparams.fanouts)

        self.n_layers = len(self.dataset.neighbor_sizes)

        self.model = Hgt(node_dict={ntype: i for i, ntype in enumerate(dataset.node_types)},
                         edge_dict={metapath[1]: i for i, metapath in enumerate(dataset.get_metapaths())},
                         n_inp=self.dataset.node_attr_shape[self.head_node_type],
                         n_hid=hparams.embedding_dim, n_out=hparams.embedding_dim,
                         n_layers=self.n_layers,
                         n_heads=hparams.attn_heads,
                         use_norm=hparams.use_norm)

        self.classifier = DenseClassification(hparams)

        self.criterion = ClassificationLoss(n_classes=dataset.n_classes, loss_type=hparams.loss_type,
                                            class_weight=dataset.class_weight if hasattr(dataset, "class_weight") and \
                                                                                 hparams.use_class_weights else None,
                                            multilabel=dataset.multilabel)

        self._name = f"HGT-{self.n_layers}"
        self.hparams.n_params = self.get_n_params()

    def forward(self, blocks, batch_inputs: dict, **kwargs):
        embeddings = self.model(blocks, batch_inputs)

        y_pred = self.classifier(embeddings[self.head_node_type])
        return y_pred

    def training_step(self, batch, batch_nb):
        input_nodes, seeds, blocks = batch
        batch_inputs = blocks[0].srcdata['feat']
        if not isinstance(batch_inputs, dict):
            batch_inputs = {self.head_node_type: batch_inputs}
        y_true = blocks[-1].dstdata['label']
        y_true = y_true[self.head_node_type] if isinstance(y_true, dict) else y_true

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
        batch_inputs = blocks[0].srcdata['feat']
        if not isinstance(batch_inputs, dict):
            batch_inputs = {self.head_node_type: batch_inputs}
        y_true = blocks[-1].dstdata['label']
        y_true = y_true[self.head_node_type] if isinstance(y_true, dict) else y_true

        y_pred = self.forward(blocks, batch_inputs)
        assert (y_true < 0).sum() == 0, f"y_true negatives: {(y_true < 0).sum()}"
        val_loss = self.criterion.forward(y_pred, y_true)

        self.valid_metrics.update_metrics(y_pred, y_true, weights=None)
        self.log("val_loss", val_loss, prog_bar=True, logger=True)
        return val_loss

    def test_step(self, batch, batch_nb):
        input_nodes, seeds, blocks = batch
        batch_inputs = blocks[0].srcdata['feat']
        if not isinstance(batch_inputs, dict):
            batch_inputs = {self.head_node_type: batch_inputs}
        y_true = blocks[-1].dstdata['label']
        y_true = y_true[self.head_node_type] if isinstance(y_true, dict) else y_true

        y_pred = self.forward(blocks, batch_inputs)
        test_loss = self.criterion.forward(y_pred, y_true)

        if batch_nb == 0:
            print_pred_class_counts(y_pred, y_true, multilabel=self.dataset.multilabel)

        self.test_metrics.update_metrics(y_pred, y_true, weights=None)
        self.log("test_loss", test_loss, logger=True)
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
        optimizer = torch.optim.AdamW(self.parameters())
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=100, max_lr=1e-3, pct_start=0.05)

        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
