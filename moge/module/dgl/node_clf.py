import copy
from argparse import Namespace
from typing import Dict, List

import dgl
import torch
from torch import nn
from torch.utils.data import DataLoader

import moge
from moge.data import DGLNodeSampler
from moge.module.classifier import DenseClassification
from moge.module.losses import ClassificationLoss
from ..trainer import NodeClfTrainer, print_pred_class_counts

from moge.module.dgl.latte import LATTE
from moge.module.dgl.HGConv.utils.utils import set_random_seed, load_dataset


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

    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if self.trainer.max_steps:
            return self.trainer.max_steps

        limit_batches = self.trainer.limit_train_batches
        batches = len(self.train_dataloader())
        batches = min(batches, limit_batches) if isinstance(limit_batches, int) else int(limit_batches * batches)

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_accum = self.trainer.accumulate_grad_batches * num_devices
        return (batches // effective_accum) * self.trainer.max_epochs


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

    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if self.trainer.max_steps:
            return self.trainer.max_steps

        limit_batches = self.trainer.limit_train_batches
        batches = len(self.train_dataloader())
        batches = min(batches, limit_batches) if isinstance(limit_batches, int) else int(limit_batches * batches)

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_accum = self.trainer.accumulate_grad_batches * num_devices
        return (batches // effective_accum) * self.trainer.max_epochs
