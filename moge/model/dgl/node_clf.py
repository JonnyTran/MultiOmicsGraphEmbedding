import copy
import logging
import os
from argparse import Namespace
from typing import Dict, List, Iterable, Tuple

import dgl
import torch
from dgl.heterograph import DGLBlock
from torch import nn, Tensor
from torch.utils.data import DataLoader

from moge.dataset.dgl.node_generator import DGLNodeGenerator, HANSampler
from moge.model.classifier import DenseClassification
from moge.model.dgl.NARS import SIGN, WeightedAggregator, sample_relation_subsets, preprocess_features, \
    read_relation_subsets
from moge.model.dgl.R_HGNN.model.R_HGNN import R_HGNN as RHGNN
from moge.model.dgl.latte import LATTE
from moge.model.losses import ClassificationLoss
from .HGConv.model.HGConv import HGConv as Hgconv
from .HGT import HGT
from .conv import HAN as Han
from ..encoder import HeteroNodeFeatureEncoder, HeteroSequenceEncoder
from ..sampling import sample_metapaths
from ..tensor import tensor_sizes, stack_tensor_dicts, filter_samples_weights
from ..trainer import NodeClfTrainer, print_pred_class_counts


class LATTENodeClf(NodeClfTrainer):
    def __init__(self, hparams, dataset: DGLNodeGenerator, metrics=["accuracy"], collate_fn="neighbor_sampler") -> None:
        if not isinstance(hparams, Namespace) and isinstance(hparams, dict):
            hparams = Namespace(**hparams)
        super().__init__(hparams=hparams, dataset=dataset, metrics=metrics)
        self.head_node_type = dataset.head_node_type
        self.node_types = dataset.node_types
        self.dataset: DGLNodeGenerator = dataset
        self.multilabel = dataset.multilabel
        self.y_types = list(dataset.y_dict.keys())
        self._name = f"LATTE-{hparams.n_layers}"
        self.collate_fn = collate_fn

        if "fanouts" in hparams:
            self.dataset.neighbor_sizes = hparams.fanouts
            self.dataset.neighbor_sampler.fanouts = hparams.fanouts
            self.dataset.neighbor_sampler.num_layers = len(hparams.fanouts)

        # Node attr input
        if hasattr(dataset, 'seq_tokenizer'):
            self.seq_encoder = HeteroSequenceEncoder(hparams, dataset)
        if not hasattr(self, "seq_encoder") or len(self.seq_encoder.seq_encoders.keys()) < len(self.node_types):
            self.encoder = HeteroNodeFeatureEncoder(hparams, dataset)

        self.embedder = LATTE(n_layers=hparams.n_layers,
                              t_order=hparams.t_order,
                              embedding_dim=hparams.embedding_dim,
                              num_nodes_dict=dataset.num_nodes_dict,
                              head_node_type=dataset.head_node_type,
                              metapaths=dataset.get_metapaths(),
                              batchnorm=hparams.batchnorm if "batchnorm" in hparams else False,
                              layernorm=hparams.layernorm if "layernorm" in hparams else False,
                              attn_heads=hparams.attn_heads,
                              attn_dropout=hparams.attn_dropout)

        self.classifier = DenseClassification(hparams)

        self.criterion = ClassificationLoss(
            loss_type=hparams.loss_type, n_classes=dataset.n_classes,
            class_weight=dataset.class_weight if hasattr(dataset, "class_weight") and \
                                                 'use_class_weights' in hparams and hparams.use_class_weights else None,
            pos_weight=dataset.pos_weight if hasattr(dataset, "pos_weight") and
                                             'use_pos_weights' in hparams and hparams.use_pos_weights else None,
            multilabel=dataset.multilabel,
            reduction="mean" if "reduction" not in hparams else hparams.reduction)

        self.hparams.n_params = self.get_n_params()

    def forward(self, blocks: List[DGLBlock], feat: Dict[str, Tensor], return_embeddings=False, **kwargs):
        h_dict = {}

        if len(h_dict) < len(blocks[0].srctypes):
            embs = self.encoder.forward(feat, global_node_index=blocks[0].srcdata["_ID"])
            h_dict.update({ntype: emb for ntype, emb in embs.items() if ntype not in h_dict})

        h_dict = self.embedder.forward(blocks, h_dict, **kwargs)

        if hasattr(self, "classifier"):
            logits = self.classifier(h_dict[self.head_node_type])
        else:
            logits = h_dict[self.head_node_type]

        if return_embeddings:
            return h_dict, logits

        return logits

    def process_blocks(self, blocks: List[DGLBlock]) -> Tuple[Dict[str, Tensor], Tensor]:
        batch_inputs = blocks[0].srcdata['feat']

        if not isinstance(batch_inputs, dict):
            batch_inputs = {self.head_node_type: batch_inputs}

        y_true = blocks[-1].dstdata['label']
        y_true = y_true[self.head_node_type] if isinstance(y_true, dict) else y_true
        return batch_inputs, y_true

    def training_step(self, batch, batch_nb):
        input_nodes, seeds, blocks = batch

        for i, block in enumerate(blocks):
            blocks[i] = block.to(self.device)

        batch_inputs, y_true = self.process_blocks(blocks)

        logits = self.forward(blocks, batch_inputs)
        loss = self.criterion.forward(logits, y_true)

        if batch_nb % 100 == 0:
            print_pred_class_counts(logits, y_true, multilabel=self.dataset.multilabel)

        self.update_node_clf_metrics(self.train_metrics, logits, y_true, weights=None)

        self.log("loss", loss, logger=True, on_step=True)

        return loss

    def validation_step(self, batch, batch_nb):
        input_nodes, seeds, blocks = batch

        for i, block in enumerate(blocks):
            blocks[i] = block.to(self.device)

        batch_inputs, y_true = self.process_blocks(blocks)

        logits = self.forward(blocks, batch_inputs)
        val_loss = self.criterion.forward(logits, y_true)

        if batch_nb == 0:
            print_pred_class_counts(logits, y_true, multilabel=self.dataset.multilabel)

        self.update_node_clf_metrics(self.valid_metrics, logits, y_true, weights=None)
        self.log("val_loss", val_loss, prog_bar=True, logger=True)
        return val_loss

    def test_step(self, batch, batch_nb):
        input_nodes, seeds, blocks = batch

        for i, block in enumerate(blocks):
            blocks[i] = block.to(self.device)

        batch_inputs, y_true = self.process_blocks(blocks)

        logits = self.forward(blocks, batch_inputs)
        test_loss = self.criterion.forward(logits, y_true)

        if batch_nb == 0:
            print_pred_class_counts(logits, y_true, multilabel=self.dataset.multilabel)

        self.update_node_clf_metrics(self.test_metrics, logits, y_true, weights=None)
        self.log("test_loss", test_loss, logger=True)
        return test_loss


class HGConv(NodeClfTrainer):
    def __init__(self, hparams: Dict, dataset: DGLNodeGenerator, metrics: List[str]):
        if not isinstance(hparams, Namespace) and isinstance(hparams, dict):
            hparams = Namespace(**hparams)
        super().__init__(Namespace(**hparams), dataset, metrics)
        self.dataset = dataset

        if "node_neighbors_min_num" in hparams:
            fanouts = [hparams["node_neighbors_min_num"], ] * len(dataset.neighbor_sizes)
            self.set_fanouts(self.dataset, fanouts)

        hgconv = Hgconv(graph=dataset.G,
                        input_dim_dict={ntype: dataset.G.nodes[ntype].data['feat'].shape[1]
                                        for ntype in dataset.G.ntypes},
                        hidden_dim=hparams['hidden_units'],
                        num_layers=len(dataset.neighbor_sizes),
                        n_heads=hparams['num_heads'],
                        dropout=hparams['dropout'],
                        residual=hparams['residual'])

        classifier = nn.Linear(hparams['hidden_units'] * hparams['num_heads'], dataset.n_classes)

        self.model = nn.Sequential(hgconv, classifier)
        self.criterion = nn.BCEWithLogitsLoss() if dataset.multilabel else nn.CrossEntropyLoss()

        hparams["n_params"] = self.get_n_params()
        print(f'Model #Params: {self.get_n_params()}')

        print(f'configuration is {hparams}')
        self._set_hparams(hparams)

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
        if y_pred.dim() == 1:
            weights = (y_true >= 0).to(torch.float)
        elif y_pred.dim() == 2:
            weights = (y_true.sum(1) > 0).to(torch.float)

        y_pred, y_true, weights = filter_samples_weights(y_pred=y_pred, y_true=y_true, weights=weights)
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
        if y_pred.dim() == 1:
            weights = (y_true >= 0).to(torch.float)
        elif y_pred.dim() == 2:
            weights = (y_true.sum(1) > 0).to(torch.float)

        y_pred, y_true, weights = filter_samples_weights(y_pred=y_pred, y_true=y_true, weights=weights)
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
        if y_pred.dim() == 1:
            weights = (y_true >= 0).to(torch.float)
        elif y_pred.dim() == 2:
            weights = (y_true.sum(1) > 0).to(torch.float)

        y_pred, y_true, weights = filter_samples_weights(y_pred=y_pred, y_true=y_true, weights=weights)
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


class R_HGNN(NodeClfTrainer):
    def __init__(self, hparams: Dict, dataset: DGLNodeGenerator, metrics: List[str]):
        if not isinstance(hparams, Namespace) and isinstance(hparams, dict):
            hparams = Namespace(**hparams)
        super().__init__(Namespace(**hparams), dataset, metrics)
        self.dataset = dataset

        if "node_neighbors_min_num" in hparams:
            fanouts = [hparams["node_neighbors_min_num"], ] * len(dataset.neighbor_sizes)
            self.set_fanouts(self.dataset, fanouts)

        self.r_hgnn = RHGNN(graph=dataset.G,
                            input_dim_dict={ntype: dataset.G.nodes[ntype].data['feat'].shape[1]
                                            for ntype in dataset.G.ntypes},
                            hidden_dim=hparams['hidden_units'],
                            relation_input_dim=hparams['relation_hidden_units'],
                            relation_hidden_dim=hparams['relation_hidden_units'],
                            num_layers=len(dataset.neighbor_sizes),
                            n_heads=hparams['num_heads'],
                            dropout=hparams['dropout'],
                            residual=hparams['residual'])

        self.classifier = nn.Linear(hparams['hidden_units'] * hparams['num_heads'], dataset.n_classes)

        self.model = nn.Sequential(self.r_hgnn, self.classifier)
        self.criterion = nn.BCEWithLogitsLoss() if dataset.multilabel else nn.CrossEntropyLoss()

        hparams["n_params"] = self.get_n_params()
        print(f'Model #Params: {self.get_n_params()}')

        print(f'configuration is {hparams}')
        self._set_hparams(hparams)

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
        if y_pred.dim() == 1:
            weights = (y_true >= 0).to(torch.float)
        elif y_pred.dim() == 2:
            weights = (y_true.sum(1) > 0).to(torch.float)

        y_pred, y_true, weights = filter_samples_weights(y_pred=y_pred, y_true=y_true, weights=weights)
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
        input_features = {(stype, etype, dtype): blocks[0].srcnodes[dtype].data['feat'] for stype, etype, dtype in
                          blocks[0].canonical_etypes}
        y_true = blocks[-1].dstnodes[self.dataset.head_node_type].data["label"]

        for i, block in enumerate(blocks):
            blocks[i] = block.to(self.device)

        y_pred = self.forward(blocks, input_features)
        if y_pred.dim() == 1:
            weights = (y_true >= 0).to(torch.float)
        elif y_pred.dim() == 2:
            weights = (y_true.sum(1) > 0).to(torch.float)

        y_pred, y_true, weights = filter_samples_weights(y_pred=y_pred, y_true=y_true, weights=weights)
        val_loss = self.criterion.forward(y_pred,
                                          y_true.type_as(y_pred) if self.dataset.multilabel else y_true)

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
        if y_pred.dim() == 1:
            weights = (y_true >= 0).to(torch.float)
        elif y_pred.dim() == 2:
            weights = (y_true.sum(1) > 0).to(torch.float)

        y_pred, y_true, weights = filter_samples_weights(y_pred=y_pred, y_true=y_true, weights=weights)
        test_loss = self.criterion.forward(y_pred,
                                           y_true.type_as(y_pred) if self.dataset.multilabel else y_true)

        if batch_nb == 0:
            print_pred_class_counts(y_pred, y_true, multilabel=self.dataset.multilabel)

        self.test_metrics.update_metrics(y_pred, y_true, weights=None)
        self.log("test_loss", test_loss, logger=True)
        return test_loss


class NARSNodeCLf(NodeClfTrainer):
    def __init__(self, hparams: Namespace, dataset: DGLNodeGenerator, metrics: List[str]):
        if not isinstance(hparams, Namespace) and isinstance(hparams, dict):
            hparams = Namespace(**hparams)
        hparams.loss_type = "BCE_WITH_LOGITS" if dataset.multilabel else "SOFTMAX_CROSS_ENTROPY"
        super().__init__(hparams, dataset, metrics)

        self.dataset = dataset

        if "use_relation_subsets" in hparams and os.path.exists(hparams.use_relation_subsets):
            rel_subsets = read_relation_subsets(hparams.use_relation_subsets)
        else:
            rel_subsets = []
            subsets = sample_relation_subsets(self.dataset.G.metagraph(), hparams)
            for relation in set(subsets):
                etypes = []

                # only save subsets that touches target node type
                target_touched = False
                for u, v, e in relation:
                    etypes.append(e)
                    if u == hparams.head_node_type or v == hparams.head_node_type:
                        target_touched = True

                print(etypes, target_touched and "touched" or "not touched")
                if target_touched:
                    rel_subsets.append(etypes)

        print("rel_subsets", rel_subsets)
        self.rel_subsets = rel_subsets

        with torch.no_grad():
            feats = preprocess_features(self.dataset.G, self.rel_subsets, hparams)
            print("Done preprocessing")

            self.dataset.feats = feats
            self.dataset.labels = self.dataset.G.nodes[hparams.head_node_type].data["label"]
            print("feats", tensor_sizes(feats))
            print("labels", tensor_sizes(self.dataset.labels))

        _, num_feats, in_feats = feats[0].shape
        logging.info(f"new input size: {num_feats} {in_feats}")

        num_hops = hparams.R + 1  # include self feature hop 0
        self.model = nn.Sequential(
            WeightedAggregator(num_feats, in_feats, num_hops),
            SIGN(in_feats, hparams.num_hidden, dataset.n_classes, num_hops,
                 hparams.ff_layer, hparams.dropout, hparams.input_dropout)
        )

        self.criterion = nn.BCEWithLogitsLoss(reduction="mean") if dataset.multilabel else nn.CrossEntropyLoss()

        hparams.n_params = self.get_n_params()
        print(f'Model #Params: {self.get_n_params()}')

        print(f'configuration is {hparams}')
        self._set_hparams(hparams)

    def forward(self, batch_feats):
        return self.model(batch_feats)

    def training_step(self, batch, batch_nb):
        input_features, y_true = batch
        if y_true.dim() == 2 and y_true.size(1) == 1:
            y_true = y_true.squeeze(-1)

        y_pred = self.forward(input_features)

        if y_pred.dim() == 1:
            weights = (y_true >= 0).to(torch.float)
        elif y_pred.dim() == 2:
            weights = (y_true.sum(1) > 0).to(torch.float)

        y_pred, y_true, weights = filter_samples_weights(y_pred=y_pred, y_true=y_true, weights=weights)
        loss = self.criterion.forward(y_pred,
                                      y_true.type_as(y_pred) if self.dataset.multilabel else y_true)

        self.train_metrics.update_metrics(y_pred, y_true, weights=None)

        self.log("loss", loss, logger=True, on_step=True)
        if batch_nb % 25 == 0:
            logs = self.train_metrics.compute_metrics()
            self.log_dict(logs, prog_bar=True, logger=True, on_step=True)

        return loss

    def validation_step(self, batch, batch_nb):
        input_features, y_true = batch
        if y_true.dim() == 2 and y_true.size(1) == 1:
            y_true = y_true.squeeze(-1)

        y_pred = self.forward(input_features)

        if y_pred.dim() == 1:
            weights = (y_true >= 0).to(torch.float)
        elif y_pred.dim() == 2:
            weights = (y_true.sum(1) > 0).to(torch.float)

        y_pred, y_true, weights = filter_samples_weights(y_pred=y_pred, y_true=y_true, weights=weights)
        val_loss = self.criterion.forward(y_pred,
                                          y_true.type_as(y_pred) if self.dataset.multilabel else y_true)

        self.valid_metrics.update_metrics(y_pred, y_true, weights=None)
        self.log("val_loss", val_loss, prog_bar=True, logger=True)
        return val_loss

    def test_step(self, batch, batch_nb):
        input_features, y_true = batch
        if y_true.dim() == 2 and y_true.size(1) == 1:
            y_true = y_true.squeeze(-1)

        y_pred = self.forward(input_features)

        if y_pred.dim() == 1:
            weights = (y_true >= 0).to(torch.float)
        elif y_pred.dim() == 2:
            weights = (y_true.sum(1) > 0).to(torch.float)

        y_pred, y_true, weights = filter_samples_weights(y_pred=y_pred, y_true=y_true, weights=weights)
        test_loss = self.criterion.forward(y_pred,
                                           y_true.type_as(y_pred) if self.dataset.multilabel else y_true)

        if batch_nb == 0:
            print_pred_class_counts(y_pred, y_true, multilabel=self.dataset.multilabel)

        self.test_metrics.update_metrics(y_pred, y_true, weights=None)
        self.log("test_loss", test_loss, logger=True)
        return test_loss

    def collate(self, batch, history=None):
        batch_feats = [x[batch] for x in self.dataset.feats]
        if history is not None:
            # Train aggregator partially using history
            batch_feats = (batch_feats, [x[batch] for x in history])

        y_true = self.dataset.labels[batch]
        return batch_feats, y_true

    def train_dataloader(self):
        return DataLoader(
            dataset=self.dataset.training_idx,
            batch_size=self.hparams.batch_size, collate_fn=self.collate, shuffle=True, )

    def val_dataloader(self, batch_size=None):
        return DataLoader(
            dataset=self.dataset.validation_idx,
            batch_size=self.hparams.batch_size, collate_fn=self.collate, shuffle=False, )

    def test_dataloader(self, batch_size=None):
        return DataLoader(
            dataset=self.dataset.testing_idx,
            batch_size=self.hparams.batch_size, collate_fn=self.collate, shuffle=False, )


class HANNodeClf(NodeClfTrainer):
    def __init__(self, hparams: Dict, dataset: DGLNodeGenerator, metrics: List[str]):
        if not isinstance(hparams, Namespace) and isinstance(hparams, dict):
            hparams = Namespace(**hparams)
        super().__init__(Namespace(**hparams), dataset, metrics)
        self.dataset = dataset

        # metapath_list = [['pa', 'ap'], ['pf', 'fp']]
        edge_paths = sample_metapaths(metagraph=dataset.G.metagraph(),
                                      source=self.dataset.head_node_type,
                                      targets=self.dataset.head_node_type,
                                      cutoff=5)

        self.metapath_list = [[etype for srctype, dsttype, etype in metapaths] for metapaths in edge_paths]
        print("metapath_list", self.metapath_list)

        self.num_neighbors = hparams['num_neighbors']
        self.han_sampler = HANSampler(dataset.G, self.metapath_list, self.num_neighbors)

        self.features = dataset.G.nodes[dataset.head_node_type].data["feat"]
        self.labels = dataset.G.nodes[dataset.head_node_type].data["label"]
        self.model = Han(num_metapath=len(self.metapath_list),
                         in_size=set(dataset.node_attr_shape.values()).pop(),
                         hidden_size=hparams['hidden_units'],
                         out_size=dataset.n_classes,
                         num_heads=hparams['num_heads'],
                         dropout=hparams['dropout'])

        self.criterion = ClassificationLoss(loss_type=hparams["loss_type"], n_classes=dataset.n_classes,
                                            multilabel=dataset.multilabel)

        hparams["n_params"] = self.get_n_params()
        print(f'Model #Params: {self.get_n_params()}')

        print(f'configuration is {hparams}')
        self._set_hparams(hparams)

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


class HGTNodeClf(NodeClfTrainer):
    def __init__(self, hparams, dataset: DGLNodeGenerator, metrics=["accuracy"]) -> None:
        if not isinstance(hparams, Namespace) and isinstance(hparams, dict):
            hparams = Namespace(**hparams)
        super().__init__(hparams=hparams, dataset=dataset, metrics=metrics)
        self.head_node_type = dataset.head_node_type
        self.dataset = dataset
        self.multilabel = dataset.multilabel
        self.y_types = list(dataset.y_dict.keys())

        if "fanouts" in hparams and isinstance(hparams.fanouts, Iterable) \
                and len(self.dataset.neighbor_sizes) != len(hparams.fanouts):
            self.set_fanouts(self.dataset, hparams.fanouts)

        self.encoder = HeteroNodeFeatureEncoder(hparams, dataset)

        self.embedder = HGT(node_dict={ntype: i for i, ntype in enumerate(dataset.node_types)},
                            edge_dict={metapath[1]: i for i, metapath in enumerate(dataset.get_metapaths())},
                            n_inp=hparams.embedding_dim,
                            n_hid=hparams.embedding_dim,
                            n_out=hparams.embedding_dim,
                            n_layers=hparams.n_layers,
                            n_heads=hparams.attn_heads,
                            dropout=hparams.dropout,
                            use_norm=hparams.use_norm)

        self.classifier = DenseClassification(hparams)

        self.criterion = ClassificationLoss(loss_type=hparams.loss_type, n_classes=dataset.n_classes,
                                            class_weight=dataset.class_weight if hasattr(dataset, "class_weight") and \
                                                                                 hparams.use_class_weights else None,
                                            multilabel=dataset.multilabel)

        self.hparams.n_params = self.get_n_params()

    def forward(self, blocks: List[DGLBlock], x: Dict[str, Tensor], return_embeddings: bool = False, **kwargs):
        input_nodes = blocks[0].srcdata["_ID"]
        x = self.encoder.forward(feats=x, global_node_index=input_nodes)

        embeddings = self.embedder(blocks, x)

        if isinstance(self.head_node_type, str):
            y_hat = self.classifier(embeddings[self.head_node_type]) \
                if hasattr(self, "classifier") else embeddings[self.head_node_type]

        elif isinstance(self.head_node_type, (tuple, list)):
            if hasattr(self, "classifier"):
                y_hat = {ntype: self.classifier(emb) for ntype, emb in embeddings.items()}
            else:
                y_hat = embeddings

        if return_embeddings:
            return embeddings, y_hat

        return y_hat

    def training_step(self, batch, batch_nb):
        input_nodes, seeds, blocks = batch
        feats = blocks[0].srcdata['feat']
        y_true = blocks[-1].dstdata['label']

        y_pred = self.forward(blocks, feats if isinstance(feats, dict) else {self.head_node_type: feats})

        weights = {ntype: (label.sum(1) > 0).to(torch.float) if label.dim() == 2 else (label >= 0).to(torch.float) \
                   for ntype, label in y_true.items() if label.numel()}

        y_pred, y_true, weights = stack_tensor_dicts(y_pred, y_true, weights=weights)
        y_pred, y_true, weights = filter_samples_weights(y_pred=y_pred, y_true=y_true, weights=weights)
        loss = self.criterion.forward(y_pred, y_true)

        self.update_node_clf_metrics(self.train_metrics, y_pred, y_true, weights=None)

        self.log("loss", loss, logger=True, on_step=True)

        return loss

    def validation_step(self, batch, batch_nb):
        input_nodes, seeds, blocks = batch
        feats = blocks[0].srcdata['feat']
        y_true = blocks[-1].dstdata['label']

        y_pred = self.forward(blocks, feats if isinstance(feats, dict) else {self.head_node_type: feats})

        weights = {ntype: (label.sum(1) > 0).to(torch.float) if label.dim() == 2 else (label >= 0).to(torch.float) \
                   for ntype, label in y_true.items() if label.numel()}

        y_pred, y_true, weights = stack_tensor_dicts(y_pred, y_true, weights=weights)
        y_pred, y_true, weights = filter_samples_weights(y_pred=y_pred, y_true=y_true, weights=weights)
        loss = self.criterion.forward(y_pred, y_true)

        self.update_node_clf_metrics(self.valid_metrics, y_pred, y_true, weights=None)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_nb):
        input_nodes, seeds, blocks = batch
        feats = blocks[0].srcdata['feat']
        y_true = blocks[-1].dstdata['label']

        y_pred = self.forward(blocks, feats if isinstance(feats, dict) else {self.head_node_type: feats})

        weights = {ntype: (label.sum(1) > 0).to(torch.float) if label.dim() == 2 else (label >= 0).to(torch.float) \
                   for ntype, label in y_true.items() if label.numel()}

        y_pred, y_true, weights = stack_tensor_dicts(y_pred, y_true, weights=weights)
        y_pred, y_true, weights = filter_samples_weights(y_pred=y_pred, y_true=y_true, weights=weights)
        loss = self.criterion.forward(y_pred, y_true)

        self.update_node_clf_metrics(self.test_metrics, y_pred, y_true, weights=None)
        self.log("test_loss", loss, logger=True)
        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters())
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=self.num_training_steps,
        #                                                 max_lr=1e-3, pct_start=0.05)

        return {"optimizer": optimizer,
                # "lr_scheduler": scheduler, "monitor": "val_loss"
                }
