import math
import traceback
from argparse import Namespace
from collections import defaultdict
from typing import Dict, List, Union, Tuple, Optional, Callable

import dgl
import dgl.function as fn
import numpy as np
import torch
import torch.nn as nn
from dgl import DGLHeteroGraph
from pandas import DataFrame
from torch import Tensor
from torch.nn import functional as F
from torch.optim import lr_scheduler

from moge.dataset import DGLLinkGenerator
from moge.dataset.dgl.utils import dgl_to_edge_index_dict, round_to_multiple
from moge.dataset.utils import tag_negative_metapath, is_negative, split_edge_index_by_namespace
from moge.model.dgl.HGT import HGT
from moge.model.dgl.latte import LATTE
from moge.model.encoder import HeteroNodeFeatureEncoder
from moge.model.losses import ClassificationLoss
from moge.model.metrics import Metrics
from moge.model.trainer import LinkPredTrainer


class DglLinkPredTrainer(LinkPredTrainer):
    dataset: DGLLinkGenerator

    def __init__(self, hparams, dataset: DGLLinkGenerator, metrics: Union[List[str], Dict[str, List[str]]], *args,
                 **kwargs):
        super().__init__(hparams, dataset, metrics, *args, **kwargs)

        self.pred_metapaths = dataset.pred_metapaths
        self.neg_pred_metapaths = dataset.neg_pred_metapaths

        if hasattr(dataset, "go_namespace"):
            self.go_namespace: Dict[str, np.ndarray] = dataset.go_namespace
        else:
            self.go_namespace = None

        if hasattr(dataset, "ntype_mapping"):
            self.ntype_mapping: Dict[str, str] = dataset.ntype_mapping
        else:
            self.ntype_mapping = None

        self.lr = self.hparams.lr

    def update_link_pred_metrics(self, metrics: Union[Metrics, Dict[str, Metrics]],
                                 pos_edge_scores: Dict[Tuple[str, str, str], Tensor],
                                 neg_batch_scores: Dict[Tuple[str, str, str], Tensor],
                                 neg_edge_scores: Dict[Tuple[str, str, str], Tensor] = {},
                                 pos_stack: Tensor = None,
                                 neg_stack: Tensor = None):

        if isinstance(metrics, dict):
            for metapath in pos_edge_scores:
                go_type = "BPO" if metapath[-1] == 'biological_process' else \
                    "CCO" if metapath[-1] == 'cellular_component' else \
                        "MFO" if metapath[-1] == 'molecular_function' else None

                metrics[go_type].update_metrics(pos_edge_scores[metapath].detach(), neg_batch_scores[metapath].detach(),
                                                weights=None, subset=["ogbl-biokg"])

                if neg_edge_scores and tag_negative_metapath(metapath) in neg_edge_scores:
                    self.update_pr_metrics(pos_scores=pos_edge_scores[metapath],
                                           true_neg_scores=neg_edge_scores[tag_negative_metapath(metapath)],
                                           metrics=metrics[go_type])

        else:
            metapath = list(pos_edge_scores.keys())[0]
            metrics.update_metrics(pos_edge_scores[metapath].detach(), neg_batch_scores[metapath].detach(),
                                   weights=None, subset=["ogbl-biokg"])
            if pos_stack is not None and neg_edge_scores:
                self.update_pr_metrics(pos_scores=pos_stack, true_neg_scores=neg_edge_scores, metrics=metrics)

    def update_pr_metrics(self, pos_scores: Tensor, true_neg_scores: Union[Tensor, Dict[Tuple[str, str, str], Tuple]],
                          metrics: Metrics, subset=["precision", "recall", "aupr"]):
        if isinstance(true_neg_scores, dict):
            neg_scores = torch.cat([edge_scores.detach() for m, edge_scores in true_neg_scores.items()])
        else:
            neg_scores = true_neg_scores

        # randomly select |e_neg| positive edges to balance precision/recall scores
        pos_edge_idx = np.random.choice(pos_scores.size(0), size=neg_scores.shape,
                                        replace=False if pos_scores.size(0) > neg_scores.size(0) else True)
        pos_scores = pos_scores[pos_edge_idx]

        y_pred = torch.cat([pos_scores, neg_scores]).unsqueeze(-1).detach()
        y_true = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)]).unsqueeze(-1)

        metrics.update_metrics(y_pred, y_true, weights=None, subset=subset)

    def stack_dict_values(self, pos_scores: Dict[Tuple[str, str, str], Tensor],
                          neg_batch_scores: Dict[Tuple[str, str, str], Tensor],
                          neg_scores: Dict[Tuple[str, str, str], Tensor] = None) \
            -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        if neg_scores is None:
            neg_scores = {}

        metapaths = list(pos_scores | neg_batch_scores | neg_scores)

        pos_stack, neg_batch_stack, neg_stack = [], [], []
        for metapath in metapaths:
            pos_stack.append(pos_scores[metapath]) if metapath in pos_scores else None
            neg_batch_stack.append(neg_batch_scores[metapath]) if metapath in neg_batch_scores else None
            neg_stack.append(neg_scores[metapath]) if metapath in neg_scores else None

        pos_stack = torch.cat(pos_stack, dim=0) if pos_stack else torch.tensor([])
        neg_batch_stack = torch.cat(neg_batch_stack, dim=0) if neg_batch_stack else torch.tensor([])
        neg_stack = torch.cat(neg_stack, dim=0) if neg_stack else None

        return pos_stack, neg_batch_stack, neg_stack

    def reshape_edge_scores(self, pos_edge_score: Dict[Tuple[str, str, str], Tensor],
                            neg_edge_score: Dict[Tuple[str, str, str], Tensor]) \
            -> Tuple[Dict[Tuple[str, str, str], Tensor], Dict[Tuple[str, str, str], Tensor],
                     Dict[Tuple[str, str, str], Tensor]]:
        """
        Reshape edge scores from pos_graph and neg_graph by selecting positive, negative_batch and true negative
        metapaths into pos_scores, neg_batch_scores, neg_scores for contrastive loss function.

        Args:
            pos_edge_score (): dict of metapath keys and predicted edge values from pos_graph
            neg_edge_score (): dict of metapath keys and predicted edge values from neg_graph

        Returns:

        """
        pos_scores, neg_batch_scores, neg_scores = {}, {}, {}

        for metapath in pos_edge_score.keys():
            if metapath in self.pred_metapaths:
                num_pos_edges = pos_edge_score[metapath].size(0)
                num_neg_samples = math.ceil(neg_edge_score[metapath].numel() / num_pos_edges)

                pos_scores[metapath] = pos_edge_score[metapath]

                if neg_edge_score[metapath].numel() < num_pos_edges * num_neg_samples:
                    idx = torch.cat([
                        torch.arange(neg_edge_score[metapath].size(0)),
                        torch.randint(neg_edge_score[metapath].size(0),
                                      size=[num_pos_edges * num_neg_samples - neg_edge_score[metapath].size(0), ])])
                else:
                    idx = torch.arange(neg_edge_score[metapath].size(0))

                neg_batch_scores[metapath] = neg_edge_score[metapath][idx].view(num_pos_edges, num_neg_samples)

            if metapath in self.neg_pred_metapaths:
                neg_scores[metapath] = pos_edge_score[metapath]

        return pos_scores, neg_batch_scores, neg_scores

    def split_edge_scores(self, pos_edge_score: Dict[Tuple[str, str, str], Tensor],
                          neg_edge_score: Dict[Tuple[str, str, str], Tensor],
                          pos_graph: DGLHeteroGraph, neg_graph: DGLHeteroGraph,
                          nodes_namespace: Dict[str, np.ndarray]) \
            -> Tuple[Dict[Tuple[str, str, str], Tensor], Dict[Tuple[str, str, str], Tensor],
                     Dict[Tuple[str, str, str], Tensor]]:
        """

        Args:
            pos_edge_score ():
            neg_edge_score ():
            pos_graph ():
            neg_graph ():
            nodes_namespace:

        Returns:

        """
        pos_scores, neg_batch_scores, neg_scores = {}, {}, {}

        for metapath in pos_edge_score.keys():
            if metapath in self.pred_metapaths:
                pos_scores[metapath] = pos_edge_score[metapath]
                neg_batch_scores[metapath] = neg_edge_score[metapath]

            if metapath in self.neg_pred_metapaths:
                neg_scores[metapath] = pos_edge_score[metapath]

        if nodes_namespace:
            _, pos_scores = split_edge_index_by_namespace(nodes_namespace=nodes_namespace,
                                                          edge_index_dict=dgl_to_edge_index_dict(pos_graph,
                                                                                                 global_ids=True),
                                                          edge_values=pos_scores)
            _, neg_batch_scores = split_edge_index_by_namespace(nodes_namespace=nodes_namespace,
                                                                edge_index_dict=dgl_to_edge_index_dict(neg_graph,
                                                                                                       global_ids=True),
                                                                edge_values=neg_batch_scores)
            _, neg_scores = split_edge_index_by_namespace(nodes_namespace=nodes_namespace,
                                                          edge_index_dict=dgl_to_edge_index_dict(pos_graph,
                                                                                                 global_ids=True),
                                                          edge_values=neg_scores)

        neg_batch_scores = {
            m: scores[torch.arange(round_to_multiple(scores.numel(), multiple=pos_scores[m].size(0)))] \
                .view(pos_scores[m].size(0), -1).detach() \
                if scores.dim() != 2 else scores \
            for m, scores in neg_batch_scores.items()}

        return pos_scores, neg_batch_scores, neg_scores

    def training_step(self, batch, batch_nb):
        input_nodes, pos_graph, neg_graph, blocks = batch

        input_features = {ntype: feat for ntype, feat in blocks[0].srcdata["feat"]}

        pos_edge_scores, neg_edge_scores = self.forward(pos_graph, neg_graph, blocks, input_features)
        pos_stack, neg_batch_stack, neg_stack = self.stack_dict_values(*self.reshape_edge_scores(
            pos_edge_scores, neg_edge_scores))

        loss = self.criterion.forward(pos_stack, neg_batch_stack)

        pos_edge_scores, neg_batch_scores, neg_edge_scores = self.split_edge_scores(
            pos_edge_scores, neg_edge_scores, pos_graph=pos_graph, neg_graph=neg_graph,
            nodes_namespace=self.go_namespace)

        self.update_link_pred_metrics(self.train_metrics, pos_edge_scores=pos_edge_scores,
                                      neg_batch_scores=neg_batch_scores, neg_edge_scores=neg_edge_scores,
                                      pos_stack=pos_stack, neg_stack=neg_stack)

        self.log("loss", loss, logger=True, on_step=True)

        if batch_nb % 25 == 0 and isinstance(self.train_metrics, Metrics):
            logs = self.train_metrics.compute_metrics()
            self.log_dict(logs, prog_bar=True, logger=True, on_step=True)

        return loss

    def validation_step(self, batch, batch_nb):
        input_nodes, pos_graph, neg_graph, blocks = batch

        input_features = {ntype: feat for ntype, feat in blocks[0].srcdata["feat"]}

        pos_edge_scores, neg_edge_scores = self.forward(pos_graph, neg_graph, blocks, input_features)
        pos_stack, neg_batch_stack, neg_stack = self.stack_dict_values(*self.reshape_edge_scores(
            pos_edge_scores, neg_edge_scores))

        loss = self.criterion.forward(pos_stack, neg_batch_stack)

        pos_edge_scores, neg_batch_scores, neg_edge_scores = self.split_edge_scores(
            pos_edge_scores, neg_edge_scores, pos_graph=pos_graph, neg_graph=neg_graph,
            nodes_namespace=self.go_namespace)

        self.update_link_pred_metrics(self.valid_metrics, pos_edge_scores=pos_edge_scores,
                                      neg_batch_scores=neg_batch_scores, neg_edge_scores=neg_edge_scores,
                                      pos_stack=pos_stack, neg_stack=neg_stack)

        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_nb):
        input_nodes, pos_graph, neg_graph, blocks = batch

        input_features = {ntype: feat for ntype, feat in blocks[0].srcdata["feat"]}

        pos_edge_scores, neg_edge_scores = self.forward(pos_graph, neg_graph, blocks, input_features)
        pos_stack, neg_batch_stack, neg_stack = self.stack_dict_values(*self.reshape_edge_scores(
            pos_edge_scores, neg_edge_scores))

        loss = self.criterion.forward(pos_stack, neg_batch_stack)

        pos_edge_scores, neg_batch_scores, neg_edge_scores = self.split_edge_scores(
            pos_edge_scores, neg_edge_scores, pos_graph=pos_graph, neg_graph=neg_graph,
            nodes_namespace=self.go_namespace)

        self.update_link_pred_metrics(self.test_metrics, pos_edge_scores=pos_edge_scores,
                                      neg_batch_scores=neg_batch_scores, neg_edge_scores=neg_edge_scores,
                                      pos_stack=pos_stack, neg_stack=neg_stack)
        self.log("test_loss", loss, logger=True)
        return loss

    def train_dataloader(self):
        return self.dataset.train_dataloader(collate_fn=None, batch_size=self.hparams.batch_size, num_workers=0)

    def val_dataloader(self, batch_size=None):
        return self.dataset.valid_dataloader(collate_fn=None, batch_size=self.hparams.batch_size, num_workers=0)

    def test_dataloader(self, batch_size=None):
        return self.dataset.test_dataloader(collate_fn=None, batch_size=self.hparams.batch_size, num_workers=0)

    def on_validation_end(self) -> None:
        try:
            if self.current_epoch % 5 == 1:
                input_nodes, pos_graph, neg_graph, blocks = next(
                    iter(self.dataset.valid_dataloader(batch_size=self.hparams.batch_size, device=self.device,
                                                       verbose=False)))
                feats = {ntype: feat for ntype, feat in blocks[0].srcdata["feat"]}

                embeddings, pos_edge_scores, neg_edge_scores = self.forward(pos_graph, neg_graph, blocks, feats,
                                                                            return_embeddings=True)

                self.log_score_averages(edge_scores_dict=pos_edge_scores | tag_negative_metapath({
                    k: v for k, v in neg_edge_scores.items() if not is_negative(k)}))
        except Exception as e:
            traceback.print_exc()
        finally:
            self.plot_sankey_flow(layer=-1)
            super().on_validation_end()

    def on_test_end(self):
        try:
            if self.wandb_experiment is not None:
                input_nodes, pos_graph, neg_graph, blocks = next(iter(self.dataset.test_dataloader(batch_size=1000)))
                feats = {ntype: feat for ntype, feat in blocks[0].srcdata["feat"]}
                embeddings, pos_edge_scores, neg_edge_scores = self.cpu().forward(pos_graph, neg_graph, blocks, feats,
                                                                                  return_embeddings=True)
                global_node_index = {ntype: blocks[-1].dstnodes[ntype].data["_ID"] \
                                     for ntype in blocks[-1].ntypes if blocks[-1].num_dst_nodes(ntype)}

                self.plot_sankey_flow(layer=-1)

                self.log_score_averages(edge_scores_dict=pos_edge_scores | tag_negative_metapath({
                    k: v for k, v in neg_edge_scores.items() if not is_negative(k)}))

                edge_index_dict = dgl_to_edge_index_dict(
                    pos_graph, edge_values=pos_edge_scores, global_ids=False) | tag_negative_metapath({
                    k: v for k, v in dgl_to_edge_index_dict(
                        neg_graph, edge_values=neg_edge_scores, global_ids=False).items() \
                    if not is_negative(k)})
                self.plot_embeddings_tsne(global_node_index=global_node_index, embeddings=embeddings,
                                          targets=edge_index_dict, y_pred=edge_index_dict)
                self.cleanup_artifacts()

        except Exception as e:
            traceback.print_exc()

        finally:
            super().on_test_end()

    def get_edge_index_loss(self, edge_index_dict: Dict[Tuple[str, str, str], Tensor],
                            edge_index_scores: Dict[Tuple[str, str, str], Tensor],
                            global_node_index: Dict[str, Tensor] = None,
                            loss_fn: Callable = F.binary_cross_entropy) \
            -> Dict[Tuple[str, str, str], Tuple[Tensor, Tensor]]:

        e_pred, e_true = defaultdict(list), defaultdict(list)

        for metapath, (edge_index, edge_value) in edge_index_dict.items():
            head_type, tail_type = metapath[0], metapath[-1]
            edge_index = torch.stack(edge_index, dim=0) if isinstance(edge_index, tuple) else edge_index
            if global_node_index:
                edge_index[0] = global_node_index[head_type][edge_index[0]]
                edge_index[1] = global_node_index[tail_type][edge_index[1]]

            true_edge_score = torch.ones_like(edge_value.view(-1)) if is_negative(metapath) else \
                torch.ones_like(edge_value.view(-1))

            e_pred[metapath].append((edge_index, edge_value.view(-1).detach()))
            e_true[metapath].append((edge_index, true_edge_score))

        e_pred = {m: (torch.cat([eid for eid, v in li_eid_v], dim=1),
                      torch.cat([v for eid, v in li_eid_v], dim=0)) for m, li_eid_v in e_pred.items()}
        e_true = {m: (torch.cat([eid for eid, v in li_eid_v], dim=1),
                      torch.cat([v for eid, v in li_eid_v], dim=0)) for m, li_eid_v in e_true.items()}

        e_losses = {m: (eid, loss_fn(e_pred[m][1], target=e_true[m][1], reduce=False)) \
                    for m, (eid, _) in e_pred.items()}
        return e_losses

    def configure_optimizers(self):
        weight_decay = self.hparams.weight_decay if 'weight_decay' in self.hparams else 0.0
        optimizer = self.hparams.optimizer if 'optimizer' in self.hparams else "adam"
        lr_annealing = self.hparams.lr_annealing if "lr_annealing" in self.hparams else None

        if optimizer.lower() == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=weight_decay)
        elif optimizer.lower() == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"wrong value for optimizer {optimizer}!")

        extra = {}
        if lr_annealing == "cosine":
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                                       T_max=self.num_training_steps,
                                                       eta_min=self.lr / 100)

            extra = {"lr_scheduler": scheduler, "monitor": "val_loss"}
            print("Using CosineAnnealingLR", scheduler.state_dict())

        elif lr_annealing == "restart":
            scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=1,
                                                                 eta_min=self.lr / 100)
            extra = {"lr_scheduler": scheduler, "monitor": "val_loss"}
            print("Using CosineAnnealingWarmRestarts", scheduler.state_dict())

        elif lr_annealing == "reduce":
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer)
            extra = {"lr_scheduler": scheduler, "monitor": "val_loss"}
            print("Using ReduceLROnPlateau", scheduler.state_dict())

        return {"optimizer": optimizer, **extra}


class HGTLinkPred(DglLinkPredTrainer):
    def __init__(self, hparams: Union[Namespace, Dict], dataset: DGLLinkGenerator, metrics: List[str]):
        if not isinstance(hparams, Namespace) and isinstance(hparams, dict):
            hparams = Namespace(**hparams)
        super().__init__(hparams, dataset, metrics)
        self.dataset = dataset

        if "node_neighbors_min_num" in hparams:
            fanouts = [hparams["node_neighbors_min_num"], ] * len(dataset.neighbor_sizes)
            self.set_fanouts(self.dataset, fanouts)

        if len(dataset.node_attr_shape) == 0 or sum(dataset.node_attr_shape.values()) == 0:
            non_seq_ntypes = [ntype for ntype in dataset.node_types if ntype not in dataset.node_attr_shape]
            print("non_seq_ntypes", non_seq_ntypes)
            self.encoder = HeteroNodeFeatureEncoder(hparams, dataset, select_ntypes=non_seq_ntypes)

        self.embedder = HGT(node_dict={ntype: i for i, ntype in enumerate(dataset.node_types)},
                            edge_dict={metapath[1]: i for i, metapath in enumerate(dataset.get_metapaths())},
                            n_inp=self.dataset.node_attr_shape[self.head_node_type] \
                                if self.dataset.node_attr_shape else hparams.embedding_dim,
                            n_hid=hparams.embedding_dim,
                            n_out=hparams.embedding_dim,
                            n_layers=hparams.n_layers,
                            n_heads=hparams.attn_heads,
                            dropout=hparams.dropout,
                            use_norm=hparams.use_norm)

        self.classifier = MLPPredictor(in_dim=hparams.embedding_dim, loss_type=hparams.loss_type)
        self.criterion = ClassificationLoss(loss_type=hparams.loss_type, multilabel=False)

        hparams.n_params = self.get_n_params()
        print(f'Model #Params: {self.get_n_params()}')

        print(f'Configuration: {hparams}')
        self._set_hparams(hparams)

    def forward(self, pos_graph, neg_graph, blocks, x, return_embeddings=False) \
            -> Tuple[Dict[Tuple[str, str, str], Tensor], Dict[Tuple[str, str, str], Tensor]]:
        if len(x) == 0 or sum(a.numel() for a in x) == 0:
            x = self.encoder.forward(feats=x, global_node_index=blocks[0].srcdata["_ID"])

        x = self.embedder(blocks, x)

        pos_edge_score = self.classifier.forward(pos_graph, x)
        neg_edge_score = self.classifier.forward(neg_graph, x)

        if return_embeddings:
            return x, pos_edge_score, neg_edge_score
        return pos_edge_score, neg_edge_score


class LATTELinkPred(DglLinkPredTrainer):
    def __init__(self, hparams: Union[Namespace, Dict], dataset: DGLLinkGenerator, metrics: List[str]):
        if not isinstance(hparams, Namespace) and isinstance(hparams, dict):
            hparams = Namespace(**hparams)
        super().__init__(hparams, dataset, metrics)
        self.dataset = dataset

        if "node_neighbors_min_num" in hparams:
            fanouts = [hparams["node_neighbors_min_num"], ] * len(dataset.neighbor_sizes)
            self.set_fanouts(self.dataset, fanouts)

        if len(dataset.node_attr_shape) == 0 or sum(dataset.node_attr_shape.values()) == 0:
            non_seq_ntypes = [ntype for ntype in dataset.node_types if ntype not in dataset.node_attr_shape]
            print("non_seq_ntypes", non_seq_ntypes)
            self.encoder = HeteroNodeFeatureEncoder(hparams, dataset, select_ntypes=non_seq_ntypes)

        self.embedder = LATTE(t_order=hparams.t_order, embedding_dim=hparams.embedding_dim,
                              num_nodes_dict=dataset.num_nodes_dict, head_node_type=dataset.head_node_type,
                              metapaths=dataset.get_metapaths(), activation="relu", dropout=hparams.dropout,
                              attn_heads=hparams.attn_heads, attn_dropout=hparams.attn_dropout)

        self.classifier = MLPPredictor(in_dim=hparams.embedding_dim, loss_type=hparams.loss_type)
        self.criterion = ClassificationLoss(loss_type=hparams.loss_type, multilabel=False)

        hparams.n_params = self.get_n_params()
        print(f'Model #Params: {self.get_n_params()}')

        print(f'Configuration: {hparams}')
        self._set_hparams(hparams)

    def forward(self, pos_graph, neg_graph, blocks, x, return_embeddings=False, **kwargs) \
            -> Tuple[Dict[Tuple[str, str, str], Tensor], Dict[Tuple[str, str, str], Tensor]]:
        if len(x) == 0 or sum(a.numel() for a in x) == 0:
            x = self.encoder.forward(feats=x, global_node_index=blocks[0].srcdata["_ID"])

        x = self.embedder.forward(blocks, x, **kwargs)

        pos_edge_score = self.classifier.forward(pos_graph, x)
        neg_edge_score = self.classifier.forward(neg_graph, x)

        if return_embeddings:
            return x, pos_edge_score, neg_edge_score
        return pos_edge_score, neg_edge_score

    @property
    def metapaths(self) -> List[Tuple[str, str, str]]:
        return [layer.metapaths for layer in self.embedder.layers]

    @property
    def betas(self) -> List[Dict[str, DataFrame]]:
        return [layer._betas for layer in self.embedder.layers]

    @property
    def beta_avg(self) -> List[Dict[Tuple[str, str, str], float]]:
        return [layer._beta_avg for layer in self.embedder.layers]


class DeepGraphGOLinkPred(DglLinkPredTrainer):
    def __init__(self, hparams: Union[Namespace, Dict], dataset: DGLLinkGenerator, metrics: List[str]):
        if not isinstance(hparams, Namespace) and isinstance(hparams, dict):
            hparams = Namespace(**hparams)
        super().__init__(hparams, dataset, metrics)
        self.dataset = dataset

        if "node_neighbors_min_num" in hparams:
            fanouts = [hparams["node_neighbors_min_num"], ] * len(dataset.neighbor_sizes)
            self.set_fanouts(self.dataset, fanouts)

        if len(dataset.node_attr_shape) == 0 or sum(dataset.node_attr_shape.values()) == 0:
            non_seq_ntypes = [ntype for ntype in dataset.node_types if ntype not in dataset.node_attr_shape]
            print("non_seq_ntypes", non_seq_ntypes)
            self.encoder = HeteroNodeFeatureEncoder(hparams, dataset, select_ntypes=non_seq_ntypes)

        self.conv = HGT(node_dict={ntype: i for i, ntype in enumerate(dataset.node_types)},
                        edge_dict={metapath[1]: i for i, metapath in enumerate(dataset.get_metapaths())},
                        n_inp=list(self.dataset.node_attr_shape.values())[0] \
                            if self.dataset.node_attr_shape else hparams.embedding_dim,
                        n_hid=hparams.embedding_dim,
                        n_out=hparams.embedding_dim,
                        n_layers=hparams.n_layers,
                        n_heads=hparams.attn_heads,
                        dropout=hparams.dropout,
                        use_norm=hparams.use_norm)

        self.classifier = MLPPredictor(in_dim=hparams.embedding_dim, loss_type=hparams.loss_type)
        self.criterion = ClassificationLoss(loss_type=hparams.loss_type, multilabel=False)

        hparams.n_params = self.get_n_params()
        print(f'Model #Params: {self.get_n_params()}')

        print(f'Configuration: {hparams}')
        self._set_hparams(hparams)

    def forward(self, pos_graph, neg_graph, blocks, x, return_embeddings=False):
        if len(x) == 0 or sum(a.numel() for a in x) == 0:
            x = self.encoder.forward(feats=x, global_node_index=blocks[0].srcdata["_ID"])

        x = self.conv(blocks, x)

        pos_edge_score = self.classifier(pos_graph, x)
        neg_edge_score = self.classifier(neg_graph, x)
        if return_embeddings:
            return x, pos_edge_score, neg_edge_score
        return pos_edge_score, neg_edge_score


class DotPredictor(nn.Module):

    def __init__(self, loss_type: str) -> None:
        super().__init__()
        if loss_type == "CONTRASTIVE":
            self.activation = nn.Sigmoid()
            print("Using sigmoid activation for link pred scores")

    def forward(self, g, h: Tensor):
        with g.local_scope():
            g.ndata['h'] = h
            # Compute a new edge feature named 'score' by a dot-product between the
            # source node feature 'h' and destination node feature 'h'.
            g.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            scores = g.edata['score'][:, 0]
            if hasattr(self, "activation"):
                scores = self.activation(scores)
            return scores


class EdgePredictor(nn.Module):
    def __init__(self, loss_type: str) -> None:
        super().__init__()
        if loss_type == "CONTRASTIVE":
            self.activation = nn.Sigmoid()
            print("Using sigmoid activation for link pred scores")

    def forward(self, g: dgl.DGLHeteroGraph, h: Dict[str, Tensor]) -> Dict[Tuple[str, str, str], Tensor]:
        with g.local_scope():
            g.ndata['h'] = h
            for etype in g.canonical_etypes:
                if g.num_edges(etype=etype) == 0: continue
                g.apply_edges(
                    fn.u_dot_v('h', 'h', 'score'), etype=etype)

            if isinstance(g.edata['score'], dict):
                scores = {etype: scores[:, 0] for etype, scores in g.edata['score'].items() if scores.size(0)}
                if hasattr(self, "activation"):
                    scores = {etype: self.activation(scores) for etype, scores in scores.items()}
            else:
                scores = g.edata['score'][:, 0]
                if hasattr(self, "activation"):
                    scores = self.activation(scores)

            return scores


class MLPPredictor(nn.Module):
    def __init__(self, in_dim, loss_type):
        super().__init__()
        self.W1 = nn.Linear(in_dim * 2, in_dim)
        self.W2 = nn.Linear(in_dim, 1)

        if loss_type == "CONTRASTIVE":
            self.activation = nn.Sigmoid()
            print("Using sigmoid activation for link pred scores")

    def apply_edges(self, edges):
        h = torch.cat([edges.src['h'], edges.dst['h']], 1)
        return {'score': self.W2(F.relu(self.W1(h))).squeeze(1)}

    def forward(self, g: dgl.DGLHeteroGraph, h: Dict[str, Tensor]) -> Dict[Tuple[str, str, str], Tensor]:
        with g.local_scope():
            g.ndata['h'] = h
            for etype in g.canonical_etypes:
                if g.num_edges(etype=etype) == 0: continue
                g.apply_edges(self.apply_edges, etype=etype)

            if isinstance(g.edata['score'], dict):
                scores = {etype: scores for etype, scores in g.edata['score'].items() if scores.size(0)}
                if hasattr(self, "activation"):
                    scores = {etype: self.activation(scores) for etype, scores in scores.items()}
            else:
                scores = g.edata['score']
                if hasattr(self, "activation"):
                    scores = self.activation(scores)
            return scores
