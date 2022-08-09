from argparse import Namespace
from typing import Dict, List, Union, Tuple, Optional

import dgl
import dgl.function as fn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLHeteroGraph
from torch import Tensor

from moge.dataset import DGLLinkSampler
from moge.dataset.dgl.utils import dgl_to_edge_index_dict, round_to_multiple
from moge.model.dgl.HGT import HGT
from moge.model.encoder import HeteroNodeFeatureEncoder
from moge.model.losses import ClassificationLoss
from moge.model.metrics import Metrics
from moge.model.trainer import LinkPredTrainer


class DglLinkPredTrainer(LinkPredTrainer):
    def __init__(self, hparams, dataset: DGLLinkSampler, metrics: Union[List[str], Dict[str, List[str]]], *args,
                 **kwargs):
        super().__init__(hparams, dataset, metrics, *args, **kwargs)
        self.dataset: DGLLinkSampler

        self.pred_metapaths = dataset.pred_metapaths
        self.neg_pred_metapaths = dataset.neg_pred_metapaths

    def update_link_pred_metrics(self, metrics: Union[Metrics, Dict[str, Metrics]],
                                 pos_edge_score: Dict[Tuple[str, str, str], Tensor],
                                 neg_batch_scores: Dict[Tuple[str, str, str], Tensor],
                                 neg_edge_score: Dict[Tuple[str, str, str], Tensor],
                                 pos_scores: Tensor = None,
                                 neg_scores: Tensor = None):

        if isinstance(metrics, dict):
            for metapath in pos_edge_score:
                go_type = "BPO" if metapath[-1] == 'biological_process' else \
                    "CCO" if metapath[-1] == 'cellular_component' else \
                        "MFO" if metapath[-1] == 'molecular_function' else None

                metrics[go_type].update_metrics(pos_edge_score[metapath].detach(), neg_batch_scores[metapath].detach(),
                                                weights=None, subset=["ogbl-biokg"])

                if metapath in pos_edge_score and neg_edge_score and metapath in neg_edge_score:
                    self.update_pr_metrics(pos_scores=pos_edge_score[metapath],
                                           neg_scores=neg_edge_score[metapath],
                                           metrics=metrics[go_type])

        elif pos_scores is not None and neg_scores is not None:
            metapath = list(pos_edge_score.keys())[0]
            metrics.update_metrics(pos_edge_score[metapath].detach(), neg_edge_score[metapath].detach(),
                                   weights=None, subset=["ogbl-biokg"])
            self.update_pr_metrics(pos_scores=pos_scores, neg_scores=neg_scores, metrics=metrics)

    def update_pr_metrics(self, pos_scores: Tensor,
                          neg_scores: Tensor,
                          metrics: Metrics,
                          subset=["precision", "recall", "aupr"]):
        # randomly select |e_neg| positive edges to balance precision/recall scores
        pos_edge_idx = np.random.choice(pos_scores.size(0), size=neg_scores.shape,
                                        replace=False if pos_scores.size(0) > neg_scores.size(0) else True)
        pos_scores = pos_scores[pos_edge_idx]

        y_pred = torch.cat([pos_scores, neg_scores]).unsqueeze(-1).detach()
        y_true = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)]).unsqueeze(-1)

        metrics.update_metrics(y_pred, y_true, weights=None, subset=subset)

    def split_by_namespace(self, edge_index_dict: Dict[Tuple[str, str, str], Tensor],
                           edge_values: Dict[Tuple[str, str, str], Tensor] = None, ) \
            -> Tuple[Dict[Tuple[str, str, str], Tuple[Tensor, Tensor]], Dict[Tuple[str, str, str], Tensor]]:
        if not hasattr(self.dataset, "go_namespace") or not hasattr(self.dataset, "ntype_mapping"):
            return edge_index_dict, edge_values
        else:
            go_namespace: np.ndarray = self.dataset.go_namespace

        split_edge_index_dict = {}
        split_edge_values = {}

        for metapath, (u, v) in edge_index_dict.items():
            if edge_values and metapath not in edge_values: continue

            # Pos or neg edges
            for namespace in np.unique(go_namespace):
                mask = go_namespace[v.detach().cpu().numpy()] == namespace
                new_metapath = metapath[:-1] + (namespace,)

                if not isinstance(mask, bool) and mask.sum():
                    split_edge_index_dict[new_metapath] = (u[mask], v[mask])
                    if edge_values:
                        split_edge_values[new_metapath] = edge_values[metapath].view(-1)[mask]
                elif isinstance(mask, bool):
                    split_edge_index_dict[new_metapath] = (u, v)
                    if edge_values:
                        split_edge_values[new_metapath] = edge_values[metapath].view(-1)

        return split_edge_index_dict, split_edge_values if split_edge_values else None

    def split_edge_scores(self, pos_edge_score: Dict[Tuple[str, str, str], Tensor],
                          neg_edge_score: Dict[Tuple[str, str, str], Tensor],
                          pos_graph: DGLHeteroGraph = None, neg_graph: DGLHeteroGraph = None):
        pos_scores, neg_batch_scores, neg_scores = {}, {}, {}

        for metapath in pos_edge_score.keys():
            if metapath in self.pred_metapaths:
                num_pos_edges = pos_edge_score[metapath].size(0)
                num_neg_samples = neg_edge_score[metapath].numel() // num_pos_edges

                pos_scores[metapath] = pos_edge_score[metapath]
                neg_batch_scores[metapath] = neg_edge_score[metapath].view(num_pos_edges, num_neg_samples)

            elif metapath in self.neg_pred_metapaths:
                neg_scores[metapath] = pos_edge_score[metapath]

        if pos_graph is not None and neg_graph is not None:
            _, pos_scores = self.split_by_namespace(dgl_to_edge_index_dict(pos_graph, global_ids=True),
                                                    edge_values=pos_scores)
            _, neg_batch_scores = self.split_by_namespace(dgl_to_edge_index_dict(neg_graph, global_ids=True),
                                                          edge_values=neg_batch_scores)
            _, neg_scores = self.split_by_namespace(dgl_to_edge_index_dict(pos_graph, global_ids=True),
                                                    edge_values=neg_scores)

            neg_batch_scores = {
                m: scores[torch.arange(round_to_multiple(scores.numel(), pos_scores[m].size(0)))] \
                    .view(pos_scores[m].size(0), -1).detach() \
                    if scores.dim() != 2 else scores \
                for m, scores in neg_batch_scores.items()}

        return pos_scores, neg_batch_scores, neg_scores

    def stack_edge_scores(self, pos_scores: Dict[Tuple[str, str, str], Tensor],
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

    def training_step(self, batch, batch_nb):
        input_nodes, pos_graph, neg_graph, blocks = batch

        input_features = {ntype: feat for ntype, feat in blocks[0].srcdata["feat"]}

        pos_edge_scores, neg_edge_scores = self.forward(pos_graph, neg_graph, blocks, input_features)
        pos_stack, neg_batch_stack, neg_stack = self.stack_edge_scores(
            *self.split_edge_scores(pos_edge_scores, neg_edge_scores))

        loss = self.criterion.forward(pos_stack, neg_batch_stack)

        pos_edge_scores, neg_batch_scores, neg_edge_scores = self.split_edge_scores(
            pos_edge_scores, neg_edge_scores, pos_graph=pos_graph, neg_graph=neg_graph)

        self.update_link_pred_metrics(self.train_metrics,
                                      pos_edge_score=pos_edge_scores,
                                      neg_batch_scores=neg_batch_scores,
                                      neg_edge_score=neg_edge_scores,
                                      pos_scores=pos_stack,
                                      neg_scores=neg_stack)

        self.log("loss", loss, logger=True, on_step=True)
        if batch_nb % 25 == 0 and isinstance(self.train_metrics, Metrics):
            logs = self.train_metrics.compute_metrics()
            self.log_dict(logs, prog_bar=True, logger=True, on_step=True)

        return loss

    def validation_step(self, batch, batch_nb):
        input_nodes, pos_graph, neg_graph, blocks = batch

        input_features = {ntype: feat for ntype, feat in blocks[0].srcdata["feat"]}

        pos_edge_scores, neg_edge_scores = self.forward(pos_graph, neg_graph, blocks, input_features)
        pos_stack, neg_batch_stack, neg_stack = self.stack_edge_scores(
            *self.split_edge_scores(pos_edge_scores, neg_edge_scores))

        loss = self.criterion.forward(pos_stack, neg_batch_stack)

        pos_edge_scores, neg_batch_scores, neg_edge_scores = self.split_edge_scores(
            pos_edge_scores, neg_edge_scores, pos_graph=pos_graph, neg_graph=neg_graph)

        self.update_link_pred_metrics(self.valid_metrics,
                                      pos_edge_score=pos_edge_scores,
                                      neg_batch_scores=neg_batch_scores,
                                      neg_edge_score=neg_edge_scores,
                                      pos_scores=pos_stack,
                                      neg_scores=neg_stack)

        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_nb):
        input_nodes, pos_graph, neg_graph, blocks = batch

        input_features = {ntype: feat for ntype, feat in blocks[0].srcdata["feat"]}

        pos_edge_scores, neg_edge_scores = self.forward(pos_graph, neg_graph, blocks, input_features)
        pos_stack, neg_batch_stack, neg_stack = self.stack_edge_scores(
            *self.split_edge_scores(pos_edge_scores, neg_edge_scores))

        loss = self.criterion.forward(pos_stack, neg_batch_stack)

        pos_edge_scores, neg_batch_scores, neg_edge_scores = self.split_edge_scores(
            pos_edge_scores, neg_edge_scores, pos_graph=pos_graph, neg_graph=neg_graph)

        self.update_link_pred_metrics(self.test_metrics,
                                      pos_edge_score=pos_edge_scores,
                                      neg_batch_scores=neg_batch_scores,
                                      neg_edge_score=neg_edge_scores,
                                      pos_scores=pos_stack,
                                      neg_scores=neg_stack)
        self.log("test_loss", loss, logger=True)
        return loss

    def train_dataloader(self):
        return self.dataset.train_dataloader(collate_fn=None, batch_size=self.hparams.batch_size, num_workers=0)

    def val_dataloader(self, batch_size=None):
        return self.dataset.valid_dataloader(collate_fn=None, batch_size=self.hparams.batch_size, num_workers=0)

    def test_dataloader(self, batch_size=None):
        return self.dataset.test_dataloader(collate_fn=None, batch_size=self.hparams.batch_size, num_workers=0)

    def configure_optimizers(self):
        if not hasattr(self.hparams, "optimizer") or self.hparams['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams['lr'],
                                         weight_decay=self.hparams['weight_decay'])
        elif self.hparams['optimizer'] == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams['lr'],
                                        weight_decay=self.hparams['weight_decay'])
        else:
            raise ValueError(f"wrong value for optimizer {self.hparams['optimizer']}!")

        return {"optimizer": optimizer}


class HGTLinkPred(DglLinkPredTrainer):
    def __init__(self, args: Union[Namespace, Dict], dataset: DGLLinkSampler, metrics: List[str]):
        if not isinstance(args, Namespace) and isinstance(args, dict):
            args = Namespace(**args)
        super().__init__(args, dataset, metrics)
        self.dataset = dataset

        if "node_neighbors_min_num" in args:
            fanouts = [args["node_neighbors_min_num"], ] * len(dataset.neighbor_sizes)
            self.set_fanouts(self.dataset, fanouts)

        if len(dataset.node_attr_shape) == 0 or sum(dataset.node_attr_shape.values()) == 0:
            non_seq_ntypes = [ntype for ntype in dataset.node_types if ntype not in dataset.node_attr_shape]
            print("non_seq_ntypes", non_seq_ntypes)
            self.encoder = HeteroNodeFeatureEncoder(args, dataset, select_ntypes=non_seq_ntypes)

        self.conv = HGT(node_dict={ntype: i for i, ntype in enumerate(dataset.node_types)},
                        edge_dict={metapath[1]: i for i, metapath in enumerate(dataset.get_metapaths())},
                        n_inp=list(self.dataset.node_attr_shape.values())[0] \
                            if self.dataset.node_attr_shape else args.embedding_dim,
                        n_hid=args.embedding_dim,
                        n_out=args.embedding_dim,
                        n_layers=args.n_layers,
                        n_heads=args.attn_heads,
                        dropout=args.dropout,
                        use_norm=args.use_norm)

        self.classifier = MLPPredictor(in_dim=args.embedding_dim, loss_type=args.loss_type)
        self.criterion = ClassificationLoss(loss_type=args.loss_type, multilabel=False)

        args.n_params = self.get_n_params()
        print(f'Model #Params: {self.get_n_params()}')

        print(f'Configuration: {args}')
        self._set_hparams(args)

    def forward(self, pos_graph, neg_graph, blocks, x, return_embeddings=False) \
            -> Tuple[Dict[Tuple[str, str, str], Tensor], Dict[Tuple[str, str, str], Tensor]]:
        if len(x) == 0 or sum(a.numel() for a in x) == 0:
            x = self.encoder.forward(node_feats=x, global_node_idx=blocks[0].srcdata["_ID"])

        x = self.conv(blocks, x)

        pos_edge_score = self.classifier.forward(pos_graph, x)
        neg_edge_score = self.classifier.forward(neg_graph, x)

        if return_embeddings:
            return x, pos_edge_score, neg_edge_score
        return pos_edge_score, neg_edge_score


class DeepGraphGOLinkPred(DglLinkPredTrainer):
    def __init__(self, args: Union[Namespace, Dict], dataset: DGLLinkSampler, metrics: List[str]):
        if not isinstance(args, Namespace) and isinstance(args, dict):
            args = Namespace(**args)
        super().__init__(args, dataset, metrics)
        self.dataset = dataset

        if "node_neighbors_min_num" in args:
            fanouts = [args["node_neighbors_min_num"], ] * len(dataset.neighbor_sizes)
            self.set_fanouts(self.dataset, fanouts)

        if len(dataset.node_attr_shape) == 0 or sum(dataset.node_attr_shape.values()) == 0:
            non_seq_ntypes = [ntype for ntype in dataset.node_types if ntype not in dataset.node_attr_shape]
            print("non_seq_ntypes", non_seq_ntypes)
            self.encoder = HeteroNodeFeatureEncoder(args, dataset, select_ntypes=non_seq_ntypes)

        self.conv = HGT(node_dict={ntype: i for i, ntype in enumerate(dataset.node_types)},
                        edge_dict={metapath[1]: i for i, metapath in enumerate(dataset.get_metapaths())},
                        n_inp=list(self.dataset.node_attr_shape.values())[0] \
                            if self.dataset.node_attr_shape else args.embedding_dim,
                        n_hid=args.embedding_dim,
                        n_out=args.embedding_dim,
                        n_layers=args.n_layers,
                        n_heads=args.attn_heads,
                        dropout=args.dropout,
                        use_norm=args.use_norm)

        self.classifier = MLPPredictor(in_dim=args.embedding_dim, loss_type=args.loss_type)
        self.criterion = ClassificationLoss(loss_type=args.loss_type, multilabel=False)

        args.n_params = self.get_n_params()
        print(f'Model #Params: {self.get_n_params()}')

        print(f'Configuration: {args}')
        self._set_hparams(args)

    def forward(self, pos_graph, neg_graph, blocks, x, return_embeddings=False):
        if len(x) == 0 or sum(a.numel() for a in x) == 0:
            x = self.encoder.forward(node_feats=x, global_node_idx=blocks[0].srcdata["_ID"])

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
