from argparse import Namespace
from argparse import Namespace
from pprint import pprint
from typing import Dict, List, Union, Tuple

import dgl
import dgl.function as fn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from moge.dataset import DGLLinkSampler
from moge.model.dgl.HGT import HGT
from moge.model.encoder import HeteroNodeFeatureEncoder
from moge.model.losses import ClassificationLoss
from moge.model.metrics import Metrics
from moge.model.trainer import LinkPredTrainer
from moge.model.utils import tensor_sizes


class DglLinkPredTrainer(LinkPredTrainer):
    def stack_edge_index_values(self, pos_edge_score: Dict[Tuple[str, str, str], Tensor],
                                neg_edge_score: Dict[Tuple[str, str, str], Tensor] = None):
        pos_stack, neg_stack = [], []
        for metapath in pos_edge_score.keys():
            num_pos_edges = pos_edge_score[metapath].size(0)
            num_neg_samples = neg_edge_score[metapath].size(0) // num_pos_edges

            pos_stack.append(pos_edge_score[metapath])
            neg_stack.append(neg_edge_score[metapath].view(num_pos_edges, num_neg_samples))

        pos_scores = torch.cat(pos_stack, dim=0)
        neg_batch_scores = torch.cat(neg_stack, dim=0)
        return pos_scores, neg_batch_scores

    def update_pr_metrics(self, pos_scores: Tensor,
                          neg_scores: Union[Tensor, Dict[Tuple[str, str, str], Tensor]],
                          metrics: Metrics,
                          subset=["precision", "recall", "aupr"]):
        if isinstance(neg_scores, dict):
            edge_neg_score = torch.cat([edge_scores.detach() for m, edge_scores in neg_scores.items()])
        else:
            edge_neg_score = neg_scores

        # randomly select |e_neg| positive edges to balance precision/recall scores
        pos_edge_idx = np.random.choice(pos_scores.size(0), size=edge_neg_score.shape,
                                        replace=False if pos_scores.size(0) > edge_neg_score.size(0) else True)
        pos_scores = pos_scores[pos_edge_idx]

        y_pred = torch.cat([pos_scores, edge_neg_score]).unsqueeze(-1).detach()
        y_true = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(edge_neg_score)]).unsqueeze(-1)

        metrics.update_metrics(y_pred, y_true, weights=None, subset=subset)

    def update_link_pred_metrics(self, metrics: Union[Metrics, Dict[str, Metrics]],
                                 pos_edge_score: Dict[Tuple[str, str, str], Tensor],
                                 neg_edge_score: Dict[Tuple[str, str, str], Tensor],
                                 neg_head_batch: Dict[Tuple[str, str, str], Tensor] = None,
                                 neg_tail_batch: Dict[Tuple[str, str, str], Tensor] = None,
                                 pos_scores: Tensor = None,
                                 neg_scores: Tensor = None):

        if neg_head_batch is None and neg_tail_batch is None:
            neg_head_batch, neg_tail_batch = {}, {}

            for metapath in pos_edge_score.keys():
                if neg_edge_score[metapath].numel() < pos_edge_score[metapath].numel(): continue
                num_pos_edges = pos_edge_score[metapath].size(0)
                num_neg_samples = neg_edge_score[metapath].size(0) // num_pos_edges

                neg_head_batch[metapath], neg_tail_batch[metapath] = neg_edge_score[metapath] \
                    .view(num_pos_edges, num_neg_samples) \
                    .split(num_neg_samples // 2, dim=1)

        if isinstance(metrics, dict):
            for metapath in pos_edge_score:
                go_type = "BPO" if metapath[-1] == 'biological_process' else \
                    "CCO" if metapath[-1] == 'cellular_component' else \
                        "MFO" if metapath[-1] == 'molecular_function' else None

                neg_batch = torch.concat([neg_head_batch[metapath], neg_tail_batch[metapath]], dim=1)
                metrics[go_type].update_metrics(pos_edge_score[metapath].detach(), neg_batch.detach(),
                                                weights=None, subset=["ogbl-biokg"])

                if metapath in pos_edge_score and neg_edge_score and metapath in neg_edge_score:
                    self.update_pr_metrics(pos_scores=pos_edge_score[metapath],
                                           neg_scores=neg_edge_score[metapath],
                                           metrics=metrics[go_type])

        elif pos_scores is not None and neg_scores is not None:
            metrics.update_metrics(pos_scores.detach(), neg_scores.detach(),
                                   weights=None, subset=["ogbl-biokg"])
            self.update_pr_metrics(pos_scores=pos_scores, neg_scores=neg_edge_score, metrics=metrics)


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
                        use_norm=args.use_norm)

        self.classifier = EdgePredictor(loss_type=args.loss_type)
        self.criterion = ClassificationLoss(loss_type=args.loss_type, multilabel=False)

        args.n_params = self.get_n_params()
        print(f'Model #Params: {self.get_n_params()}')

        print(f'Configuration: {args}')
        self._set_hparams(args)

    def forward(self, pos_graph, neg_graph, blocks, x):
        if len(x) == 0 or sum(a.numel() for a in x) == 0:
            x = self.encoder.forward(node_feats=x, global_node_idx=blocks[0].srcdata["_ID"])

        x = self.conv(blocks, x)
        pos_edge_score = self.classifier(pos_graph, x)
        neg_edge_score = self.classifier(neg_graph, x)

        return pos_edge_score, neg_edge_score

    def training_step(self, batch, batch_nb):
        input_nodes, pos_graph, neg_graph, blocks = batch
        input_features = {ntype: feat for ntype, feat in blocks[0].srcdata["feat"]}
        pprint(tensor_sizes({"input_nodes": input_nodes, "pos_graph": pos_graph, "neg_graph": neg_graph}))

        # for i, block in enumerate(blocks):
        #     blocks[i] = block.to(self.device)
        pos_edge_scores, neg_edge_scores = self.forward(pos_graph, neg_graph, blocks, input_features)
        pos_scores, neg_batch_scores = self.stack_edge_index_values(pos_edge_scores, neg_edge_scores)
        loss = self.criterion.forward(pos_scores, neg_batch_scores)

        neg_scores = neg_batch_scores[:, torch.randint(neg_batch_scores.size(1), size=[1]).item()]
        self.update_link_pred_metrics(self.train_metrics,
                                      pos_edge_score=pos_edge_scores,
                                      neg_edge_score=neg_edge_scores,
                                      pos_scores=pos_scores,
                                      neg_scores=neg_scores)

        self.log("loss", loss, logger=True, on_step=True)
        if batch_nb % 25 == 0:
            logs = self.train_metrics.compute_metrics()
            self.log_dict(logs, prog_bar=True, logger=True, on_step=True)
        print("trainloss", loss)
        loss.backward()
        return loss

    def validation_step(self, batch, batch_nb):
        input_nodes, pos_graph, neg_graph, blocks = batch

        input_features = {ntype: feat for ntype, feat in blocks[0].srcdata["feat"]}

        pos_edge_scores, neg_edge_scores = self.forward(pos_graph, neg_graph, blocks, input_features)
        pos_scores, neg_batch_scores = self.stack_edge_index_values(pos_edge_scores, neg_edge_scores)
        loss = self.criterion.forward(pos_scores, neg_batch_scores)

        neg_scores = neg_batch_scores[:, torch.randint(neg_batch_scores.size(1), size=[1]).item()]
        self.update_link_pred_metrics(self.valid_metrics,
                                      pos_edge_score=pos_edge_scores,
                                      neg_edge_score=neg_edge_scores,
                                      pos_scores=pos_scores,
                                      neg_scores=neg_scores)
        print("loss", loss)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_nb):
        input_nodes, pos_graph, neg_graph, blocks = batch
        input_features = {ntype: feat for ntype, feat in blocks[0].srcdata["feat"]}

        # for i, block in enumerate(blocks):
        #     blocks[i] = block.to(self.device)

        pos_edge_scores, neg_edge_scores = self.forward(pos_graph, neg_graph, blocks, input_features)
        pos_scores, neg_batch_scores = self.stack_edge_index_values(pos_edge_scores, neg_edge_scores)
        loss = self.criterion.forward(pos_scores, neg_batch_scores)

        neg_scores = neg_batch_scores[:, torch.randint(neg_batch_scores.size(1), size=[1]).item()]
        self.update_link_pred_metrics(self.test_metrics,
                                      pos_edge_score=pos_edge_scores,
                                      neg_edge_score=neg_edge_scores,
                                      pos_scores=pos_scores,
                                      neg_scores=neg_scores)
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


class DotPredictor(nn.Module):

    def __init__(self, loss_type: str) -> None:
        super().__init__()
        if loss_type == "CONTRASTIVE":
            self.activation = torch.sigmoid
            print("Using sigmoid activation for link pred scores")

    def forward(self, g, h: Tensor):
        with g.local_scope():
            g.ndata['h'] = h
            # Compute a new edge feature named 'score' by a dot-product between the
            # source node feature 'h' and destination node feature 'h'.
            g.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            return g.edata['score'][:, 0]


class EdgePredictor(nn.Module):
    def __init__(self, loss_type: str) -> None:
        super().__init__()
        if loss_type == "CONTRASTIVE":
            self.activation = torch.sigmoid
            print("Using sigmoid activation for link pred scores")

    def forward(self, g: dgl.DGLHeteroGraph, h: Dict[str, Tensor]) -> Dict[Tuple[str, str, str], Tensor]:
        with g.local_scope():
            g.ndata['h'] = h
            for etype in g.canonical_etypes:
                g.apply_edges(
                    fn.u_dot_v('h', 'h', 'score'), etype=etype)

            scores_dict = {etype: scores[:, 0] for etype, scores in g.edata['score'].items() if scores.size(0)}
            if hasattr(self, "activation"):
                scores_dict = {etype: self.activation(scores) for etype, scores in scores_dict.items()}
            return scores_dict


class MLPPredictor(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.W1 = nn.Linear(in_dim * 2, in_dim)
        self.W2 = nn.Linear(in_dim, 1)

    def apply_edges(self, edges):
        h = torch.cat([edges.src['h'], edges.dst['h']], 1)
        return {'score': self.W2(F.relu(self.W1(h))).squeeze(1)}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)
            return g.edata['score']
