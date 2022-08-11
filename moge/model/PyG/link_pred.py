import logging
import math
import traceback
from argparse import Namespace
from collections import defaultdict
from typing import List, Tuple, Dict, Any, Union, Optional, Callable

import numpy as np
import torch
from fairscale.nn import auto_wrap
from logzero import logger
from pandas import DataFrame
from torch import nn, Tensor
from torch.nn import functional as F
from torch.optim import lr_scheduler

from moge.model.PyG.latte_flat import LATTE as LATTEFlat
from moge.model.losses import ClassificationLoss
from .conv import HGT
from .utils import get_edge_index_from_neg_batch, batch2global_edge_index
from ..encoder import HeteroSequenceEncoder, HeteroNodeFeatureEncoder
from ..metrics import Metrics
from ..trainer import LinkPredTrainer
from ...dataset import HeteroLinkPredDataset


class PyGLinkPredTrainer(LinkPredTrainer):
    def stack_edge_index_values(self,
                                edge_pos: Dict[Tuple[str, str, str], Tensor],
                                neg_head_batch: Dict[Tuple[str, str, str], Tensor],
                                neg_tail_batch: Dict[Tuple[str, str, str], Tensor],
                                edge_neg: Dict[Tuple[str, str, str], Tensor] = None,
                                activation: Optional[Callable] = None) -> \
            Tuple[Tensor, Tensor, Optional[Tensor]]:
        pos_scores = []
        neg_batch_scores = []
        neg_scores = []

        for metapath, edge_pos_score in edge_pos.items():
            num_edges = edge_pos_score.shape[0]

            # Positive edges
            pos_scores.append(edge_pos_score)

            # Negative sampling
            if neg_head_batch or neg_tail_batch:
                e_pred_head = neg_head_batch[metapath].view(num_edges, -1)
                e_pred_tail = neg_tail_batch[metapath].view(num_edges, -1)
                neg_batch_scores.append(torch.cat([e_pred_head, e_pred_tail], dim=1))

            # True negatives
            if edge_neg:
                neg_scores.append(edge_neg[metapath])

        pos_scores = torch.cat(pos_scores, dim=0)
        neg_batch_scores = torch.cat(neg_batch_scores, dim=0)
        neg_scores = torch.cat(neg_scores, dim=0) if neg_scores else None

        if callable(activation):
            pos_scores = activation(pos_scores)
            neg_batch_scores = activation(neg_batch_scores)
            neg_scores = activation(neg_scores) if neg_scores is not None else neg_scores

        return pos_scores, neg_batch_scores, neg_scores

    def update_link_pred_metrics(self, metrics: Union[Metrics, Dict[str, Metrics]],
                                 pos_edge_score: Dict[Tuple[str, str, str], Tensor],
                                 neg_edge_score: Dict[Tuple[str, str, str], Tensor],
                                 neg_head_batch: Dict[Tuple[str, str, str], Tensor] = None,
                                 neg_tail_batch: Dict[Tuple[str, str, str], Tensor] = None,
                                 pos_scores: Tensor = None,
                                 neg_scores: Tensor = None):

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
            metapath = list(pos_edge_score.keys())[0]
            neg_batch = torch.concat([neg_head_batch[metapath], neg_tail_batch[metapath]], dim=1)
            metrics.update_metrics(pos_edge_score[metapath].detach(), neg_batch.detach(),
                                   weights=None, subset=["ogbl-biokg"])
            self.update_pr_metrics(pos_scores=pos_scores, neg_scores=neg_scores, metrics=metrics)

    def update_pr_metrics(self, pos_scores: Tensor,
                          neg_scores: Union[Tensor, Dict[Tuple[str, str, str], Tuple]],
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

    def get_edge_index_loss(self, edge_pred: Dict[str, Dict[Tuple[str, str, str], Tensor]],
                            edge_true: Dict[str, Dict[Tuple[str, str, str], Tensor]],
                            global_node_index: Dict[str, Tensor],
                            loss_fn: Callable = F.binary_cross_entropy) \
            -> Dict[Tuple[str, str, str], Tuple[Tensor, Tensor]]:

        e_pred, e_true = defaultdict(list), defaultdict(list)

        ntype_mapping = self.dataset.ntype_mapping if hasattr(self.dataset, "ntype_mapping") else None
        for metapath, edge_pos_pred in edge_pred["edge_pos"].items():
            corrected_metapath = tuple(ntype_mapping[i] if i in ntype_mapping else i for i in metapath) \
                if ntype_mapping else metapath
            edge_index = batch2global_edge_index(edge_true["edge_pos"][metapath], metapath=corrected_metapath,
                                                 global_node_index=global_node_index)
            e_pred[corrected_metapath].append((edge_index, edge_pos_pred.view(-1).detach()))
            e_true[corrected_metapath].append((edge_index, torch.ones_like(edge_pos_pred.view(-1))))

        for metapath, edge_neg_pred in edge_pred["edge_neg"].items():
            corrected_metapath = tuple(ntype_mapping[i] if i in ntype_mapping else i for i in metapath) \
                if ntype_mapping else metapath
            edge_index = batch2global_edge_index(edge_true["edge_neg"][metapath], metapath=corrected_metapath,
                                                 global_node_index=global_node_index)
            e_pred[corrected_metapath].append((edge_index, edge_neg_pred.view(-1).detach()))
            e_true[corrected_metapath].append((edge_index, torch.zeros_like(edge_neg_pred.view(-1))))

        head_batch_edge_index_dict, _ = get_edge_index_from_neg_batch(edge_true["head_batch"],
                                                                      edge_pos=edge_true["edge_pos"], mode="head_batch")
        for metapath, edge_neg_pred in edge_pred["head_batch"].items():
            corrected_metapath = tuple(ntype_mapping[i] if i in ntype_mapping else i for i in metapath) \
                if ntype_mapping else metapath
            edge_index = batch2global_edge_index(head_batch_edge_index_dict[metapath], metapath=corrected_metapath,
                                                 global_node_index=global_node_index)
            e_pred[corrected_metapath].append((edge_index, edge_neg_pred.view(-1).detach()))
            e_true[corrected_metapath].append((edge_index, torch.zeros_like(edge_neg_pred.view(-1))))
        tail_batch_edge_index_dict, _ = get_edge_index_from_neg_batch(edge_true["tail_batch"],
                                                                      edge_pos=edge_true["edge_pos"], mode="tail_batch")

        for metapath, edge_neg_pred in edge_pred["tail_batch"].items():
            corrected_metapath = tuple(ntype_mapping[i] if i in ntype_mapping else i for i in metapath) \
                if ntype_mapping else metapath
            edge_index = batch2global_edge_index(tail_batch_edge_index_dict[metapath], metapath=corrected_metapath,
                                                 global_node_index=global_node_index)
            e_pred[corrected_metapath].append((edge_index, edge_neg_pred.view(-1).detach()))
            e_true[corrected_metapath].append((edge_index, torch.zeros_like(edge_neg_pred.view(-1))))

        e_pred = {m: (torch.cat([eid for eid, v in li_eid_v], dim=1),
                      torch.cat([v for eid, v in li_eid_v], dim=0)) for m, li_eid_v in e_pred.items()}
        e_true = {m: (torch.cat([eid for eid, v in li_eid_v], dim=1),
                      torch.cat([v for eid, v in li_eid_v], dim=0)) for m, li_eid_v in e_true.items()}

        e_losses = {m: (eid, loss_fn(e_pred[m][1], target=e_true[m][1], reduce=False)) \
                    for m, (eid, _) in e_pred.items()}

        return e_losses

    def on_validation_end(self) -> None:
        try:
            if self.current_epoch % 5 == 1:
                X, e_true, _ = self.dataset.full_batch(
                    edge_idx=np.random.choice(self.dataset.validation_idx, size=10, replace=False), device=self.device)
                embeddings, e_pred = self.forward(X, e_true, save_betas=True)

                self.log_beta_degree_correlation(global_node_index=X["global_node_index"], batch_size=X["batch_size"])
                self.log_score_averages(edge_scores_dict=e_pred)
        except Exception as e:
            traceback.print_exc()
        finally:
            self.plot_sankey_flow(layer=-1, width=max(250 * self.embedder.t_order, 500))
            super().on_validation_end()

    def on_test_end(self):
        try:
            if self.wandb_experiment is not None:
                X, e_true, _ = self.dataset.full_batch(device="cpu")
                embeddings, e_pred = self.cpu().forward(X, e_true, save_betas=True)

                self.log_score_averages(edge_scores_dict=e_pred)
                self.plot_sankey_flow(layer=-1, width=max(250 * self.embedder.t_order, 500))
                self.plot_embeddings_tsne(global_node_index=X['global_node_index'], embeddings=embeddings,
                                          targets=e_true, y_pred=e_pred)
                self.cleanup_artifacts()

        except Exception as e:
            traceback.print_exc()

        finally:
            super().on_test_end()

    def configure_optimizers(self):
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'alpha_activation', 'batchnorm', 'layernorm', "activation", "embedding",
                    'LayerNorm.bias', 'LayerNorm.weight',
                    'BatchNorm.bias', 'BatchNorm.weight']

        weight_decay = self.hparams.weight_decay if 'weight_decay' in self.hparams else 0.0
        lr_annealing = self.hparams.lr_annealing if "lr_annealing" in self.hparams else None

        optimizer_grouped_parameters = [
            {'params': [p for name, p in param_optimizer \
                        if not any(key in name for key in no_decay) \
                        and "embeddings" not in name],
             'weight_decay': weight_decay},
            {'params': [p for name, p in param_optimizer if any(key in name for key in no_decay)],
             'weight_decay': 0.0},
        ]

        optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=self.lr)

        extra = {}
        if lr_annealing == "cosine":
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_training_steps,
                                                       eta_min=self.lr / 100)

            extra = {"lr_scheduler": scheduler, "monitor": "val_loss"}
            logger.info("Using CosineAnnealingLR", scheduler.state_dict())

        elif lr_annealing == "restart":
            scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=1,
                                                                 eta_min=self.lr / 100)
            extra = {"lr_scheduler": scheduler, "monitor": "val_loss"}
            logger.info("Using CosineAnnealingWarmRestarts", scheduler.state_dict())

        elif lr_annealing == "reduce":
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer)
            extra = {"lr_scheduler": scheduler, "monitor": "val_loss"}
            logger.info("Using ReduceLROnPlateau", scheduler.state_dict())

        return {"optimizer": optimizer, **extra}


class EdgePredictor(torch.nn.Module):
    def __init__(self, embedding_dim: int, pred_metapaths: List[Tuple[str, str, str]],
                 scoring: str = "DistMult",
                 loss_type: str = "CONTRASTIVE_LOSS",
                 ntype_mapping: Dict[str, str] = None):
        """

        Args:
            embedding_dim (): dimension of node embeddings.
            pred_metapaths (): List of metapaths to predict
        """
        super(EdgePredictor, self).__init__()
        self.pred_metapaths = list({edge_type for head_type, edge_type, tail_type in pred_metapaths})
        self.embedding_dim = embedding_dim
        self.head_batch = "head_batch"
        self.tail_batch = "tail_batch"
        print("LinkPred", scoring, self.pred_metapaths)

        self.ntype_mapping = {ntype: ntype for m in pred_metapaths for ntype in [m[0], m[-1]]}
        if ntype_mapping:
            self.ntype_mapping = {**self.ntype_mapping, **ntype_mapping}

        if scoring == "DistMult":
            self.rel_embedding = nn.Parameter(torch.rand((len(pred_metapaths), embedding_dim)), requires_grad=True)
            self.score_func = self.dist_mult
            if "LOGITS" not in loss_type:
                self.activation = torch.sigmoid
                print("Using sigmoid activation for link pred scores")
            else:
                print("Using no activation for link pred scores")

        elif scoring == "TransE":
            self.rel_embedding = nn.Parameter(torch.rand((len(pred_metapaths), embedding_dim)), requires_grad=True)
            self.score_func = self.transE
            if "LOGITS" in loss_type:
                raise Exception(f"Cannot use loss type {loss_type} with scoring function {scoring}")
            print("Using exp(-x) activation for link pred scores")

        else:
            raise Exception(f"Scoring function parameter `scoring` not supported: {scoring}")

    def forward(self, edges_input: Dict[str, Dict[Tuple[str, str, str], Tensor]],
                embeddings: Dict[str, Tensor]) -> Dict[str, Dict[Tuple[str, str, str], Tensor]]:
        if edges_input is None or len(edges_input) == 0:
            return {}

        output = {}
        # True positive edges
        output["edge_pos"] = self.score_func(edges_input["edge_pos"], embeddings=embeddings, mode="single_pos")

        # True negative edges
        if "edge_neg" in edges_input:
            output["edge_neg"] = self.score_func(edges_input["edge_neg"], embeddings=embeddings, mode="single_neg")

        # Sampled head or tail negative sampling
        if self.head_batch in edges_input or self.tail_batch in edges_input:
            # Head batch
            edge_head_batch, neg_samp_size = get_edge_index_from_neg_batch(neg_batch=edges_input[self.head_batch],
                                                                           edge_pos=edges_input["edge_pos"],
                                                                           mode=self.head_batch)
            output[self.head_batch] = self.score_func(edge_head_batch, embeddings=embeddings, mode=self.tail_batch,
                                                      neg_sampling_batch_size=neg_samp_size)

            # Tail batch
            edge_tail_batch, neg_samp_size = get_edge_index_from_neg_batch(neg_batch=edges_input[self.tail_batch],
                                                                           edge_pos=edges_input["edge_pos"],
                                                                           mode=self.tail_batch)
            output[self.tail_batch] = self.score_func(edge_tail_batch, embeddings=embeddings, mode=self.tail_batch,
                                                      neg_sampling_batch_size=neg_samp_size)

        assert "edge_neg" in output or self.head_batch in output, f"No negative edges in inputs {edges_input.keys()}"

        return output

    def dist_mult(self, edge_index_dict: Dict[Tuple[str, str, str], Tensor],
                  embeddings: Dict[str, Tensor],
                  mode: str,
                  neg_sampling_batch_size: int = None) -> Dict[Tuple[str, str, str], Tensor]:
        edge_index_dict_logits = {}

        for metapath, edge_index in edge_index_dict.items():
            head_type, edge_type, tail_type = metapath
            head_type = self.ntype_mapping[head_type] if head_type not in embeddings else head_type
            tail_type = self.ntype_mapping[tail_type] if tail_type not in embeddings else tail_type

            metapath_idx = self.pred_metapaths.index(edge_type)
            kernel = self.rel_embedding[metapath_idx]  # (emb_dim,emb_dim)

            # if mode == "tail_batch":
            side_A = (embeddings[head_type] * kernel)[edge_index[0]].unsqueeze(1)  # (n_edges, 1, emb_dim)
            emb_B = embeddings[tail_type][edge_index[1]].unsqueeze(2)  # (n_edges, emb_dim, 1)
            scores = torch.bmm(side_A, emb_B).squeeze(-1)
            # else:
            #     emb_A = embeddings[head_type][edge_index[0]].unsqueeze(1)  # (n_edges, 1, emb_dim)
            #     side_B = (kernel * embeddings[tail_type])[edge_index[1]].unsqueeze(2)  # (n_edges, emb_dim, 1)
            #     scores = torch.bmm(emb_A, side_B).squeeze(-1)

            # scores = (embeddings[head_type][edge_index[0]] * kernel * embeddings[tail_type][edge_index[1]])
            scores = scores.sum(dim=1)
            if hasattr(self, "activation") and callable(self.activation):
                scores = self.activation(scores)

            if neg_sampling_batch_size:
                scores = scores.view(-1, neg_sampling_batch_size)

            edge_index_dict_logits[metapath] = scores

        # print("\n", mode)
        # torch.set_printoptions(precision=3, linewidth=300)
        # pprint(edge_pred_dict)
        return edge_index_dict_logits

    def transE(self, edge_index_dict: Dict[Tuple[str, str, str], Tensor],
               embeddings: Dict[str, Tensor],
               mode: str,
               neg_sampling_batch_size: int = None):
        edge_pred_dict = {}

        for metapath, edge_index in edge_index_dict.items():
            head_type, edge_type, tail_type = metapath
            head_type = self.ntype_mapping[head_type] if head_type not in embeddings else head_type
            tail_type = self.ntype_mapping[tail_type] if tail_type not in embeddings else tail_type

            metapath_idx = self.pred_metapaths.index(edge_type)
            kernel = self.rel_embedding[metapath_idx]  # (emb_dim,emb_dim)

            # if mode == "tail_batch":
            side_A = (embeddings[head_type] + kernel)[edge_index[0]]  # (n_edges, emb_dim)
            emb_B = embeddings[tail_type][edge_index[1]]  # (n_edges, emb_dim)
            scores = side_A - emb_B
            # else:
            #     emb_A = embeddings[head_type][edge_index[0]]  # (n_edges, emb_dim)
            #     side_B = (kernel - embeddings[tail_type])[edge_index[1]]  # (n_edges, emb_dim)
            #     scores = emb_A + side_B

            scores = torch.exp(-scores.norm(p=2, dim=1))

            if neg_sampling_batch_size:
                scores = scores.view(-1, neg_sampling_batch_size)

            edge_pred_dict[metapath] = scores

        return edge_pred_dict


class LATTELinkPred(PyGLinkPredTrainer):
    def __init__(self, hparams: Namespace, dataset: HeteroLinkPredDataset, metrics: Dict[str, List[str]],
                 collate_fn=None) -> None:
        if not isinstance(hparams, Namespace) and isinstance(hparams, dict):
            hparams = Namespace(**hparams)
        super().__init__(hparams, dataset, metrics)
        self.head_node_type = dataset.head_node_type
        self.dataset = dataset
        self.multilabel = dataset.multilabel
        self._name = f"LATTE-{hparams.n_layers}-{hparams.t_order}th_Link"
        self.collate_fn = collate_fn

        if hasattr(hparams, "neighbor_sizes"):
            self.dataset.neighbor_sizes = hparams.neighbor_sizes
        else:
            hparams.neighbor_sizes = self.dataset.neighbor_sizes

        # Node attr input
        if hasattr(dataset, 'seq_tokenizer'):
            self.seq_encoder = HeteroSequenceEncoder(hparams, dataset)

        non_seq_ntypes = list(set(self.dataset.node_types).difference(
            set(self.seq_encoder.seq_encoders.keys())) if hasattr(dataset, 'seq_tokenizer') else set())
        if not hasattr(self, "seq_encoder") or len(non_seq_ntypes):
            self.encoder = HeteroNodeFeatureEncoder(hparams, dataset, select_ntypes=non_seq_ntypes)

        self.embedder = LATTEFlat(n_layers=hparams.n_layers,
                                  t_order=min(hparams.t_order, hparams.n_layers),
                                  embedding_dim=hparams.embedding_dim,
                                  num_nodes_dict=dataset.num_nodes_dict,
                                  metapaths=dataset.get_metapaths(),
                                  layer_pooling=hparams.layer_pooling,
                                  activation=hparams.activation,
                                  attn_heads=hparams.attn_heads,
                                  attn_activation=hparams.attn_activation,
                                  attn_dropout=hparams.attn_dropout,
                                  use_proximity=hparams.use_proximity if "use_proximity" in hparams else False,
                                  neg_sampling_ratio=hparams.neg_sampling_ratio if "neg_sampling_ratio" in hparams else None,
                                  edge_sampling=hparams.edge_sampling if "edge_sampling" in hparams else False,
                                  hparams=hparams)

        if hparams.layer_pooling == "concat":
            hparams.embedding_dim = hparams.embedding_dim * hparams.t_order
            logging.info("embedding_dim {}".format(hparams.embedding_dim))

        if "negative_sampling_size" in hparams:
            self.dataset.negative_sampling_size = hparams.negative_sampling_size

        self.classifier = EdgePredictor(embedding_dim=hparams.embedding_dim,
                                        pred_metapaths=dataset.pred_metapaths,
                                        scoring=hparams.scoring if "scoring" in hparams else "DistMult",
                                        loss_type=hparams.loss_type,
                                        ntype_mapping=dataset.ntype_mapping if hasattr(dataset,
                                                                                       "ntype_mapping") else None)

        self.criterion = ClassificationLoss(loss_type=hparams.loss_type, multilabel=False)

        self.hparams.n_params = self.get_n_params()
        self.lr = self.hparams.lr

    @property
    def metapaths(self) -> List[Tuple[str, str, str]]:
        return [layer.metapaths for layer in self.embedder.layers]

    @property
    def betas(self) -> List[Dict[str, DataFrame]]:
        return [layer._betas for layer in self.embedder.layers]

    @property
    def beta_avg(self) -> List[Dict[Tuple[str, str, str], float]]:
        return [layer._beta_avg for layer in self.embedder.layers]

    def configure_sharded_model(self):
        # modules are sharded across processes
        # as soon as they are wrapped with ``wrap`` or ``auto_wrap``.
        # During the forward/backward passes, weights get synced across processes
        # and de-allocated once computation is complete, saving memory.

        # Wraps the layer in a Fully Sharded Wrapper automatically
        if hasattr(self, "seq_encoder"):
            self.seq_encoder = auto_wrap(self.seq_encoder)
        if hasattr(self, "encoder"):
            self.encoder = auto_wrap(self.encoder)

    def forward(self, inputs: Dict[str, Any], edges_true: Dict[str, Dict[Tuple[str, str, str], Tensor]], **kwargs) \
            -> Tuple[Dict[str, Tensor], Dict[str, Dict[Tuple[str, str, str], Tensor]]]:
        input_nodes = inputs["global_node_index"][0] if isinstance(inputs["global_node_index"], list) else inputs[
            "global_node_index"]

        if not self.training:
            self._node_ids = input_nodes

        h_out = {}
        if 'sequences' in inputs and hasattr(self, "seq_encoder"):
            h_out.update(self.seq_encoder.forward(inputs['sequences'],
                                                  minibatch=math.sqrt(self.hparams.batch_size // 3)))

        if len(h_out) < len(input_nodes.keys()):
            embs = self.encoder.forward(inputs["x_dict"], global_node_index=input_nodes)
            h_out.update({ntype: emb for ntype, emb in embs.items() if ntype not in h_out})

        embeddings = self.embedder.forward(h_out, edge_index_dict=inputs["edge_index_dict"],
                                           global_node_index=inputs["global_node_index"],
                                           sizes=inputs["sizes"], **kwargs)

        edges_pred = self.classifier.forward(edges_true, embeddings)

        return embeddings, edges_pred

    def training_step(self, batch, batch_nb):
        X, edge_true, _ = batch
        embeddings, edge_batch_dict = self.forward(X, edge_true)

        pos_scores, neg_batch_scores, neg_scores = self.stack_edge_index_values(
            edge_pos=edge_batch_dict["edge_pos"], edge_neg=edge_batch_dict["edge_neg"],
            neg_head_batch=edge_batch_dict["head_batch"], neg_tail_batch=edge_batch_dict["tail_batch"], )
        loss = self.criterion.forward(pos_scores, neg_batch_scores)

        self.update_link_pred_metrics(self.train_metrics,
                                      pos_edge_score=edge_batch_dict["edge_pos"],
                                      neg_edge_score=edge_batch_dict["edge_neg"],
                                      neg_head_batch=edge_batch_dict["head_batch"],
                                      neg_tail_batch=edge_batch_dict["tail_batch"],
                                      pos_scores=pos_scores,
                                      neg_scores=neg_scores)

        logs = {'loss': loss}
        self.log_dict(logs, prog_bar=True, logger=True, on_step=True)

        return loss

    def validation_step(self, batch, batch_nb):
        X, edge_true, _ = batch
        embeddings, edge_batch_dict = self.forward(X, edge_true, save_betas=True)

        pos_scores, neg_batch_scores, neg_scores = self.stack_edge_index_values(
            edge_pos=edge_batch_dict["edge_pos"], neg_head_batch=edge_batch_dict["head_batch"],
            neg_tail_batch=edge_batch_dict["tail_batch"], edge_neg=edge_batch_dict["edge_neg"])
        loss = self.criterion.forward(pos_scores, neg_batch_scores)

        self.update_link_pred_metrics(self.valid_metrics,
                                      pos_edge_score=edge_batch_dict["edge_pos"],
                                      neg_edge_score=edge_batch_dict["edge_neg"],
                                      neg_head_batch=edge_batch_dict["head_batch"],
                                      neg_tail_batch=edge_batch_dict["tail_batch"],
                                      pos_scores=pos_scores,
                                      neg_scores=neg_scores)

        self.log("val_loss", loss, prog_bar=True)

        return loss

    def test_step(self, batch, batch_nb):
        X, edge_true, _ = batch
        embeddings, edge_batch_dict = self.forward(X, edge_true)

        pos_scores, neg_batch_scores, neg_scores = self.stack_edge_index_values(
            edge_pos=edge_batch_dict["edge_pos"], neg_head_batch=edge_batch_dict["head_batch"],
            neg_tail_batch=edge_batch_dict["tail_batch"], edge_neg=edge_batch_dict["edge_neg"])
        loss = self.criterion.forward(pos_scores, neg_batch_scores)

        self.update_link_pred_metrics(self.test_metrics,
                                      pos_edge_score=edge_batch_dict["edge_pos"],
                                      neg_edge_score=edge_batch_dict["edge_neg"],
                                      neg_head_batch=edge_batch_dict["head_batch"],
                                      neg_tail_batch=edge_batch_dict["tail_batch"],
                                      pos_scores=pos_scores,
                                      neg_scores=neg_scores)

        self.log("test_loss", loss)
        return loss


class HGTLinkPred(LATTELinkPred):
    def __init__(self, hparams, dataset: HeteroLinkPredDataset, metrics=["obgl-biokg"],
                 collate_fn=None) -> None:
        super().__init__(hparams, dataset, metrics)
        self.head_node_type = dataset.head_node_type
        self.dataset = dataset
        self.multilabel = dataset.multilabel
        self._name = f"HGT-{hparams.n_layers}_Link"
        self.collate_fn = collate_fn

        # Node attr input
        if hasattr(dataset, 'seq_tokenizer'):
            self.seq_encoder = HeteroSequenceEncoder(hparams, dataset)

        if not hasattr(self, "seq_encoder") or len(self.seq_encoder.seq_encoders.keys()) < len(dataset.node_types):
            self.encoder = HeteroNodeFeatureEncoder(hparams, dataset)

        self.embedder = HGT(embedding_dim=hparams.embedding_dim, num_layers=hparams.n_layers,
                            num_heads=hparams.attn_heads,
                            node_types=dataset.G.node_types, metadata=dataset.G.metadata())

        self.classifier = EdgePredictor(embedding_dim=hparams.embedding_dim,
                                        pred_metapaths=dataset.pred_metapaths,
                                        ntype_mapping=dataset.ntype_mapping if hasattr(dataset,
                                                                                       "ntype_mapping") else None)

        self.criterion = ClassificationLoss(loss_type=hparams.loss_type, multilabel=False)

        self.hparams.n_params = self.get_n_params()
        self.lr = self.hparams.lr
