import logging
import math
import traceback
from argparse import Namespace
from typing import List, Tuple, Dict, Any, Union

import numpy as np
import torch
from fairscale.nn import auto_wrap
from torch import nn, Tensor

from moge.model.PyG.latte_flat import LATTE
from moge.model.losses import ClassificationLoss
from .conv import HGT
from ..encoder import HeteroSequenceEncoder, HeteroNodeFeatureEncoder
from ..metrics import Metrics
from ..trainer import LinkPredTrainer
from ...dataset import HeteroLinkPredDataset


class LinkPred(torch.nn.Module):
    def __init__(self, embedding_dim: int, pred_metapaths: List[Tuple[str, str, str]], scoring, loss_type: str,
                 ntype_mapping: Dict[str, str] = None):
        """

        Args:
            embedding_dim (): dimension of node embeddings.
            pred_metapaths (): List of metapaths to predict
        """
        super(LinkPred, self).__init__()
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

        nn.init.uniform_(tensor=self.rel_embedding)

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
            edge_head_batch, neg_samp_size = self.get_edge_index_from_neg_batch(edges_input["edge_pos"],
                                                                                neg_edges=edges_input[self.head_batch],
                                                                                mode=self.head_batch)
            output[self.head_batch] = self.score_func(edge_head_batch, embeddings=embeddings, mode=self.tail_batch,
                                                      neg_sampling_batch_size=neg_samp_size)

            # Tail batch
            edge_tail_batch, neg_samp_size = self.get_edge_index_from_neg_batch(edges_input["edge_pos"],
                                                                                neg_edges=edges_input[self.tail_batch],
                                                                                mode=self.tail_batch)
            output[self.tail_batch] = self.score_func(edge_tail_batch, embeddings=embeddings, mode=self.tail_batch,
                                                      neg_sampling_batch_size=neg_samp_size)

        assert "edge_neg" in output or self.head_batch in output, f"No negative edges in inputs {edges_input.keys()}"

        return output

    def dist_mult(self, edge_index_dict: Dict[Tuple[str, str, str], Tensor],
                  embeddings: Dict[str, Tensor],
                  mode: str,
                  neg_sampling_batch_size: int = None) -> Dict[Tuple[str, str, str], Tensor]:
        edge_pred_dict = {}

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

            edge_pred_dict[metapath] = scores

        # print("\n", mode)
        # torch.set_printoptions(precision=3, linewidth=300)
        # pprint(edge_pred_dict)
        return edge_pred_dict

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

    def get_edge_index_from_neg_batch(self, pos_edges: Dict[Tuple[str, str, str], Tensor],
                                      neg_edges: Dict[Tuple[str, str, str], Tensor],
                                      mode: str) -> Tuple[Dict[Tuple[str, str, str], Tensor], int]:
        edge_index_dict = {}

        for metapath, edge_index in pos_edges.items():
            num_edges, neg_samp_size = neg_edges[metapath].shape

            if mode == "head_batch":
                head_nodes = neg_edges[metapath].view(-1)
                tail_nodes = pos_edges[metapath][1].repeat_interleave(neg_samp_size)
                edge_index_dict[metapath] = torch.stack([head_nodes, tail_nodes], dim=0)

            elif mode == "tail_batch":
                head_nodes = pos_edges[metapath][0].repeat_interleave(neg_samp_size)
                tail_nodes = neg_edges[metapath].view(-1)
                edge_index_dict[metapath] = torch.stack([head_nodes, tail_nodes], dim=0)

        return edge_index_dict, neg_samp_size


class LATTELinkPred(LinkPredTrainer):
    def __init__(self, hparams: Namespace, dataset: HeteroLinkPredDataset,
                 metrics: Dict[str, List[str]] = ["obgl-biokg"],
                 collate_fn=None) -> None:
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

        self.embedder = LATTE(n_layers=hparams.n_layers,
                              t_order=hparams.t_order,
                              embedding_dim=hparams.embedding_dim,
                              num_nodes_dict=dataset.num_nodes_dict,
                              metapaths=dataset.get_metapaths(khop=None),
                              layer_pooling=hparams.layer_pooling,
                              activation=hparams.activation,
                              attn_heads=hparams.attn_heads,
                              attn_activation=hparams.attn_activation,
                              attn_dropout=hparams.attn_dropout,
                              use_proximity=hparams.use_proximity if hasattr(hparams, "use_proximity") else False,
                              neg_sampling_ratio=hparams.neg_sampling_ratio \
                                  if hasattr(hparams, "neg_sampling_ratio") else None,
                              edge_sampling=hparams.edge_sampling if hasattr(hparams, "edge_sampling") else False,
                              hparams=hparams)

        if hparams.layer_pooling == "concat":
            hparams.embedding_dim = hparams.embedding_dim * hparams.t_order
            logging.info("embedding_dim {}".format(hparams.embedding_dim))

        if "negative_sampling_size" in hparams:
            self.dataset.negative_sampling_size = hparams.negative_sampling_size

        self.classifier = LinkPred(embedding_dim=hparams.embedding_dim,
                                   pred_metapaths=dataset.pred_metapaths,
                                   scoring=hparams.scoring if hasattr(hparams, "scoring") else "DistMult",
                                   loss_type=hparams.loss_type,
                                   ntype_mapping=dataset.ntype_mapping if hasattr(dataset, "ntype_mapping") else None)

        self.criterion = ClassificationLoss(loss_type=hparams.loss_type, multilabel=False)

        self.hparams.n_params = self.get_n_params()
        self.lr = self.hparams.lr

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

    def forward(self, inputs: Dict[str, Any], edges_true: Dict[str, Dict[Tuple[str, str, str], Tensor]],
                return_score=False,
                **kwargs) \
            -> Tuple[Dict[str, Tensor], Any, Dict[str, Dict[Tuple[str, str, str], Tensor]]]:
        if not self.training:
            self._node_ids = inputs["global_node_index"]

        h_out = {}
        if 'sequences' in inputs and hasattr(self, "seq_encoder"):
            h_out.update(self.seq_encoder.forward(inputs['sequences'],
                                                  minibatch=math.sqrt(self.hparams.batch_size // 3)))

        if len(h_out) < len(inputs["global_node_index"].keys()):
            embs = self.encoder.forward(inputs["x_dict"], global_node_idx=inputs["global_node_index"])
            h_out.update({ntype: emb for ntype, emb in embs.items() if ntype not in h_out})

        embeddings = self.embedder.forward(h_out,
                                           edge_index_dict=inputs["edge_index_dict"],
                                           global_node_idx=inputs["global_node_index"],
                                           sizes=inputs["sizes"],
                                           **kwargs)

        edges_pred = self.classifier.forward(edges_true, embeddings)
        if return_score:
            edges_pred = {pos_neg: {metapath: torch.sigmoid(edge_logits) for metapath, edge_logits in edge_dict.items()} \
                          for pos_neg, edge_dict in edges_pred.items()}

        return embeddings, edges_pred

    def training_step(self, batch, batch_nb):
        X, edge_true, edge_weights = batch
        embeddings, edge_pred_dict = self.forward(X, edge_true)

        e_pos, e_neg, e_weights = self.stack_pos_head_tail_batch(edge_pred_dict, edge_weights)
        # loss = self.criterion.forward(*self.reshape_edge_pred_dict(edge_pred_dict))
        loss = self.criterion.forward(e_pos, e_neg, e_weights)

        self.update_link_pred_metrics(self.train_metrics, edge_pred_dict, e_pos, e_neg)

        logs = {'loss': loss,
                # **self.train_metrics.compute_metrics()
                }
        self.log_dict(logs, prog_bar=True, logger=True, on_step=True)

        return loss

    def validation_step(self, batch, batch_nb):
        X, edge_true, edge_weights = batch
        embeddings, edge_pred_dict = self.forward(X, edge_true, save_betas=True)

        e_pos, e_neg, e_weights = self.stack_pos_head_tail_batch(edge_pred_dict, edge_weights)
        # loss = self.criterion.forward(*self.reshape_edge_pred_dict(edge_pred_dict))
        loss = self.criterion.forward(e_pos, e_neg, e_weights)

        self.update_link_pred_metrics(self.valid_metrics, edge_pred_dict, e_pos, e_neg)

        self.log("val_loss", loss, prog_bar=True)

        return loss

    def on_validation_end(self) -> None:
        super().on_validation_end()
        if self.current_epoch % 5 == 1:
            self.plot_sankey_flow(layer=-1, width=max(250 * self.embedder.t_order, 500))

    def test_step(self, batch, batch_nb):
        X, edge_true, edge_weights = batch
        embeddings, edge_pred_dict = self.forward(X, edge_true)

        e_pos, e_neg, e_weights = self.stack_pos_head_tail_batch(edge_pred_dict, edge_weights)
        # loss = self.criterion.forward(*self.reshape_edge_pred_dict(edge_pred_dict))
        loss = self.criterion.forward(e_pos, e_neg, e_weights)

        self.update_link_pred_metrics(self.test_metrics, edge_pred_dict, e_pos, e_neg)
        # np.set_printoptions(precision=3, suppress=True, linewidth=300)
        # print("\npos", torch.sigmoid(e_pos[:20]).detach().cpu().numpy(),
        #       "\nneg", torch.sigmoid(e_neg[:20, 0].view(-1)).detach().cpu().numpy()) if batch_nb == 1 else None

        self.log("test_loss", loss)
        return loss

    def on_test_end(self):
        try:
            if self.wandb_experiment is not None:
                # X, y, _ = self.dataset.get_full_graph()
                X, y, _ = self.dataset.transform(
                    edge_idx=torch.cat(
                        [self.dataset.training_idx, self.dataset.validation_idx, self.dataset.testing_idx]))
                embs, edge_pred_dict = self.cpu().forward(X, y, save_betas=True)

                self.predict_umap(X, embs, log_table=True)
                self.plot_sankey_flow(layer=-1, width=max(250 * self.embedder.t_order, 500))
                test_pred_dict = edge_pred_dict
                self.log_score_averages(edge_pred_dict=test_pred_dict)
                self.cleanup_artifacts()

        except Exception as e:
            traceback.print_exc()

        finally:
            super().on_test_end()

    def update_link_pred_metrics(self, metrics: Union[Metrics, Dict[str, Metrics]],
                                 edge_pred_dict: Dict[str, Dict[Tuple[str, str, str], Tensor]],
                                 e_pos: Tensor, e_neg: Tensor):
        if isinstance(metrics, dict):
            for metapath in edge_pred_dict["edge_pos"]:
                go_type = "BPO" if metapath[-1] == 'biological_process' else \
                    "CCO" if metapath[-1] == 'cellular_component' else \
                        "MFO" if metapath[-1] == 'molecular_function' else None

                neg_batch = torch.concat([edge_pred_dict["head_batch"][metapath],
                                          edge_pred_dict["tail_batch"][metapath]], dim=1)
                metrics[go_type].update_metrics(edge_pred_dict["edge_pos"][metapath].detach(),
                                                neg_batch.detach(),
                                                weights=None, subset=["ogbl-biokg"])

                if metapath in edge_pred_dict["edge_pos"] and "edge_neg" in edge_pred_dict and metapath in \
                        edge_pred_dict["edge_neg"]:
                    self.update_pr_metrics(e_pos=edge_pred_dict["edge_pos"][metapath],
                                           edge_pred=edge_pred_dict["edge_neg"][metapath],
                                           metrics=metrics[go_type])

        else:
            metrics.update_metrics(e_pos.detach(), e_neg.detach(),
                                   weights=None, subset=["ogbl-biokg"])
            self.update_pr_metrics(e_pos=e_pos, edge_pred=edge_pred_dict["edge_neg"],
                                   metrics=metrics)

    def update_pr_metrics(self, e_pos: Tensor, edge_pred: Union[Tensor, Dict[Tuple[str, str, str], Tensor]],
                          metrics: Metrics, subset=["precision", "recall", "aupr"]):
        if isinstance(edge_pred, dict):
            edge_neg_score = torch.cat([edge_scores.detach() for m, edge_scores in edge_pred.items()])
        else:
            edge_neg_score = edge_pred

        # randomly select |e_neg| positive edges to balance precision/recall scores
        e_pos = e_pos[np.random.choice(e_pos.size(0), size=edge_neg_score.shape,
                                       replace=False if e_pos.size(0) > edge_neg_score.size(0) else True)]

        y_pred = torch.cat([e_pos, edge_neg_score]).unsqueeze(-1).detach()
        y_true = torch.cat([torch.ones_like(e_pos), torch.zeros_like(edge_neg_score)]).unsqueeze(-1)

        metrics.update_metrics(y_pred, y_true, weights=None, subset=subset)

    def configure_optimizers(self):
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'alpha_activation', 'batchnorm', 'layernorm', "activation", "embedding",
                    'LayerNorm.bias', 'LayerNorm.weight',
                    'BatchNorm.bias', 'BatchNorm.weight']

        optimizer_grouped_parameters = [
            {'params': [p for name, p in param_optimizer \
                        if not any(key in name for key in no_decay) \
                        and "embeddings" not in name],
             'weight_decay': self.hparams.weight_decay if isinstance(self.hparams.weight_decay, float) else 0.0},
            {'params': [p for name, p in param_optimizer if any(key in name for key in no_decay)],
             'weight_decay': 0.0},
        ]

        optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=self.lr)

        extra = {}
        if "lr_annealing" in self.hparams and self.hparams.lr_annealing == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                   T_max=self.num_training_steps,
                                                                   eta_min=self.lr / 100)

            extra = {"lr_scheduler": scheduler, "monitor": "val_loss"}
            print("Using CosineAnnealingLR", scheduler.state_dict())

        elif "lr_annealing" in self.hparams and self.hparams.lr_annealing == "restart":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                             T_0=50, T_mult=1,
                                                                             eta_min=self.lr / 100)
            extra = {"lr_scheduler": scheduler, "monitor": "val_loss"}
            print("Using CosineAnnealingWarmRestarts", scheduler.state_dict())

        return {"optimizer": optimizer, **extra}


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

        self.classifier = LinkPred(embedding_dim=hparams.embedding_dim,
                                   pred_metapaths=dataset.pred_metapaths,
                                   ntype_mapping=dataset.ntype_mapping if hasattr(dataset, "ntype_mapping") else None)

        self.criterion = ClassificationLoss(loss_type=hparams.loss_type, multilabel=False)

        self.hparams.n_params = self.get_n_params()
        self.lr = self.hparams.lr
