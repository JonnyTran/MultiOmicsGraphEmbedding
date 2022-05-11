import logging
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor

from moge.model.PyG.latte_flat import LATTE
from moge.model.losses import LinkPredLoss
from ..encoder import HeteroSequenceEncoder, HeteroNodeEncoder
from ..metrics import Metrics
from ..trainer import LinkPredTrainer
from ...dataset import HeteroLinkPredDataset


class DistMulti(torch.nn.Module):
    def __init__(self, embedding_dim: int, metapaths: List[Tuple[str, str, str]]):
        """

        Args:
            embedding_dim ():
            metapaths (): List of metapaths to predict
        """
        super(DistMulti, self).__init__()
        self.metapaths = metapaths
        self.embedding_dim = embedding_dim

        self.relation_embedding = nn.Parameter(torch.zeros(len(metapaths), embedding_dim), requires_grad=True)
        nn.init.uniform_(tensor=self.relation_embedding, a=-1, b=1)

    def forward(self, edges_true: Dict[str, Dict[Tuple[str, str, str], Tensor]],
                embeddings: Dict[str, Tensor]) -> Dict[str, Dict[Tuple[str, str, str], Tensor]]:
        output = {}

        # Single edges
        output["edge_pos"] = self.score(edges_true["edge_pos"], embeddings=embeddings, mode="single")

        # True negative edges
        if "edge_neg" in edges_true:
            output["edge_neg"] = self.score(edges_true["edge_neg"], embeddings=embeddings, mode="single")

        # Sampled head or tail negative sampling
        if "head_batch" in edges_true or "tail_batch" in edges_true:
            # Head batch
            edge_head_batch = self.get_edge_index_from_neg_batch(edges_true["edge_pos"],
                                                                 neg_edges=edges_true["head_batch"],
                                                                 mode="head")
            output["head_batch"] = self.score(edge_head_batch, embeddings, mode="head")

            # Tail batch
            edge_tail_batch = self.get_edge_index_from_neg_batch(edges_true["edge_pos"],
                                                                 neg_edges=edges_true["tail_batch"],
                                                                 mode="tail")
            output["tail_batch"] = self.score(edge_tail_batch, embeddings, mode="tail")

        assert "edge_neg" in output or "head_batch" in output, f"No negative edges in inputs {edges_true.keys()}"

        return output

    def score(self, edge_index_dict: Dict[Tuple[str, str, str], Tensor],
              embeddings: Dict[str, Tensor],
              mode: str) -> Dict[Tuple[str, str, str], Tensor]:
        edge_pred_dict = {}

        for metapath, edge_index in edge_index_dict.items():
            metapath_idx = self.metapaths.index(metapath)
            head_type, edge_type, tail_type = metapath

            kernel = self.relation_embedding[metapath_idx]  # (emb_dim)

            if "tail" == mode:
                side_A = (embeddings[head_type] * kernel)[edge_index[0]].unsqueeze(1)  # (n_nodes, 1, emb_dim)
                emb_B = embeddings[tail_type][edge_index[1]].unsqueeze(2)  # (n_nodes, emb_dim, 1)
                # score = side_A * emb_B
                score = torch.bmm(side_A, emb_B).sum(-1)
            else:
                emb_A = embeddings[head_type][edge_index[0]].unsqueeze(1)  # (n_nodes, 1, emb_dim)
                side_B = (kernel * embeddings[tail_type][edge_index[1]]).unsqueeze(2)  # (n_nodes, emb_dim, 1)
                # score = emb_A * side_B
                score = torch.bmm(emb_A, side_B)
                score = score.sum(-1)

            score = score.sum(dim=1)
            # assert score.dim() == 1, f"{mode} score={score.shape}"
            edge_pred_dict[metapath] = score

        return edge_pred_dict

    def get_edge_index_from_neg_batch(self, pos_edges: Dict[Tuple[str, str, str], Tensor],
                                      neg_edges: Dict[Tuple[str, str, str], Tensor],
                                      mode: str) -> Dict[Tuple[str, str, str], Tensor]:
        edge_index_dict = {}

        for metapath, edge_index in pos_edges.items():
            e_size, neg_samp_size = neg_edges[metapath].shape

            if mode == "head":
                head_nodes = neg_edges[metapath].reshape(-1)
                tail_nodes = pos_edges[metapath][1].repeat_interleave(neg_samp_size)
                edge_index_dict[metapath] = torch.stack([head_nodes, tail_nodes], dim=0)
            elif mode == "tail":
                head_nodes = pos_edges[metapath][0].repeat_interleave(neg_samp_size)
                tail_nodes = neg_edges[metapath].reshape(-1)
                edge_index_dict[metapath] = torch.stack([head_nodes, tail_nodes], dim=0)

        return edge_index_dict


class LATTELinkPred(LinkPredTrainer):
    def __init__(self, hparams, dataset: HeteroLinkPredDataset, metrics=["obgl-biokg"],
                 collate_fn=None) -> None:
        super().__init__(hparams, dataset, metrics)
        self.head_node_type = dataset.head_node_type
        self.dataset = dataset
        self.multilabel = dataset.multilabel
        self._name = f"LATTE-{hparams.t_order} Link"
        self.collate_fn = collate_fn

        # Node attr input
        if hasattr(dataset, 'seq_tokenizer'):
            self.encoder = HeteroSequenceEncoder(hparams, dataset)
        else:
            self.encoder = HeteroNodeEncoder(hparams, dataset)

        self.embedder = LATTE(n_layers=hparams.n_layers,
                              t_order=min(hparams.t_order, hparams.n_layers),
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

        self.classifier = DistMulti(embedding_dim=hparams.embedding_dim, metapaths=dataset.pred_metapaths)
        self.criterion = LinkPredLoss()

        self.hparams.n_params = self.get_n_params()
        self.lr = self.hparams.lr

    def forward(self, inputs: Dict[str, Any], edges_true: Dict[str, Dict[Tuple[str, str, str], Tensor]],
                return_score=False,
                **kwargs) \
            -> Tuple[Dict[str, Tensor], Any, Dict[str, Dict[Tuple[str, str, str], Tensor]]]:
        if not self.training:
            self._node_ids = inputs["global_node_index"]

        if "sequences" in inputs and isinstance(self.encoder, HeteroSequenceEncoder):
            h_out = self.encoder.forward(inputs['sequences'])
        elif "x_dict" in inputs:
            h_out = self.encoder.forward(inputs["x_dict"], global_node_idx=inputs["global_node_index"])

        embeddings, aux_loss, _ = self.embedder.forward(h_out,
                                                        edge_index_dict=inputs["edge_index_dict"],
                                                        global_node_idx=inputs["global_node_index"],
                                                        sizes=inputs["sizes"],
                                                        **kwargs)

        edges_pred = self.classifier.forward(edges_true, embeddings)
        if return_score:
            edges_pred = {pos_neg: {m: F.sigmoid(edge_weight) for m, edge_weight in edge_dict.items()} \
                          for pos_neg, edge_dict in edges_pred.items()}

        return embeddings, aux_loss, edges_pred

    def training_step(self, batch, batch_nb):
        X, edge_true, edge_weights = batch
        embeddings, _, edge_pred_dict = self.forward(X, edge_true)

        e_pos, e_neg, e_weights = self.get_pos_neg_edges(edge_pred_dict, edge_weights)
        loss = self.criterion.forward(e_pos, e_neg, pos_weights=e_weights)

        self.train_metrics.update_metrics(F.sigmoid(e_pos.detach()), F.sigmoid(e_neg.detach()),
                                          weights=None, subset=["ogbl-biokg"])

        if "edge_neg" in edge_pred_dict:
            self.update_pr_metrics(e_pos=e_pos, e_neg=edge_pred_dict["edge_neg"],
                                   metrics=self.train_metrics, subset=["precision", "recall"])

        logs = {'loss': loss, **self.train_metrics.compute_metrics()}
        self.log_dict(logs, prog_bar=True, logger=True, on_step=True)

        return loss

    def validation_step(self, batch, batch_nb):
        X, edge_true, edge_weights = batch
        embeddings, _, edge_pred_dict = self.forward(X, edge_true)

        e_pos, e_neg, e_weights = self.get_pos_neg_edges(edge_pred_dict, edge_weights)
        loss = self.criterion.forward(e_pos, e_neg, pos_weights=e_weights)

        self.valid_metrics.update_metrics(F.sigmoid(e_pos.detach()), F.sigmoid(e_neg.detach()),
                                          weights=None, subset=["ogbl-biokg"])

        if "edge_neg" in edge_pred_dict:
            self.update_pr_metrics(e_pos=e_pos, e_neg=edge_pred_dict["edge_neg"],
                                   metrics=self.valid_metrics, subset=["precision", "recall"])

        self.log("val_loss", loss, prog_bar=True)

        return loss

    def test_step(self, batch, batch_nb):
        X, edge_true, edge_weights = batch
        embeddings, _, edge_pred_dict = self.forward(X, edge_true)

        e_pos, e_neg, e_weights = self.get_pos_neg_edges(edge_pred_dict, edge_weights)

        np.set_printoptions(precision=3, suppress=True)
        print("pos", F.sigmoid(e_pos[:10]).detach().cpu().numpy(),
              "\nneg", F.sigmoid(e_neg[:10, 0].view(-1)).detach().cpu().numpy()) if batch_nb == 1 else None

        loss = self.criterion.forward(e_pos, e_neg, pos_weights=e_weights)
        self.test_metrics.update_metrics(F.sigmoid(e_pos.detach()), F.sigmoid(e_neg.detach()),
                                         weights=None, subset=["ogbl-biokg"])

        if "edge_neg" in edge_pred_dict:
            self.update_pr_metrics(e_pos=e_pos, e_neg=edge_pred_dict["edge_neg"],
                                   metrics=self.test_metrics, subset=["precision", "recall"])

        self.log("test_loss", loss)
        return loss

    def update_pr_metrics(self, e_pos, e_neg, metrics: Metrics, subset=["precision", "recall"]):
        edge_neg_score = torch.cat([edge_scores.detach() for m, edge_scores in e_neg.items()])
        e_pos = e_pos[torch.randint(high=e_pos.size(0), size=edge_neg_score.shape)]  # randomly select |e_neg| edges
        y_pred = F.sigmoid(torch.cat([e_pos, edge_neg_score]).unsqueeze(-1).detach())
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
             'weight_decay': self.hparams.weight_decay},
            {'params': [p for name, p in param_optimizer if any(key in name for key in no_decay)],
             'weight_decay': 0.0},
        ]

        optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=self.lr)

        extra = {}
        if "lr_annealing" in self.hparams and self.hparams.lr_annealing == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                   T_max=self.num_training_steps,
                                                                   eta_min=self.lr / 100
                                                                   )
            extra = {"lr_scheduler": scheduler, "monitor": "val_loss"}
            print("Using CosineAnnealingLR", scheduler.state_dict())

        elif "lr_annealing" in self.hparams and self.hparams.lr_annealing == "restart":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                             T_0=50, T_mult=1,
                                                                             eta_min=self.lr / 100)
            extra = {"lr_scheduler": scheduler, "monitor": "val_loss"}
            print("Using CosineAnnealingWarmRestarts", scheduler.state_dict())

        return {"optimizer": optimizer, **extra}
