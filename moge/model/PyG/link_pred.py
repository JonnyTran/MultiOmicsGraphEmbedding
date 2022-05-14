import logging
from typing import List, Tuple, Dict, Any, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor

from moge.model.PyG.latte_flat import LATTE
from moge.model.losses import PULoss
from ..encoder import HeteroSequenceEncoder, HeteroNodeEncoder
from ..metrics import Metrics
from ..trainer import LinkPredTrainer
from ...dataset import HeteroLinkPredDataset


class DistMulti(torch.nn.Module):
    def __init__(self, embedding_dim: int, metapaths: List[Tuple[str, str, str]], ntype_mapping: Dict[str, str] = None):
        """

        Args:
            embedding_dim ():
            metapaths (): List of metapaths to predict
        """
        super(DistMulti, self).__init__()
        self.metapaths = metapaths
        self.embedding_dim = embedding_dim

        self.ntype_mapping = {ntype: ntype for m in metapaths for ntype in [m[0], m[-1]]}
        if ntype_mapping:
            self.ntype_mapping = {**self.ntype_mapping, **ntype_mapping}

        self.rel_embedding = nn.Parameter(torch.zeros(len(metapaths), embedding_dim), requires_grad=True)
        nn.init.uniform_(tensor=self.rel_embedding, a=-1, b=1)

    def forward(self, edges_input: Dict[str, Dict[Tuple[str, str, str], Tensor]],
                embeddings: Dict[str, Tensor]) -> Dict[str, Dict[Tuple[str, str, str], Tensor]]:
        output = {}

        # True positive edges
        output["edge_pos"] = self.score(edges_input["edge_pos"], embeddings=embeddings, mode="single_pos")

        # True negative edges
        if "edge_neg" in edges_input:
            output["edge_neg"] = self.score(edges_input["edge_neg"], embeddings=embeddings, mode="single_neg")

        # Sampled head or tail negative sampling
        if "head_batch" in edges_input or "tail_batch" in edges_input:
            # Head batch
            edge_head_batch, neg_samp_size = self.get_edge_index_from_neg_batch(edges_input["edge_pos"],
                                                                                neg_edges=edges_input["head_batch"],
                                                                                mode="head_batch")
            output["head_batch"] = self.score(edge_head_batch, embeddings=embeddings, mode="tail_batch",
                                              neg_sampling_batch_size=neg_samp_size)

            # Tail batch
            edge_tail_batch, neg_samp_size = self.get_edge_index_from_neg_batch(edges_input["edge_pos"],
                                                                                neg_edges=edges_input["tail_batch"],
                                                                                mode="tail_batch")
            output["tail_batch"] = self.score(edge_tail_batch, embeddings=embeddings, mode="tail_batch",
                                              neg_sampling_batch_size=neg_samp_size)

        assert "edge_neg" in output or "head_batch" in output, f"No negative edges in inputs {edges_input.keys()}"

        return output

    def score(self, edge_index_dict: Dict[Tuple[str, str, str], Tensor],
              embeddings: Dict[str, Tensor],
              mode: str,
              neg_sampling_batch_size: int = None) -> Dict[Tuple[str, str, str], Tensor]:
        edge_pred_dict = {}

        for metapath, edge_index in edge_index_dict.items():
            metapath_idx = self.metapaths.index(metapath)
            kernel = self.rel_embedding[metapath_idx]  # (emb_dim)

            head_type, edge_type, tail_type = metapath
            head_type = self.ntype_mapping[head_type] if head_type not in embeddings else head_type
            tail_type = self.ntype_mapping[tail_type] if tail_type not in embeddings else tail_type

            if mode == "tail_batch":
                side_A = (embeddings[head_type] * kernel)[edge_index[0]].unsqueeze(1)  # (n_edges, 1, emb_dim)
                emb_B = embeddings[tail_type][edge_index[1]].unsqueeze(2)  # (n_edges, emb_dim, 1)
                scores = torch.bmm(side_A, emb_B).squeeze(-1)
            else:
                emb_A = embeddings[head_type][edge_index[0]].unsqueeze(1)  # (n_edges, 1, emb_dim)
                side_B = (kernel * embeddings[tail_type])[edge_index[1]].unsqueeze(2)  # (n_edges, emb_dim, 1)
                scores = torch.bmm(emb_A, side_B).squeeze(-1)

            # scores = (embeddings[head_type][edge_index[0]] * kernel * embeddings[tail_type][edge_index[1]])
            scores = scores.sum(dim=1)

            if neg_sampling_batch_size:
                scores = scores.view(-1, neg_sampling_batch_size)
            edge_pred_dict[metapath] = scores

        # print("\n", mode)
        # torch.set_printoptions(precision=3, linewidth=300)
        # pprint(edge_pred_dict)
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

        self.classifier = DistMulti(embedding_dim=hparams.embedding_dim, metapaths=dataset.pred_metapaths,
                                    ntype_mapping=dataset.ntype_mapping if hasattr(dataset, "ntype_mapping") else None)

        self.criterion = PULoss(prior=dataset.get_prior())
        # self.criterion = LinkPredLoss()

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

        e_pos, e_neg, e_weights = self.stack_pos_head_tail_batch(edge_pred_dict, edge_weights)

        loss = self.criterion.forward(*self.create_pu_learning_tensors(edge_pred_dict), e_weights)
        # loss = self.criterion.forward(e_pos, e_neg, e_weights)

        metrics = self.train_metrics
        self.update_link_pred_metrics(metrics, edge_pred_dict, e_pos, e_neg)
        # print(self.valid_metrics.compute_metrics())

        if "edge_neg" in edge_pred_dict:
            self.update_pr_metrics(e_pos=e_pos, e_neg=edge_pred_dict["edge_neg"],
                                   metrics=metrics, subset=["precision", "recall"])

        logs = {'loss': loss,
                # **self.train_metrics.compute_metrics()
                }
        self.log_dict(logs, prog_bar=True, logger=True, on_step=True)

        return loss

    def validation_step(self, batch, batch_nb):
        X, edge_true, edge_weights = batch
        embeddings, _, edge_pred_dict = self.forward(X, edge_true)

        e_pos, e_neg, e_weights = self.stack_pos_head_tail_batch(edge_pred_dict, edge_weights)
        loss = self.criterion.forward(*self.create_pu_learning_tensors(edge_pred_dict), e_weights)
        # loss = self.criterion.forward(e_pos, e_neg, e_weights)

        metrics = self.valid_metrics
        self.update_link_pred_metrics(metrics, edge_pred_dict, e_pos, e_neg)
        # print(self.valid_metrics.compute_metrics())

        if "edge_neg" in edge_pred_dict:
            self.update_pr_metrics(e_pos=e_pos, e_neg=edge_pred_dict["edge_neg"],
                                   metrics=metrics, subset=["precision", "recall"])

        self.log("val_loss", loss, prog_bar=True)

        return loss

    def test_step(self, batch, batch_nb):
        X, edge_true, edge_weights = batch
        embeddings, _, edge_pred_dict = self.forward(X, edge_true)

        e_pos, e_neg, e_weights = self.stack_pos_head_tail_batch(edge_pred_dict, edge_weights)

        np.set_printoptions(precision=3, suppress=True, linewidth=300)
        print("\npos", F.sigmoid(e_pos[:20]).detach().cpu().numpy(),
              "\nneg", F.sigmoid(e_neg[:20, 0].view(-1)).detach().cpu().numpy()) if batch_nb == 1 else None

        loss = self.criterion.forward(*self.create_pu_learning_tensors(edge_pred_dict), e_weights)

        metrics = self.test_metrics
        self.update_link_pred_metrics(metrics, edge_pred_dict, e_pos, e_neg)

        if "edge_neg" in edge_pred_dict:
            self.update_pr_metrics(e_pos=e_pos, e_neg=edge_pred_dict["edge_neg"],
                                   metrics=metrics, subset=["precision", "recall"])

        self.log("test_loss", loss)
        return loss

    def update_link_pred_metrics(self, metrics: Union[Metrics, Dict[str, Metrics]],
                                 edge_pred_dict, e_pos: Tensor, e_neg: Tensor):
        if isinstance(metrics, dict):
            for metapath in edge_pred_dict["edge_pos"]:
                go_type = "BPO" if metapath[-1] == 'biological_process' else \
                    "CCO" if metapath[-1] == 'cellular_component' else \
                        "MFO" if metapath[-1] == 'molecular_function' else None

                neg_batch = torch.concat([edge_pred_dict["head_batch"][metapath],
                                          edge_pred_dict["tail_batch"][metapath]], dim=1)
                metrics[go_type].update_metrics(F.sigmoid(edge_pred_dict["edge_pos"][metapath].detach()),
                                                F.sigmoid(neg_batch.detach()),
                                                weights=None, subset=["ogbl-biokg"])

        else:
            metrics.update_metrics(F.sigmoid(e_pos.detach()), F.sigmoid(e_neg.detach()),
                                   weights=None, subset=["ogbl-biokg"])

    def update_pr_metrics(self, e_pos, e_neg, metrics: Metrics, subset=["precision", "recall"]):
        if not isinstance(metrics, Metrics): return
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
