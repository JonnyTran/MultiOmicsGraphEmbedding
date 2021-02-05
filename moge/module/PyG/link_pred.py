import multiprocessing
from typing import Callable
import logging

import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.hooks import RemovableHandle

from moge.generator import HeteroNetDataset
from moge.module.PyG.latte import LATTE, untag_negative, is_negative
from moge.module.utils import tensor_sizes
from ..trainer import LinkPredTrainer
from moge.module.losses import LinkPredLoss

class DistMulti(torch.nn.Module):
    def __init__(self, embedding_dim, metapaths):
        super(DistMulti, self).__init__()
        self.metapaths = metapaths
        self.embedding_dim = embedding_dim

        # self.linears = nn.ModuleDict(
        #     {metapath: nn.Parameter(torch.Tensor((embedding_dim, embedding_dim))) \
        #      for metapath in metapaths}
        # )

        self.relation_embedding = nn.Parameter(torch.zeros(len(metapaths), embedding_dim, embedding_dim))
        nn.init.uniform_(tensor=self.relation_embedding, a=-1, b=1)

    def forward(self, inputs, embeddings):
        output = {}

        # Single edges
        output["edge_pos"] = self.predict(inputs["edge_index_dict"], embeddings, neg_samp_size=None, mode=None)

        # Head batch
        edge_head_batch, neg_samp_size = self.get_edge_index_from_batch(inputs["edge_index_dict"],
                                                                        neg_batch=inputs["edge_neg_head"],
                                                                        mode="head")
        output["edge_neg_head"] = self.predict(edge_head_batch, embeddings, neg_samp_size=neg_samp_size, mode="head")

        # Tail batch
        edge_tail_batch, neg_samp_size = self.get_edge_index_from_batch(inputs["edge_index_dict"],
                                                                        neg_batch=inputs["edge_neg_tail"],
                                                                        mode="tail")
        output["edge_neg_tail"] = self.predict(edge_tail_batch, embeddings, neg_samp_size=neg_samp_size, mode="tail")

        return output

    def predict(self, edge_index_dict, embeddings, neg_samp_size, mode=None):
        # Find neg_sampling size

        edge_pred_dict = {}

        for metapath, edge_index in edge_index_dict.items():
            metapath_idx = self.metapaths.index(metapath)
            kernel = self.relation_embedding[metapath_idx]

            if "head" == mode:
                emb_A = embeddings[metapath[0]][edge_index[0]]
                num_edges = edge_index.shape[1] / neg_samp_size
                idx_B = edge_index[0][torch.arange(num_edges, dtype=torch.long) * neg_samp_size]
                side_B = kernel @ embeddings[metapath[0]][idx_B].t()
                side_B = side_B.repeat_interleave(neg_samp_size, dim=-1)

                score = emb_A @ side_B
            elif "tail" == mode:
                emb_B = embeddings[metapath[-1]][edge_index[1]]
                num_edges = edge_index.shape[1] / neg_samp_size
                idx_A = edge_index[1][torch.arange(num_edges, dtype=torch.long) * neg_samp_size]
                side_A = embeddings[metapath[-1]][idx_A] @ kernel
                side_A = side_A.repeat_interleave(neg_samp_size, dim=0)
                # logging.info(emb_A)
                score = side_A @ emb_B.t()
            else:
                emb_A = embeddings[metapath[0]][edge_index[0]]
                emb_B = embeddings[metapath[-1]][edge_index[1]]
                score = (emb_A @ kernel) @ emb_B.t()

            score = score.sum(dim=1)
            edge_pred_dict[metapath] = score

        return edge_pred_dict

    def get_edge_index_from_batch(self, pos_batch, neg_batch, mode):
        edge_index_dict = {}

        for metapath, edge_index in pos_batch.items():
            e_size, neg_samp_size = neg_batch[metapath].shape

            if mode == "head":
                nid_A = neg_batch[metapath].reshape(-1)
                nid_B = pos_batch[metapath][1].repeat_interleave(neg_samp_size)
                edge_index_dict[metapath] = torch.stack([nid_A, nid_B], dim=0)
            elif mode == "tail":
                nid_A = pos_batch[metapath][0].repeat_interleave(neg_samp_size)
                nid_B = neg_batch[metapath].reshape(-1)
                edge_index_dict[metapath] = torch.stack([nid_A, nid_B], dim=0)

        return edge_index_dict, neg_samp_size


class LATTELinkPred(LinkPredTrainer):
    def __init__(self, hparams, dataset: HeteroNetDataset, metrics=["obgl-biokg"],
                 collate_fn="neighbor_sampler") -> None:
        super(LATTELinkPred, self).__init__(hparams, dataset, metrics)
        self.head_node_type = dataset.head_node_type
        self.dataset = dataset
        self.multilabel = dataset.multilabel
        self._name = f"LATTE-{hparams.t_order}{' Link' if hparams.use_proximity else ''}"
        self.collate_fn = collate_fn

        self.embedder = LATTE(t_order=hparams.t_order, embedding_dim=hparams.embedding_dim,
                              in_channels_dict=dataset.node_attr_shape, num_nodes_dict=dataset.num_nodes_dict,
                              metapaths=dataset.get_metapaths(), attn_heads=hparams.attn_heads,
                              attn_activation=hparams.attn_activation, attn_dropout=hparams.attn_dropout,
                              use_proximity=True, neg_sampling_ratio=hparams.neg_sampling_ratio)

        self.classifier = DistMulti(embedding_dim=hparams.embedding_dim, metapaths=dataset.get_metapaths())
        self.criterion = LinkPredLoss()

        hparams.embedding_dim = hparams.embedding_dim * hparams.t_order

    def forward(self, inputs: dict, **kwargs):
        embeddings, proximity_loss, _ = self.embedder(inputs["x_dict"],
                                                      edge_index_dict=inputs["edge_index_dict"],
                                                      global_node_idx=inputs["global_node_index"],
                                                      **kwargs)

        output = self.classifier(inputs, embeddings)

        return embeddings, proximity_loss, output

    @DeprecationWarning
    def get_e_pos_neg(self, edge_pred_dict: dict):
        """
        Given pos edges and sampled neg edges from LATTE proximity, align e_pos and e_neg to shape (num_edge, ) and (num_edge, num_nodes_neg). This ignores reverse metapaths.

        :param edge_pred_dict:
        :return:
        """
        e_pos = torch.cat([e_pred for metapath, e_pred in edge_pred_dict.items() \
                           if not is_negative(metapath) and metapath in self.dataset.metapaths], dim=0)
        e_neg = torch.cat([e_pred for metapath, e_pred in edge_pred_dict.items() if
                           is_negative(metapath) and untag_negative(metapath) in self.dataset.metapaths], dim=0)

        if self.training:
            num_nodes_neg = int(self.hparams.neg_sampling_ratio)
        else:
            num_nodes_neg = int(e_neg.numel() // e_pos.numel())

        if e_neg.size(0) % num_nodes_neg:
            e_neg = e_neg[:e_neg.size(0) - e_neg.size(0) % num_nodes_neg]
        e_neg = e_neg.view(-1, num_nodes_neg)

        # ensure same num_edge in dim 0
        min_idx = min(e_pos.size(0), e_neg.size(0))
        e_pos = e_pos[:min_idx]
        e_neg = e_neg[:min_idx]

        return e_pos, e_neg

    def training_step(self, batch, batch_nb):
        X, _, _ = batch

        _, prox_loss, edge_pred_dict = self.forward(X)

        e_pos, e_neg = self.reshape_e_pos_neg(edge_pred_dict)
        loss = self.criterion.forward(e_pos, e_neg)
        if prox_loss is not None:
            loss += prox_loss

        self.train_metrics.update_metrics(e_pos, e_neg, weights=None)
        self.log_dict(self.train_metrics.compute_metrics())

        outputs = {'loss': loss}
        return outputs

    def validation_step(self, batch, batch_nb):
        X, _, _ = batch
        _, prox_loss, edge_pred_dict = self.forward(X)

        e_pos, e_neg = self.reshape_e_pos_neg(edge_pred_dict)
        loss = self.criterion.forward(e_pos, e_neg)
        if prox_loss is not None:
            loss += prox_loss

        self.valid_metrics.update_metrics(e_pos, e_neg, weights=None)

        return {"val_loss": loss}

    def test_step(self, batch, batch_nb):
        X, _, _ = batch
        _, prox_loss, edge_pred_dict = self.forward(X)

        e_pos, e_neg = self.reshape_e_pos_neg(edge_pred_dict)
        loss = self.criterion.forward(e_pos, e_neg)
        if prox_loss is not None:
            loss += prox_loss

        self.test_metrics.update_metrics(e_pos, e_neg, weights=None)

        return {"test_loss": loss}

    def configure_optimizers(self):
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'alpha_activation']
        optimizer_grouped_parameters = [
            {'params': [p for name, p in param_optimizer if not any(key in name for key in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for name, p in param_optimizer if any(key in name for key in no_decay)], 'weight_decay': 0.0}
        ]

        # optimizer = torch.optim.AdamW(optimizer_grouped_parameters, eps=1e-06, lr=self.hparams.lr)
        optimizer = torch.optim.Adam(optimizer_grouped_parameters,
                                     lr=self.hparams.lr,  # momentum=self.hparams.momentum,
                                     weight_decay=self.hparams.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer)

        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
