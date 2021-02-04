import multiprocessing
from typing import Callable

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.hooks import RemovableHandle

from moge.generator import HeteroNetDataset
from moge.module.PyG.latte import LATTE, untag_negative, is_negative
from ..trainer import LinkPredTrainer


class DistMulti(torch.nn.Module):
    def __init__(self, embedding_dim, metapaths):
        super(DistMulti, self).__init__()

    def forward(self):
        pass


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
        hparams.embedding_dim = hparams.embedding_dim * hparams.t_order

    def forward(self, inputs: dict, **kwargs):
        embeddings, proximity_loss, edge_pred_dict = self.embedder(inputs["x_dict"],
                                                                   edge_index_dict=inputs["edge_index_dict"],
                                                                   global_node_idx=inputs["global_node_index"],
                                                                   **kwargs)
        return embeddings, proximity_loss, edge_pred_dict

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

        _, loss, edge_pred_dict = self.forward(X)
        e_pos, e_neg = self.get_e_pos_neg(edge_pred_dict)
        self.train_metrics.update_metrics(e_pos, e_neg, weights=None)

        outputs = {'loss': loss, **self.train_metrics.compute_metrics()}
        return outputs

    def validation_step(self, batch, batch_nb):
        X, _, _ = batch

        _, loss, edge_pred_dict = self.forward(X)
        e_pos, e_neg = self.get_e_pos_neg(edge_pred_dict)
        self.valid_metrics.update_metrics(e_pos, e_neg, weights=None)

        return {"val_loss": loss}

    def test_step(self, batch, batch_nb):
        X, _, _ = batch
        y_hat, loss, edge_pred_dict = self.forward(X)
        e_pos, e_neg = self.get_e_pos_neg(edge_pred_dict)
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
