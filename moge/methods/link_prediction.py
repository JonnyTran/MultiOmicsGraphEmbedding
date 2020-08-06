import multiprocessing
import torch
import pytorch_lightning as pl

from .node_classification import NodeClfMetrics, LATTENodeClassifier
from ..generator.datasets import HeteroNetDataset
from ..generator.sampler.link_sampler import LinkSampler
from ..module.latte import LATTE


class LinkPredMetrics(NodeClfMetrics):
    def __init__(self, hparams, dataset, metrics):
        super(LinkPredMetrics, self).__init__(hparams, dataset, metrics)


class LATTELinkPredictor(LinkPredMetrics):
    def __init__(self, hparams, dataset: HeteroNetDataset, metrics=["obgl-biokg"],
                 collate_fn="neighbor_sampler") -> None:
        super(LATTELinkPredictor, self).__init__(hparams, dataset, metrics)
        self.head_node_type = dataset.head_node_type
        self.dataset = dataset
        self.multilabel = dataset.multilabel
        self._name = f"LATTE-{hparams.t_order}{' link_pred' if hparams.use_proximity_loss else ''}"
        self.collate_fn = collate_fn

        self.latte = LATTE(in_channels_dict=dataset.node_attr_shape, embedding_dim=hparams.embedding_dim,
                           t_order=hparams.t_order, num_nodes_dict=dataset.num_nodes_dict,
                           metapaths=dataset.get_metapaths(), use_proximity_loss=hparams.use_proximity_loss,
                           neg_sampling_ratio=hparams.neg_sampling_ratio)
        hparams.embedding_dim = hparams.embedding_dim * hparams.t_order

    def forward(self, x_dict, global_node_index, edge_index_dict):
        embeddings, proximity_loss, edge_pred_dict = self.latte.forward(x_dict, global_node_index, edge_index_dict)
        return embeddings, proximity_loss, edge_pred_dict

    def training_step(self, batch, batch_nb):
        X, _, _ = batch
        _, loss, edge_pred_dict = self.forward(X["x_dict"], X["global_node_index"], X["edge_index_dict"])
        e_pos = torch.cat([e_pred for metapath, e_pred in edge_pred_dict.items() if "neg" not in metapath], dim=0)
        e_neg = torch.cat([e_pred for metapath, e_pred in edge_pred_dict.items() if "neg" in metapath], dim=0)
        self.train_metrics.update_metrics(e_pos, e_neg, weights=None)

        outputs = {'loss': loss}
        return outputs

    def validation_step(self, batch, batch_nb):
        X, _, _ = batch
        _, loss, edge_pred_dict = self.forward(X["x_dict"], X["global_node_index"], X["edge_index_dict"])
        e_pos = torch.cat([e_pred for metapath, e_pred in edge_pred_dict.items() if "neg" not in metapath], dim=0)
        e_neg = torch.cat([e_pred for metapath, e_pred in edge_pred_dict.items() if "neg" in metapath], dim=0)
        self.valid_metrics.update_metrics(e_pos, e_neg, weights=None)

        return {"val_loss": loss}

    def test_step(self, batch, batch_nb):
        X, y, weights = batch
        y_hat, loss, edge_pred_dict = self.forward(X["x_dict"], X["global_node_index"], X["edge_index_dict"])
        e_pos = torch.cat([e_pred for metapath, e_pred in edge_pred_dict.items() if "neg" not in metapath], dim=0)
        e_neg = torch.cat([e_pred for metapath, e_pred in edge_pred_dict.items() if "neg" in metapath], dim=0)
        self.test_metrics.update_metrics(e_pos, e_neg, weights=None)

        return {"test_loss": loss}

    def train_dataloader(self):
        return self.dataset.train_dataloader(collate_fn=self.collate_fn,
                                             batch_size=self.hparams.batch_size,
                                             num_workers=int(0.8 * multiprocessing.cpu_count()))

    def val_dataloader(self, batch_size=None):
        return self.dataset.val_dataloader(collate_fn=self.collate_fn,
                                           batch_size=self.hparams.batch_size * 2,
                                           num_workers=int(0.2 * multiprocessing.cpu_count()))

    def test_dataloader(self, batch_size=None):
        return self.dataset.test_dataloader(collate_fn=self.collate_fn,
                                            batch_size=self.hparams.batch_size * 2,
                                            num_workers=int(0.2 * multiprocessing.cpu_count()))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
