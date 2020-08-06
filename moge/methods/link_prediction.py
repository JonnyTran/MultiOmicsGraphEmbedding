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
        super(LATTELinkPredictor).__init__(hparams, dataset, metrics)
        self.head_node_type = dataset.head_node_type
        self.dataset = dataset
        self.multilabel = dataset.multilabel
        self._name = f"LATTE-{hparams.t_order}{' proximity' if hparams.use_proximity_loss else ''}"
        self.collate_fn = collate_fn

        self.latte = LATTE(in_channels_dict=dataset.node_attr_shape, embedding_dim=hparams.embedding_dim,
                           t_order=hparams.t_order, num_nodes_dict=dataset.num_nodes_dict,
                           metapaths=dataset.get_metapaths(), use_proximity_loss=hparams.use_proximity_loss,
                           neg_sampling_ratio=hparams.neg_sampling_ratio)
        hparams.embedding_dim = hparams.embedding_dim * hparams.t_order

    def forward(self, x_dict, global_node_index, edge_index_dict):
        embeddings, proximity_loss = self.latte.forward(x_dict, global_node_index, edge_index_dict)
        return proximity_loss

    def training_step(self, batch, batch_nb):
        X, y, weights = batch
        _, loss = self.forward(X["x_dict"], X["global_node_index"], X["edge_index_dict"])

        # self.train_metrics.update_metrics(y_hat, y, weights=None)

        outputs = {'loss': loss}
        return outputs

    def validation_step(self, batch, batch_nb):
        X, y, weights = batch
        _, loss = self.forward(X["x_dict"], X["global_node_index"], X["edge_index_dict"])
        # self.valid_metrics.update_metrics(y_hat, y, weights=None)

        return {"val_loss": loss}

    def test_step(self, batch, batch_nb):
        X, y, weights = batch
        y_hat, loss = self.forward(X["x_dict"], X["global_node_index"], X["edge_index_dict"])

        if batch_nb == 0:
            self.print_pred_class_counts(y_hat, y, multilabel=self.dataset.multilabel)
        # self.test_metrics.update_metrics(y_hat, y, weights=None)

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
