import multiprocessing
import torch

from .node_classification import NodeClfMetrics
from moge.generator.sampler.datasets import HeteroNetDataset
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
        self._name = f"LATTE-{hparams.t_order}{' Link' if hparams.use_proximity_loss else ''}"
        self.collate_fn = collate_fn
        self.num_nodes_neg = int(hparams.neg_sampling_ratio * (2 if self.dataset.use_reverse else 1))
        self.neg_sampling_test_size = hparams.neg_sampling_test_size

        self.latte = LATTE(in_channels_dict=dataset.node_attr_shape, embedding_dim=hparams.embedding_dim,
                           t_order=hparams.t_order, num_nodes_dict=dataset.num_nodes_dict,
                           metapaths=dataset.get_metapaths(), use_proximity_loss=True,
                           neg_sampling_ratio=hparams.neg_sampling_ratio,
                           neg_sampling_test_size=hparams.neg_sampling_test_size)
        hparams.embedding_dim = hparams.embedding_dim * hparams.t_order

    def forward(self, x_dict, global_node_index, edge_index_dict):
        embeddings, proximity_loss, edge_pred_dict = self.latte.forward(x_dict, global_node_index, edge_index_dict)
        return embeddings, proximity_loss, edge_pred_dict

    def get_e_pos_neg(self, edge_pred_dict, training=True):
        """
        Align e_pos and e_neg to shape shape (num_edge, ) and (num_edge, num_nodes_neg)
        :param edge_pred_dict:
        :return:
        """
        e_pos = torch.cat([e_pred for metapath, e_pred in edge_pred_dict.items() \
                           if "neg" not in metapath and metapath in self.dataset.metapaths], dim=0)
        e_neg = torch.cat([e_pred for metapath, e_pred in edge_pred_dict.items() if "neg" in metapath], dim=0)

        if training:
            num_nodes_neg = self.num_nodes_neg
        else:
            num_nodes_neg = self.neg_sampling_test_size * (2 if self.dataset.use_reverse else 1)

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
        # print("X", {k: v.shape for k, v in X.items()})
        _, loss, edge_pred_dict = self.forward(X["x_dict"], X["global_node_index"], X["edge_index_dict"])
        e_pos, e_neg = self.get_e_pos_neg(edge_pred_dict, training=True)
        self.train_metrics.update_metrics(e_pos, e_neg, weights=None)

        outputs = {'loss': loss}
        return outputs

    def validation_step(self, batch, batch_nb):
        X, _, _ = batch
        # print("X", {k: v.shape for k, v in X.items()})
        _, loss, edge_pred_dict = self.forward(X["x_dict"], X["global_node_index"], X["edge_index_dict"])
        e_pos, e_neg = self.get_e_pos_neg(edge_pred_dict, training=False)
        self.valid_metrics.update_metrics(e_pos, e_neg, weights=None)

        return {"val_loss": loss}

    def test_step(self, batch, batch_nb):
        X, _, _ = batch
        y_hat, loss, edge_pred_dict = self.forward(X["x_dict"], X["global_node_index"], X["edge_index_dict"])
        e_pos, e_neg = self.get_e_pos_neg(edge_pred_dict, training=False)
        self.test_metrics.update_metrics(e_pos, e_neg, weights=None)

        return {"test_loss": loss}

    def train_dataloader(self):
        return self.dataset.train_dataloader(collate_fn=self.collate_fn,
                                             batch_size=self.hparams.batch_size,
                                             num_workers=0)  # int(0.8 * multiprocessing.cpu_count())

    def val_dataloader(self, batch_size=None):
        return self.dataset.val_dataloader(collate_fn=self.collate_fn,
                                           batch_size=self.hparams.batch_size * 2,
                                           num_workers=0)

    def test_dataloader(self, batch_size=None):
        return self.dataset.test_dataloader(collate_fn=self.collate_fn,
                                            batch_size=self.hparams.batch_size * 2,
                                            num_workers=0)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
