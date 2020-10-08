import multiprocessing

import torch

from moge.generator import DGLNodeSampler
from moge.module.classifier import DenseClassification
from moge.module.dgl.latte import LATTE
from moge.module.losses import ClassificationLoss
from moge.module.utils import filter_samples
from ..trainer import NodeClfMetrics


class LATTENodeClassifier(NodeClfMetrics):
    def __init__(self, hparams, dataset: DGLNodeSampler, metrics=["accuracy"], collate_fn="neighbor_sampler") -> None:
        super(LATTENodeClassifier, self).__init__(hparams=hparams, dataset=dataset, metrics=metrics)
        self.head_node_type = dataset.head_node_type
        self.dataset = dataset
        self.multilabel = dataset.multilabel
        self.y_types = list(dataset.y_dict.keys())
        self._name = f"LATTE-{hparams.t_order}{' proximity' if hparams.use_proximity else ''}"
        self.collate_fn = collate_fn

        self.latte = LATTE(t_order=hparams.t_order, embedding_dim=hparams.embedding_dim,
                           in_channels_dict=dataset.node_attr_shape, num_nodes_dict=dataset.num_nodes_dict,
                           metapaths=dataset.get_metapaths(), activation=hparams.activation,
                           attn_heads=hparams.attn_heads, attn_activation=hparams.attn_activation,
                           attn_dropout=hparams.attn_dropout, use_proximity=hparams.use_proximity,
                           neg_sampling_ratio=hparams.neg_sampling_ratio)
        hparams.embedding_dim = hparams.embedding_dim * hparams.t_order

        self.classifier = DenseClassification(hparams)
        # self.classifier = MulticlassClassification(num_feature=hparams.embedding_dim,
        #                                            num_class=hparams.n_classes,
        #                                            loss_type=hparams.loss_type)
        self.criterion = ClassificationLoss(n_classes=dataset.n_classes,
                                            class_weight=dataset.class_weight if hasattr(dataset, "class_weight") and \
                                                                                 hparams.use_class_weights else None,
                                            loss_type=hparams.loss_type,
                                            multilabel=dataset.multilabel)
        self.hparams.n_params = self.get_n_params()

    def forward(self, input: dict, **kwargs):
        embeddings, proximity_loss, _ = self.latte.forward(X=input["x_dict"], edge_index_dict=input["edge_index_dict"],
                                                           global_node_idx=input["global_node_index"], **kwargs)
        y_hat = self.classifier.forward(embeddings[self.head_node_type])
        return y_hat, proximity_loss

    def training_step(self, batch, batch_nb):
        input_nodes, seeds, blocks = batch
        y_hat, proximity_loss = self.forward(X)


        loss = self.criterion.forward(y_hat, y)

        self.train_metrics.update_metrics(y_hat, y, weights=None)

        logs = None
        if self.hparams.use_proximity:
            loss = loss + proximity_loss
            logs = {"proximity_loss": proximity_loss}

        outputs = {'loss': loss}
        if logs is not None:
            outputs.update({'progress_bar': logs, "logs": logs})
        return outputs

    def validation_step(self, batch, batch_nb):
        input_nodes, seeds, blocks = batch
        y_hat, proximity_loss = self.forward(X)

        val_loss = self.criterion.forward(y_hat, y)
        # if batch_nb == 0:
        #     self.print_pred_class_counts(y_hat, y, multilabel=self.dataset.multilabel)

        self.valid_metrics.update_metrics(y_hat, y, weights=None)

        if self.hparams.use_proximity:
            val_loss = val_loss + proximity_loss

        return {"val_loss": val_loss}

    def test_step(self, batch, batch_nb):
        X, y, weights = batch
        y_hat, proximity_loss = self.forward(X, save_betas=True)
        if isinstance(y, dict) and len(y) > 1:
            y = y[self.head_node_type]
        y_hat, y = filter_samples(Y_hat=y_hat, Y=y, weights=weights)
        test_loss = self.criterion(y_hat, y)

        if batch_nb == 0:
            self.print_pred_class_counts(y_hat, y, multilabel=self.dataset.multilabel)

        self.test_metrics.update_metrics(y_hat, y, weights=None)

        if self.hparams.use_proximity:
            test_loss = test_loss + proximity_loss

        return {"test_loss": test_loss}

    def train_dataloader(self):
        return self.dataset.train_dataloader(collate_fn=self.collate_fn,
                                             batch_size=self.hparams.batch_size,
                                             num_workers=int(0.4 * multiprocessing.cpu_count()))

    def val_dataloader(self, batch_size=None):
        return self.dataset.valid_dataloader(collate_fn=self.collate_fn,
                                             batch_size=self.hparams.batch_size,
                                             num_workers=max(1, int(0.1 * multiprocessing.cpu_count())))

    def valtrain_dataloader(self):
        return self.dataset.valtrain_dataloader(collate_fn=self.collate_fn,
                                                batch_size=self.hparams.batch_size,
                                                num_workers=max(1, int(0.1 * multiprocessing.cpu_count())))

    def test_dataloader(self, batch_size=None):
        return self.dataset.test_dataloader(collate_fn=self.collate_fn,
                                            batch_size=self.hparams.batch_size,
                                            num_workers=max(1, int(0.1 * multiprocessing.cpu_count())))

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

        return [optimizer], [scheduler]
