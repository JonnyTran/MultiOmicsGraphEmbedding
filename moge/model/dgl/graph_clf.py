import multiprocessing

import dgl
import dgl.nn.pytorch as dglnn
import torch
from dgl.heterograph import DGLHeteroGraph
from torch.optim.lr_scheduler import ReduceLROnPlateau

from moge.model.classifier import DenseClassification
from moge.model.dgl.latte import LATTE
from moge.model.losses import ClassificationLoss
from .pooling import SAGPool
from ..trainer import GraphClfTrainer, print_pred_class_counts


class LATTEGraphClassifier(GraphClfTrainer):
    def __init__(self, hparams, dataset, metrics, *args, **kwargs):
        super(LATTEGraphClassifier, self).__init__(hparams, dataset, metrics, *args, **kwargs)

        self.dataset = dataset
        self.multilabel = dataset.multilabel
        self._name = f"LATTE-{hparams.n_layers}{' proximity' if hparams.use_proximity else ''}"
        self.collate_fn = None

        self.embedder = LATTE(n_layers=hparams.n_layers, t_order=hparams.t_order, embedding_dim=hparams.embedding_dim,
                              num_nodes_dict=dataset.num_nodes_dict, head_node_type=dataset.head_node_type,
                              metapaths=dataset.get_metapaths(),
                              activation=hparams.activation, attn_heads=hparams.attn_heads,
                              attn_activation=hparams.attn_activation, attn_dropout=hparams.attn_dropout)

        self.pooling = SAGPool(in_dim=hparams.embedding_dim,
                               conv_layer=dglnn.GraphConv(in_feats=hparams.embedding_dim,
                                                          out_feats=1,
                                                          allow_zero_in_degree=True),
                               ratio=0.5,
                               non_linearity=torch.relu)

        # self.pooling = DiffPoolBatchedGraphLayer(input_dim=hparams.embedding_dim,
        #                                          assign_dim=32,
        #                                          output_feat_dim=hparams.embedding_dim,
        #                                          activation=torch.relu,
        #                                          dropout=hparams.attn_dropout,
        #                                          aggregator_type="mean",
        #                                          link_pred=False)
        self.readout = hparams.readout

        if "batchnorm" in hparams and hparams.layernorm:
            self.batchnorm = torch.nn.BatchNorm1d(hparams.embedding_dim)

        self.classifier = DenseClassification(hparams)
        self.criterion = ClassificationLoss(loss_type=hparams.loss_type, n_classes=dataset.n_classes,
                                            class_weight=dataset.class_weight if hasattr(dataset, "class_weight") and \
                                                                                 hparams.use_class_weights else None,
                                            multilabel=dataset.multilabel)
        self.hparams.n_params = self.get_n_params()

    def forward(self, multigraph: DGLHeteroGraph, feat, **kwargs):
        embeddings = self.embedder.forward(multigraph, feat, **kwargs)

        multigraph, feature, perm = self.pooling(multigraph, embeddings[self.dataset.head_node_type])
        multigraph.ndata["feature"] = feature

        # adj_new, feature = self.pooling(multigraph, embeddings[self.dataset.head_node_type])
        # multigraph.ndata["feature"] = feature

        graph_emb = dgl.readout_nodes(multigraph, 'feature', op=self.readout)

        if hasattr(self, "batchnorm"):
            graph_emb = self.batchnorm(graph_emb)

        y_hat = self.classifier.forward(graph_emb)
        return y_hat

    def training_step(self, batch, batch_nb):
        graphs, labels = batch
        graphs = graphs.to(self.device)
        input_feat = {self.dataset.head_node_type: graphs.ndata["feat"]}

        y_hat = self.forward(graphs, input_feat)
        loss = self.criterion.forward(y_hat, labels)
        self.train_metrics.update_metrics(y_hat, labels, weights=None)

        logs = self.train_metrics.compute_metrics() if batch_nb % 50 == 0 else {}
        outputs = {'loss': loss}
        if logs is not None:
            outputs.update({'progress_bar': logs, "logs": logs})
        return outputs

    def validation_step(self, batch, batch_nb):
        graphs, labels = batch
        graphs = graphs.to(self.device)
        input_feat = {self.dataset.head_node_type: graphs.ndata["feat"]}

        y_hat = self.forward(graphs, input_feat)
        val_loss = self.criterion.forward(y_hat, labels)
        self.valid_metrics.update_metrics(y_hat, labels, weights=None)

        return {"val_loss": val_loss}

    def test_step(self, batch, batch_nb):
        graphs, labels = batch
        graphs = graphs.to(self.device)
        input_feat = {self.dataset.head_node_type: graphs.ndata["feat"]}

        y_hat = self.forward(graphs, input_feat)
        test_loss = self.criterion.forward(y_hat, labels)

        if batch_nb == 0:
            print_pred_class_counts(y_hat, labels, multilabel=self.dataset.multilabel)
        self.test_metrics.update_metrics(y_hat, labels, weights=None)

        return {"test_loss": test_loss}

    def train_dataloader(self):
        return self.dataset.train_dataloader(collate_fn=None,
                                             batch_size=self.hparams.batch_size,
                                             num_workers=int(0.4 * multiprocessing.cpu_count()))

    def val_dataloader(self, batch_size=None):
        return self.dataset.valid_dataloader(collate_fn=None,
                                             batch_size=self.hparams.batch_size,
                                             num_workers=max(1, int(0.1 * multiprocessing.cpu_count())))

    def valtrain_dataloader(self):
        return self.dataset.valtrain_dataloader(collate_fn=None,
                                                batch_size=self.hparams.batch_size,
                                                num_workers=max(1, int(0.1 * multiprocessing.cpu_count())))

    def test_dataloader(self, batch_size=None):
        return self.dataset.test_dataloader(collate_fn=None,
                                            batch_size=self.hparams.batch_size,
                                            num_workers=max(1, int(0.1 * multiprocessing.cpu_count())))

    def configure_optimizers(self):
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'alpha_activation']
        optimizer_grouped_parameters = [
            {'params': [p for name, p in param_optimizer if not any(key in name for key in no_decay)],
             'weight_decay': self.hparams.weight_decay},
            {'params': [p for name, p in param_optimizer if any(key in name for key in no_decay)], 'weight_decay': 0.0}
        ]

        # optimizer = torch.optim.AdamW(optimizer_grouped_parameters, eps=1e-06, lr=self.hparams.lr)
        optimizer = torch.optim.Adam(optimizer_grouped_parameters,
                                     lr=self.hparams.lr,  # momentum=self.hparams.momentum,
                                     weight_decay=self.hparams.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer)

        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
