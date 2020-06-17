from typing import Optional, Union, Sequence, Dict, Tuple, List

import torch
import pytorch_lightning as pl
from torch import Tensor
from torch.utils.data import DataLoader

from torch_geometric.nn import MetaPath2Vec
from sklearn.linear_model import LogisticRegression


class MetaPath2Vec(MetaPath2Vec, pl.LightningModule):
    def __init__(self, hparams, dataset, metapath):
        self.data = dataset
        if hasattr(dataset, "num_nodes_dict"):
            num_nodes_dict = dataset.num_nodes_dict

        self.train_ratio = hparams.train_ratio
        self.batch_size = hparams.batch_size
        self.sparse = hparams.sparse

        embedding_dim = hparams.embedding_dim
        walk_length = hparams.walk_length
        context_size = hparams.context_size
        walks_per_node = hparams.walks_per_node
        num_negative_samples = hparams.num_negative_samples

        self.head_node_type = self.data.head_node_type
        perm = torch.randperm(self.data.y_index_dict[self.head_node_type].size(0))
        self.training_idx = perm[:int(self.data.y_index_dict[self.head_node_type].size(0) * self.train_ratio)]
        self.validation_idx = perm[int(self.data.y_index_dict[self.head_node_type].size(0) * self.train_ratio):]

        edge_index_dict = dataset.edge_index_dict
        super().__init__(edge_index_dict, embedding_dim, metapath, walk_length, context_size, walks_per_node,
                         num_negative_samples, num_nodes_dict, self.sparse)
        self.hparams = hparams

    def node_classification(self, training=True):
        if training:
            z = self.forward(self.head_node_type, batch=self.data.y_index_dict[self.head_node_type][self.training_idx])
            y = self.data.y_dict[self.head_node_type][self.training_idx]

            perm = torch.randperm(z.size(0))
            train_perm = perm[:int(z.size(0) * self.train_ratio)]
            test_perm = perm[int(z.size(0) * self.train_ratio):]

            clf = LogisticRegression(solver="lbfgs", multi_class="auto", max_iter=150) \
                .fit(z[train_perm].detach().cpu().numpy(),
                     y[train_perm].detach().cpu().numpy())

            accuracy = clf.score(z[test_perm].detach().cpu().numpy(),
                                 y[test_perm].detach().cpu().numpy())
        else:
            z_train = self.forward(self.head_node_type,
                                   batch=self.data.y_index_dict[self.head_node_type][self.training_idx])
            y_train = self.data.y_dict[self.head_node_type][self.training_idx]

            z_val = self.forward(self.head_node_type,
                                 batch=self.data.y_index_dict[self.head_node_type][self.validation_idx])
            y_val = self.data.y_dict[self.head_node_type][self.validation_idx]

            clf = LogisticRegression(solver="lbfgs", multi_class="auto", max_iter=150) \
                .fit(z_train.detach().cpu().numpy(),
                     y_train.detach().cpu().numpy())

            accuracy = clf.score(z_val.detach().cpu().numpy(),
                                 y_val.detach().cpu().numpy())

        return accuracy

    def training_step(self, batch, batch_nb):
        pos_rw, neg_rw = batch
        loss = self.loss(pos_rw, neg_rw)
        return {'loss': loss}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).sum().item()

        return {"progress_bar": {"accuracy": self.node_classification(training=True)},
                "log": {"loss": avg_loss, "accuracy": self.node_classification(training=True)}}

    def validation_step(self, batch, batch_nb):
        pos_rw, neg_rw = batch
        loss = self.loss(pos_rw, neg_rw)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).sum().item()
        logs = {"val_loss": avg_loss,
                "val_accuracy": self.node_classification(training=False)}

        return {"progress_bar": logs, "log": logs}

    def train_dataloader(self):
        loader = DataLoader(self.training_idx, batch_size=self.batch_size,
                            shuffle=True, collate_fn=self.sample, num_workers=12)
        return loader

    def val_dataloader(self):
        loader = DataLoader(self.validation_idx, batch_size=self.batch_size,
                            shuffle=False, num_workers=4, collate_fn=self.sample)
        return loader

    def configure_optimizers(self):
        if self.sparse:
            return torch.optim.SparseAdam(self.parameters(), lr=self.hparams.lr)
        else:
            return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
