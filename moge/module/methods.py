from typing import Optional, Union, Sequence, Dict, Tuple, List

import torch
import pytorch_lightning as pl
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader

from torch_geometric.nn import MetaPath2Vec


class MetaPath2Vec(MetaPath2Vec, pl.LightningModule):
    def __init__(self, dataset, embedding_dim, metapath, walk_length, context_size, walks_per_node=1,
                 num_negative_samples=1, num_nodes_dict=None, sparse=False, train_ratio=0.8):
        self.train_ratio = train_ratio
        self.data = dataset
        edge_index_dict = dataset.edge_index_dict
        super().__init__(edge_index_dict, embedding_dim, metapath, walk_length, context_size, walks_per_node,
                         num_negative_samples, num_nodes_dict, sparse)

    def node_classification(self):
        z = self.forward('author', batch=self.data.y_index_dict['author'])
        y = self.data.y_dict['author']

        perm = torch.randperm(z.size(0))
        train_perm = perm[:int(z.size(0) * self.train_ratio)]
        test_perm = perm[int(z.size(0) * self.train_ratio):]

        return self.test(z[train_perm], y[train_perm], z[test_perm], y[test_perm],
                         max_iter=150)

    def training_step(self, batch, batch_nb):
        pos_rw, neg_rw = batch

        loss = self.loss(pos_rw, neg_rw)

        return {'loss': loss}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean().item()
        logs = {"loss": avg_loss, "accuracy": self.node_classification}

        return {"log": logs}

    def validation_step(self, batch, batch_nb):
        pos_rw, neg_rw = batch
        loss = self.loss(pos_rw, neg_rw)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.cat([x["val_loss"] for x in outputs]).mean().item()
        logs = {"val_loss": avg_loss}

        return {"progress_bar": logs, "log": logs}

    def train_dataloader(self):
        training_idx = torch.range(0, int(self.y_index_dict["author"].size(0) * self.train_ratio))
        loader = DataLoader(training_idx, batch_size=self.batch_size,
                            shuffle=True, collate_fn=self.sample, num_workers=12)
        return loader

    def val_dataloader(self):
        validation_idx = torch.arange(int(self.y_index_dict["author"].size(0) * self.train_ratio),
                                      self.y_index_dict["author"].size(0))
        loader = DataLoader(validation_idx, batch_size=self.batch_size,
                            shuffle=True, num_workers=12, collate_fn=self.sample)
        return loader

    def configure_optimizers(self):
        optimizer = torch.optim.SparseAdam(self.parameters(), lr=0.01)
        return optimizer
