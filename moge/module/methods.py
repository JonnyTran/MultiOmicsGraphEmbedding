import torch
import pytorch_lightning as pl

from torch_geometric.nn import MetaPath2Vec
from sklearn.linear_model import LogisticRegression

from moge.generator.datasets import HeterogeneousNetworkDataset

class EmbeddingMethod(pl.LightningModule, MetaPath2Vec):
    def __init__(self, hparams):
        self.hparams = hparams
        super(EmbeddingMethod, self).__init__()

    def node_classification(self, training=True):
        if training:
            z = self.forward(self.head_node_type,
                             batch=self.data.y_index_dict[self.head_node_type][self.data.training_idx])
            y = self.data.y_dict[self.head_node_type][self.data.training_idx]

            perm = torch.randperm(z.size(0))
            train_perm = perm[:int(z.size(0) * self.data.train_ratio)]
            test_perm = perm[int(z.size(0) * self.data.train_ratio):]

            clf = LogisticRegression(solver="lbfgs", multi_class="auto", max_iter=150) \
                .fit(z[train_perm].detach().cpu().numpy(),
                     y[train_perm].detach().cpu().numpy())

            accuracy = clf.score(z[test_perm].detach().cpu().numpy(),
                                 y[test_perm].detach().cpu().numpy())
        else:
            z_train = self.forward(self.head_node_type,
                                   batch=self.data.y_index_dict[self.head_node_type][self.data.training_idx])
            y_train = self.data.y_dict[self.head_node_type][self.data.training_idx]

            z_val = self.forward(self.head_node_type,
                                 batch=self.data.y_index_dict[self.head_node_type][self.data.validation_idx])
            y_val = self.data.y_dict[self.head_node_type][self.data.validation_idx]

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
        print(logs)
        return {"progress_bar": logs, "log": logs}


class MetaPath2Vec(MetaPath2Vec, EmbeddingMethod):
    def __init__(self, hparams, dataset: HeterogeneousNetworkDataset):
        # Hparams
        self.train_ratio = hparams.train_ratio
        self.batch_size = hparams.batch_size
        self.sparse = hparams.sparse

        embedding_dim = hparams.embedding_dim
        walk_length = hparams.walk_length
        context_size = hparams.context_size
        walks_per_node = hparams.walks_per_node
        num_negative_samples = hparams.num_negative_samples

        # Dataset
        self.data = dataset
        if hasattr(dataset, "num_nodes_dict"):
            num_nodes_dict = dataset.num_nodes_dict
        else:
            num_nodes_dict = None
        metapath = self.data.metapath
        self.head_node_type = self.data.head_node_type
        edge_index_dict = dataset.edge_index_dict

        super().__init__(edge_index_dict, embedding_dim, metapath, walk_length, context_size, walks_per_node,
                         num_negative_samples, num_nodes_dict, self.sparse)
        self.hparams = hparams

    def sample(self, batch):
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch)
        return self.pos_sample(batch), self.neg_sample(batch)

    def train_dataloader(self):
        return self.data.train_dataloader(self.hparams.batch_size, collate_fn=self.sample)

    def val_dataloader(self):
        return self.data.val_dataloader(self.hparams.batch_size, collate_fn=self.sample)

    def configure_optimizers(self):
        if self.sparse:
            return torch.optim.SparseAdam(self.parameters(), lr=self.hparams.lr)
        else:
            return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
