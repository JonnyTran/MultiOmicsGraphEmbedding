import itertools, logging

import pandas as pd
import pytorch_lightning as pl
import torch

from moge.module.metrics import Metrics
from moge.evaluation.clustering import clustering_metrics
from moge.module.utils import tensor_sizes

class NodeClfMetrics(pl.LightningModule):
    def __init__(self, hparams, dataset, metrics, *args):
        super().__init__(*args)

        self.train_metrics = Metrics(prefix="", loss_type=hparams.loss_type, n_classes=dataset.n_classes,
                                     multilabel=dataset.multilabel, metrics=metrics)
        self.valid_metrics = Metrics(prefix="val_", loss_type=hparams.loss_type, n_classes=dataset.n_classes,
                                     multilabel=dataset.multilabel, metrics=metrics)
        self.test_metrics = Metrics(prefix="test_", loss_type=hparams.loss_type, n_classes=dataset.n_classes,
                                    multilabel=dataset.multilabel, metrics=metrics)
        hparams.name = self.name()
        hparams.inductive = dataset.inductive
        self.hparams = hparams

    def register_hooks(self):
        # Register a hook for embedding layer
        for name, layer in self.named_children():
            layer.__name__ = name
            print(name)
            layer.register_forward_hook(self.save_embedding)
            layer.register_forward_hook(self.save_pred)

    def save_embedding(self, module, inputs, output):
        if self.training:
            return

        if module.__name__ in ["embedder"]:
            logging.info(
                f"Saved to _embeddings and _inputs @ {module.__name__}, input {tensor_sizes(inputs)}, output {tensor_sizes(output)}")
            self._embeddings = output
            self._inputs = inputs

    def save_pred(self, module, inputs, output):
        if self.training:
            return

        if module.__name__ in ["classifier"]:
            logging.info(
                f"Saved to _y_pred @ {module.__name__}, input {tensor_sizes(inputs)}, output {tensor_sizes(output)}")
            self._y_pred = output

    def name(self):
        if hasattr(self, "_name"):
            return self._name
        else:
            return self.__class__.__name__

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean().item()
        logs = self.train_metrics.compute_metrics()
        # logs = _fix_dp_return_type(logs, device=outputs[0]["loss"].device)

        logs.update({"loss": avg_loss})
        self.train_metrics.reset_metrics()
        self.log_dict(logs)
        return None

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean().item()
        logs = self.valid_metrics.compute_metrics()
        # logs = _fix_dp_return_type(logs, device=outputs[0]["val_loss"].device)
        # print({k: np.around(v.item(), decimals=3) for k, v in logs.items()})

        logs.update({"val_loss": avg_loss})
        self.valid_metrics.reset_metrics()
        self.log_dict(logs, prog_bar=logs)
        return None

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean().item()
        if hasattr(self, "test_metrics"):
            logs = self.test_metrics.compute_metrics()
            self.test_metrics.reset_metrics()
        else:
            logs = {}
        logs.update({"test_loss": avg_loss})

        self.log_dict(logs, prog_bar=logs)
        return None

    def train_dataloader(self):
        return self.dataset.train_dataloader(collate_fn=self.collate_fn, batch_size=self.hparams.batch_size)

    def val_dataloader(self):
        return self.dataset.valid_dataloader(collate_fn=self.collate_fn, batch_size=self.hparams.batch_size)

    def valtrain_dataloader(self):
        return self.dataset.valtrain_dataloader(collate_fn=self.collate_fn,
                                                batch_size=self.hparams.batch_size)

    def test_dataloader(self):
        return self.dataset.test_dataloader(collate_fn=self.collate_fn, batch_size=self.hparams.batch_size)

    def clustering_metrics(self, dataset):
        X_all, y_all, _ = dataset.sample(torch.hstack([dataset.training_idx,
                                                       dataset.validation_idx,
                                                       dataset.testing_idx]), mode="testing")

        X_emb, _, _ = self.forward_emb(X_all["x_dict"], X_all["edge_index_dict"], X_all["global_node_index"])

        embeddings_all, types_all, labels = dataset.get_embeddings_labels(X_emb, X["global_node_index"])

        {k: v.shape for k, v in X_emb.items()}, embeddings_all.shape, y_pred.shape

    def print_pred_class_counts(self, y_hat, y, multilabel, n_top_class=8):
        if multilabel:
            y_pred_dict = pd.Series(y_hat.sum(1).detach().cpu().type(torch.int).numpy()).value_counts().to_dict()
            y_true_dict = pd.Series(y.sum(1).detach().cpu().type(torch.int).numpy()).value_counts().to_dict()
            print(f"y_pred {len(y_pred_dict)} classes",
                  {str(k): v for k, v in itertools.islice(y_pred_dict.items(), n_top_class)})
            print(f"y_true {len(y_true_dict)} classes",
                  {str(k): v for k, v in itertools.islice(y_true_dict.items(), n_top_class)})
        else:
            y_pred_dict = pd.Series(y_hat.argmax(1).detach().cpu().type(torch.int).numpy()).value_counts().to_dict()
            y_true_dict = pd.Series(y.detach().cpu().type(torch.int).numpy()).value_counts().to_dict()
            print(f"y_pred {len(y_pred_dict)} classes",
                  {str(k): v for k, v in itertools.islice(y_pred_dict.items(), n_top_class)})
            print(f"y_true {len(y_true_dict)} classes",
                  {str(k): v for k, v in itertools.islice(y_true_dict.items(), n_top_class)})

    def get_n_params(self):
        size = 0
        for name, param in dict(self.named_parameters()).items():
            nn = 1
            for s in list(param.size()):
                nn = nn * s
            size += nn
        return size
