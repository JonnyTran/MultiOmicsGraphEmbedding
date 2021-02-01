import itertools, logging

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning import LightningModule

from moge.module.metrics import Metrics
from moge.evaluation.clustering import clustering_metrics
from moge.module.utils import tensor_sizes, preprocess_input


class ClusteringMetrics(LightningModule):
    def register_hooks(self):
        # Register hooks for embedding layer and classifier layer
        for name, layer in self.named_children():
            layer.__name__ = name
            print(name)
            layer.register_forward_hook(self.save_embedding)
            layer.register_forward_hook(self.save_pred)

        # def save_node_ids(module, inputs):
        #     # if module.training: return
        #     logging.info(f"save_node_ids @ {module.__name__} {tensor_sizes(inputs)}")
        #
        # # Register a hook to get node_ids input
        # for layer in itertools.islice(self.modules(), 1):
        #     print(layer.name())
        #     layer.register_forward_pre_hook(save_node_ids)

    def save_embedding(self, module, _, outputs):
        if self.training:
            return
        logging.info(module.__name__)

        if module.__name__ == "embedder":
            logging.info(f"save_embedding @ {module.__name__}")

            if isinstance(outputs, (list, tuple)):
                self._embeddings = outputs[0]
            else:
                self._embeddings = outputs

    def save_pred(self, module, _, outputs):
        if self.training:
            return

        if module.__name__ in ["classifier"]:
            logging.info(
                f"save_pred @ {module.__name__}, output {tensor_sizes(outputs)}")

            if isinstance(outputs, (list, tuple)):
                self._y_pred = outputs[0]
            else:
                self._y_pred = outputs

    def trainvalidtest_dataloader(self):
        return self.dataset.trainvalidtest_dataloader(collate_fn=self.collate_fn, )

    def clustering_metrics(self):
        loader = self.trainvalidtest_dataloader()
        X_all, y_all, _ = next(iter(loader))

        self.forward(preprocess_input(X_all, device=self.device))

        if not isinstance(self._embeddings, dict):
            self._embeddings = {list(self._node_ids.keys())[0]: self._embeddings}

        logging.info(f"Embeddings {tensor_sizes(self._embeddings)}, node_ids {tensor_sizes(self._node_ids)}")
        embeddings_all, types_all, labels = self.dataset.get_embeddings_labels(self._embeddings, self._node_ids)

        y_pred = self.dataset.predict_cluster(n_clusters=len(y_all.unique()))
        return clustering_metrics(labels, y_pred, metrics=["homogeneity", "completeness", "nmi"])


class NodeClfMetrics(ClusteringMetrics):
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
