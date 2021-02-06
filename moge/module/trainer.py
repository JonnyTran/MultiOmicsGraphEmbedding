import itertools
import logging
from abc import ABCMeta

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning import LightningModule

from .metrics import Metrics
from .utils import tensor_sizes, preprocess_input
from ..evaluation.clustering import clustering_metrics


class ClusteringEvaluator(LightningModule):
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

    def clustering_metrics(self, n_runs=10, compare_node_types=True):
        loader = self.trainvalidtest_dataloader()
        X_all, y_all, _ = next(iter(loader))
        self.cpu().forward(preprocess_input(X_all, device="cpu"))

        if not isinstance(self._embeddings, dict):
            self._embeddings = {list(self._node_ids.keys())[0]: self._embeddings}

        embeddings_all, types_all, y_true = self.dataset.get_embeddings_labels(self._embeddings, self._node_ids)

        # Record metrics for each run in a list of dict's
        res = [{}, ] * n_runs
        for i in range(n_runs):
            y_pred = self.dataset.predict_cluster(n_clusters=len(y_all.unique()), seed=i)

            if compare_node_types and len(self.dataset.node_types) > 1:
                res[i].update(clustering_metrics(y_true=types_all,
                                                 # Match y_pred to type_all's index
                                                 y_pred=types_all.index.map(lambda idx: y_pred.get(idx, "")),
                                                 metrics=["homogeneity_ntype", "completeness_ntype", "nmi_ntype"]))

            if y_pred.shape[0] != y_true.shape[0]:
                y_pred = y_pred.loc[y_true.index]
            res[i].update(clustering_metrics(y_true,
                                             y_pred,
                                             metrics=["homogeneity", "completeness", "nmi"]))

        res_df = pd.DataFrame(res)
        metrics = res_df.mean(0).to_dict()
        return metrics

class NodeClfTrainer(ClusteringEvaluator):
    def __init__(self, hparams, dataset, metrics, *args, **kwargs):
        super().__init__(*args, **kwargs)

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


class LinkPredTrainer(NodeClfTrainer):
    def __init__(self, hparams, dataset, metrics, *args, **kwargs):
        super(LinkPredTrainer, self).__init__(hparams, dataset, metrics, *args, **kwargs)

    def reshape_e_pos_neg(self, edge_pred_dict):
        e_pos = []
        e_neg = []
        for metapath, edge_pred in edge_pred_dict["edge_pos"].items():
            num_edges = edge_pred.shape[0]
            e_pos.append(edge_pred)

            # Negative sampling
            if "head-batch" in edge_pred_dict:
                e_pred_head = edge_pred_dict["head-batch"][metapath].reshape(num_edges, -1)
                e_pred_tail = edge_pred_dict["tail-batch"][metapath].reshape(num_edges, -1)
                e_neg.append(torch.cat([e_pred_head, e_pred_tail], dim=1))

            # True negatives
            elif "edge_neg" in edge_pred_dict:
                e_neg.append(edge_pred_dict["edge_neg"][metapath].reshape(num_edges, -1))
            else:
                raise Exception(f"No negative edges in {edge_pred_dict.keys()}")

        e_pos = torch.cat(e_pos, dim=0)
        e_neg = torch.cat(e_neg, dim=0)

        return e_pos, e_neg

    def train_dataloader(self):
        return self.dataset.train_dataloader(collate_fn=self.collate_fn, batch_size=self.hparams.batch_size)

    def valtrain_dataloader(self):
        return self.dataset.valtrain_dataloader(collate_fn=self.collate_fn,
                                                batch_size=self.hparams.batch_size)

    def val_dataloader(self):
        return self.dataset.valid_dataloader(collate_fn=self.collate_fn, batch_size=self.hparams.batch_size // 10)

    def test_dataloader(self):
        return self.dataset.test_dataloader(collate_fn=self.collate_fn,
                                            batch_size=self.hparams.batch_size // 10)
