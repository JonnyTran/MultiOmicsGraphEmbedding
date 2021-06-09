import itertools
import logging

import pandas as pd
import torch
from torch.utils.data.distributed import DistributedSampler

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
            y_pred = self.dataset.predict_cluster(n_clusters=len(y_true.unique()), seed=i)

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
        self._set_hparams(hparams)

    def name(self):
        if hasattr(self, "_name"):
            return self._name
        else:
            return self.__class__.__name__

    def training_epoch_end(self, outputs):
        logs = self.train_metrics.compute_metrics()
        self.log_dict(logs, prog_bar=True)
        self.train_metrics.reset_metrics()
        return None

    def validation_epoch_end(self, outputs):
        logs = self.valid_metrics.compute_metrics()

        if hasattr(self, "val_moving_loss"):
            val_loss = torch.stack([l for l in outputs]).mean()
            self.val_moving_loss[self.current_epoch % self.val_moving_loss.numel()] = val_loss
            self.log("val_moving_loss", self.val_moving_loss.mean(),
                     logger=True, prog_bar=False, on_epoch=True)

        self.log_dict(logs, prog_bar=True)

        self.valid_metrics.reset_metrics()
        return None

    def test_epoch_end(self, outputs):
        logs = self.test_metrics.compute_metrics()
        self.log_dict(logs, prog_bar=True)
        self.test_metrics.reset_metrics()
        return None

    def train_dataloader(self):
        if hasattr(self.hparams, "num_gpus") and self.hparams.num_gpus > 1:
            train_sampler = DistributedSampler(self.dataset.training_idx, num_replicas=self.hparams.num_gpus,
                                               rank=self.local_rank)
        else:
            train_sampler = None

        dataset = self.dataset.train_dataloader(collate_fn=self.collate_fn,
                                                batch_size=self.hparams.batch_size, batch_sampler=train_sampler)
        return dataset

    def val_dataloader(self):
        if hasattr(self.hparams, "num_gpus") and self.hparams.num_gpus > 1:
            train_sampler = DistributedSampler(self.dataset.validation_idx, num_replicas=self.hparams.num_gpus,
                                               rank=self.local_rank)
        else:
            train_sampler = None

        dataset = self.dataset.valid_dataloader(collate_fn=self.collate_fn,
                                                batch_size=self.hparams.batch_size, batch_sampler=train_sampler)

        return dataset

    def valtrain_dataloader(self):
        dataset = self.dataset.valtrain_dataloader(collate_fn=self.collate_fn,
                                                   batch_size=self.hparams.batch_size)
        return dataset

    def test_dataloader(self):
        if hasattr(self.hparams, "num_gpus") and self.hparams.num_gpus > 1:
            train_sampler = DistributedSampler(self.dataset.testing_idx, num_replicas=self.hparams.num_gpus,
                                               rank=self.local_rank)
        else:
            train_sampler = None

        dataset = self.dataset.test_dataloader(collate_fn=self.collate_fn,
                                               batch_size=self.hparams.batch_size, batch_sampler=train_sampler)
        return dataset


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

        self.test_batch_size = 2048

    def reshape_e_pos_neg(self, edge_pred_dict, edge_weights_dict=None):
        e_pos = []
        e_neg = []
        e_weights = []

        for metapath, edge_pred in edge_pred_dict["edge_pos"].items():
            num_edges = edge_pred.shape[0]

            # Positive edges
            e_pos.append(edge_pred)

            if edge_weights_dict:
                e_weights.append(edge_weights_dict[metapath])

            # Negative sampling
            if "head-batch" in edge_pred_dict:
                e_pred_head = edge_pred_dict["head-batch"][metapath].view(num_edges, -1)
                e_pred_tail = edge_pred_dict["tail-batch"][metapath].view(num_edges, -1)
                e_neg.append(torch.cat([e_pred_head, e_pred_tail], dim=1))

            # True negatives
            elif "edge_neg" in edge_pred_dict:
                e_neg.append(edge_pred_dict["edge_neg"][metapath].view(num_edges, -1))
            else:
                raise Exception(f"No negative edges in {edge_pred_dict.keys()}")

        e_pos = torch.cat(e_pos, dim=0)
        e_neg = torch.cat(e_neg, dim=0)

        if e_weights:
            e_weights = torch.cat(e_weights, dim=0)
        else:
            e_weights = None

        return e_pos, e_neg, e_weights

    def train_dataloader(self):
        return self.dataset.train_dataloader(collate_fn=self.collate_fn,
                                             batch_size=self.hparams.batch_size)

    def valtrain_dataloader(self):
        return self.dataset.valtrain_dataloader(collate_fn=self.collate_fn,
                                                batch_size=self.hparams.batch_size)

    def val_dataloader(self):
        if self.dataset.name() in ["ogbl-biokg", "ogbl-wikikg"]:
            batch_size = self.test_batch_size
        else:
            batch_size = self.hparams.batch_size
        return self.dataset.valid_dataloader(collate_fn=self.collate_fn,
                                             batch_size=batch_size)

    def test_dataloader(self):
        if self.dataset.name() in ["ogbl-biokg", "ogbl-wikikg"]:
            batch_size = self.test_batch_size
        else:
            batch_size = self.hparams.batch_size
        return self.dataset.test_dataloader(collate_fn=self.collate_fn,
                                            batch_size=batch_size)


class GraphClfTrainer(LightningModule):
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

    def get_n_params(self):
        size = 0
        for name, param in dict(self.named_parameters()).items():
            nn = 1
            for s in list(param.size()):
                nn = nn * s
            size += nn
        return size

    def training_epoch_end(self, outputs):
        logs = self.train_metrics.compute_metrics()
        self.train_metrics.reset_metrics()
        self.log_dict(logs, logger=True, prog_bar=True)
        return None

    def validation_epoch_end(self, outputs):
        logs = self.valid_metrics.compute_metrics()
        # print({k: np.around(v.item(), decimals=3) for k, v in logs.items()})

        self.valid_metrics.reset_metrics()
        self.log_dict(logs, logger=True, prog_bar=True)
        return None

    def test_epoch_end(self, outputs):
        if hasattr(self, "test_metrics"):
            logs = self.test_metrics.compute_metrics()
            self.test_metrics.reset_metrics()
        else:
            logs = {}

        self.log_dict(logs, logger=True, prog_bar=True)
        return None

    def train_dataloader(self):
        return self.dataset.train_dataloader(collate_fn=self.collate_fn,
                                             batch_size=self.hparams.batch_size)

    def val_dataloader(self):
        return self.dataset.valid_dataloader(collate_fn=self.collate_fn,
                                             batch_size=self.hparams.batch_size)

    def valtrain_dataloader(self):
        return self.dataset.valtrain_dataloader(collate_fn=self.collate_fn,
                                                batch_size=self.hparams.batch_size)

    def test_dataloader(self):
        return self.dataset.test_dataloader(collate_fn=self.collate_fn,
                                            batch_size=self.hparams.batch_size)


def print_pred_class_counts(y_pred, y_true, multilabel, n_top_class=8):
    if (y_pred < 0.0).any():
        y_pred = torch.sigmoid(y_pred.clone()) if multilabel else torch.softmax(y_pred.clone(), dim=-1)

    if multilabel:
        y_pred_dict = pd.Series(y_pred.sum(1).detach().cpu().type(torch.int).numpy()).value_counts().to_dict()
        y_true_dict = pd.Series(y_true.sum(1).detach().cpu().type(torch.int).numpy()).value_counts().to_dict()
        print(f"y_pred {len(y_pred_dict)} classes",
              {str(k): v for k, v in itertools.islice(y_pred_dict.items(), n_top_class)})
        print(f"y_true {len(y_true_dict)} classes",
              {str(k): v for k, v in itertools.islice(y_true_dict.items(), n_top_class)})
    else:
        y_pred_dict = pd.Series(y_pred.argmax(1).detach().cpu().type(torch.int).numpy()).value_counts().to_dict()
        y_true_dict = pd.Series(y_true.detach().cpu().type(torch.int).numpy()).value_counts().to_dict()
        print(f"y_pred {len(y_pred_dict)} classes",
              {str(k): v for k, v in itertools.islice(y_pred_dict.items(), n_top_class)})
        print(f"y_true {len(y_true_dict)} classes",
              {str(k): v for k, v in itertools.islice(y_true_dict.items(), n_top_class)})
