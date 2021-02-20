from typing import Callable, Optional, Union
import pandas as pd
import numpy as np

import torch
from ignite.exceptions import NotComputableError
from ignite.metrics import Precision, Recall, Accuracy, TopKCategoricalAccuracy, MetricsLambda, Fbeta
from ignite.metrics.metric import Metric
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced
from ogb.graphproppred import Evaluator as GraphEvaluator
from ogb.nodeproppred import Evaluator as NodeEvaluator
from ogb.linkproppred import Evaluator as LinkEvaluator

from .utils import filter_samples


class Metrics():
    def __init__(self, prefix, loss_type: str, threshold=0.5, top_k=[1, 5, 10], n_classes: int = None,
                 multilabel: bool = None,
                 metrics=["precision", "recall", "top_k", "accuracy"]):
        self.loss_type = loss_type.upper()
        self.threshold = threshold
        self.n_classes = n_classes
        self.multilabel = multilabel
        self.top_ks = top_k
        self.prefix = prefix
        add_f1_metric = False

        if n_classes:
            top_k = [k for k in top_k if k < n_classes]

        self.metrics = {}
        for metric in metrics:
            if "precision" == metric:
                self.metrics[metric] = Precision(average=False, is_multilabel=multilabel, output_transform=None)
                if "micro_f1" in metrics:
                    self.metrics["precision_avg"] = Precision(average=True, is_multilabel=multilabel,
                                                              output_transform=None)
            elif "recall" == metric:
                self.metrics[metric] = Recall(average=False, is_multilabel=multilabel, output_transform=None)
                if "micro_f1" in metrics:
                    self.metrics["recall_avg"] = Recall(average=True, is_multilabel=multilabel, output_transform=None)
            elif "top_k" in metric:
                if multilabel:
                    self.metrics[metric] = TopKMultilabelAccuracy(k_s=top_k)
                else:
                    self.metrics[metric] = TopKCategoricalAccuracy(k=max(int(np.log(n_classes)), 1),
                                                                   output_transform=None)
            elif "f1" in metric:
                add_f1_metric = True
                continue
            elif "accuracy" in metric:
                self.metrics[metric] = Accuracy(is_multilabel=multilabel, output_transform=None)
            elif "ogbn" in metric:
                self.metrics[metric] = OGBNodeClfMetrics(NodeEvaluator(metric))
            elif "ogbg" in metric:
                self.metrics[metric] = OGBNodeClfMetrics(GraphEvaluator(metric))
            elif "ogbl" in metric:
                self.metrics[metric] = OGBLinkPredMetrics(LinkEvaluator(metric))
            else:
                print(f"WARNING: metric {metric} doesn't exist")

        if add_f1_metric:
            assert "precision" in self.metrics and "recall" in self.metrics

            def macro_f1(precision, recall):
                return (precision * recall * 2 / (precision + recall + 1e-12)).mean()

            self.metrics["macro_f1"] = MetricsLambda(macro_f1, self.metrics["precision"], self.metrics["recall"])

            if "micro_f1" in metrics:
                def micro_f1(precision, recall):
                    return (precision * recall * 2 / (precision + recall + 1e-12))

                self.metrics["micro_f1"] = MetricsLambda(micro_f1, self.metrics["precision_avg"],
                                                         self.metrics["recall_avg"])

        self.reset_metrics()

    def update_metrics(self, y_hat: torch.Tensor, y: torch.Tensor, weights):
        """
        :param y_pred:
        :param y_true:
        :param weights:
        """
        y_pred = y_hat.detach()
        y_true = y.detach()
        y_pred, y_true = filter_samples(y_pred, y_true, weights)

        # Apply softmax/sigmoid activation if needed
        if "LOGITS" in self.loss_type or "FOCAL" in self.loss_type:
            if "SOFTMAX" in self.loss_type:
                y_pred = torch.softmax(y_pred, dim=1)
            else:
                torch.sigmoid(y_pred)
        elif "NEGATIVE_LOG_LIKELIHOOD" == self.loss_type or "SOFTMAX_CROSS_ENTROPY" in self.loss_type:
            y_pred = torch.softmax(y_pred, dim=1)

        for metric in self.metrics:
            if "precision" in metric or "recall" in metric or "f1" in metric or "accuracy" in metric:
                if not self.multilabel and y_true.dim() == 1:
                    self.metrics[metric].update((self.hot_encode(y_pred.argmax(1, keepdim=False), type_as=y_true),
                                                 self.hot_encode(y_true, type_as=y_pred)))
                    # self.metrics[metric].update(
                    # ((y_pred > self.threshold).type_as(y_true),
                    #  self.hot_encode(y_true, y_pred)))
                else:
                    self.metrics[metric].update(((y_pred > self.threshold).type_as(y_true), y_true))

            elif metric == "top_k":
                self.metrics[metric].update((y_pred, y_true))

            elif "ogb" in metric:
                if metric in ["ogbl-ddi", "ogbl-collab"]:
                    y_true = y_true[:, 0]

                self.metrics[metric].update((y_pred, y_true))
            else:
                raise Exception(f"Metric {metric} has problem at .update()")

    def hot_encode(self, labels, type_as):
        if labels.dim() == 2:
            return labels
        elif labels.dim() == 1:
            labels = torch.eye(self.n_classes)[labels].type_as(type_as)
            return labels

    def compute_metrics(self):
        logs = {}
        for metric in self.metrics:
            try:
                if "ogb" in metric or (metric == "top_k" and isinstance(self.metrics[metric], TopKMultilabelAccuracy)):
                    logs.update(self.metrics[metric].compute(prefix=self.prefix))
                elif metric == "top_k" and isinstance(self.metrics[metric], TopKCategoricalAccuracy):
                    metric_name = (
                                      metric if self.prefix is None else self.prefix + metric) + f"@{self.metrics[metric]._k}"
                    logs[metric_name] = self.metrics[metric].compute()
                else:
                    metric_name = metric if self.prefix is None else self.prefix + metric
                    logs[metric_name] = self.metrics[metric].compute()
            except Exception as e:
                print(f"Had problem with metric {metric}, {str(e)}\r")

        # Needed for Precision(average=False) metrics
        logs = {k: v.mean() if isinstance(v, torch.Tensor) and v.numel() > 1 else v for k, v in logs.items()}

        return logs

    def reset_metrics(self):
        for metric in self.metrics:
            self.metrics[metric].reset()


class OGBNodeClfMetrics(Metric):
    def __init__(self, evaluator: NodeEvaluator, output_transform=None, device=None):
        super().__init__(output_transform, device)
        self.evaluator = evaluator
        self.y_pred = []
        self.y_true = []

    def reset(self):
        self.y_pred = []
        self.y_true = []

    def update(self, outputs):
        y_pred, y_true = outputs
        if isinstance(self.evaluator, NodeEvaluator):
            assert y_pred.dim() == 2
            y_pred = y_pred.argmax(axis=1)

        if y_pred.dim() <= 1:
            y_pred = y_pred.unsqueeze(-1)
        if y_true.dim() <= 1:
            y_true = y_true.unsqueeze(-1)

        self.y_true.append(y_true)
        self.y_pred.append(y_pred)

    def compute(self, prefix=None):
        if isinstance(self.evaluator, NodeEvaluator):
            output = self.evaluator.eval({"y_pred": torch.cat(self.y_pred, dim=0),
                                          "y_true": torch.cat(self.y_true, dim=0)})
        elif isinstance(self.evaluator, LinkEvaluator):
            y_pred_pos = torch.cat(self.y_pred, dim=0).squeeze(-1)
            y_pred_neg = torch.cat(self.y_true, dim=0)

            output = self.evaluator.eval({"y_pred_pos": y_pred_pos,
                                          "y_pred_neg": y_pred_neg})
            output = {k.strip("_list"): v.mean().item() for k, v in output.items()}
        else:
            raise Exception(f"implement eval for {self.evaluator}")

        if prefix is None:
            return {f"{k}": v for k, v in output.items()}
        else:
            return {f"{prefix}{k}": v for k, v in output.items()}


class OGBLinkPredMetrics(Metric):
    def __init__(self, evaluator: LinkEvaluator, output_transform=None, device=None):
        super(OGBLinkPredMetrics, self).__init__(output_transform)
        self.evaluator = evaluator
        self.outputs = {}

    def reset(self):
        self.outputs = {}

    def update(self, outputs):
        e_pred_pos, e_pred_neg = outputs

        if e_pred_pos.dim() > 1:
            e_pred_pos = e_pred_pos.squeeze(-1)

        # if e_pred_neg.dim() <= 1:
        #     e_pred_neg = e_pred_neg.unsqueeze(-1)

        # print("e_pred_pos", e_pred_pos.shape)
        # print("e_pred_neg", e_pred_neg.shape)
        output = self.evaluator.eval({"y_pred_pos": e_pred_pos,
                                      "y_pred_neg": e_pred_neg})
        for k, v in output.items():
            if not isinstance(v, float):
                self.outputs.setdefault(k.strip("_list"), []).append(v.mean())
            else:
                self.outputs.setdefault(k.strip("_list"), []).append(v)

    def compute(self, prefix=None):
        output = {k: torch.stack(v, dim=0).mean().item() for k, v in self.outputs.items()}

        if prefix is None:
            return {f"{k}": v for k, v in output.items()}
        else:
            return {f"{prefix}{k}": v for k, v in output.items()}


class TopKMultilabelAccuracy(Metric):
    """
    Calculates the top-k categorical accuracy.

    - `update` must receive output of the form `(y_pred, y)` or `{'y_pred': y_pred, 'y': y}` Tensors of size (batch_size, n_classes).
    """

    def __init__(self, k_s=[5, 10, 50, 100, 200], output_transform=None, device=None):
        self.k_s = k_s
        super(TopKMultilabelAccuracy, self).__init__(output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self._num_correct = {k: 0 for k in self.k_s}
        self._num_examples = 0

    @reinit__is_reduced
    def update(self, outputs):
        y_pred, y_true = outputs
        batch_size, n_classes = y_true.size()
        _, top_indices = y_pred.topk(k=max(self.k_s), dim=1, largest=True, sorted=True)

        for k in self.k_s:
            y_true_select = torch.gather(y_true, 1, top_indices[:, :k])
            corrects_in_k = y_true_select.sum(1) * 1.0 / k
            corrects_in_k = corrects_in_k.sum(0)  # sum across all samples to get # of true positives
            self._num_correct[k] += corrects_in_k.item()
        self._num_examples += batch_size

    @sync_all_reduce("_num_correct", "_num_examples")
    def compute(self, prefix=None) -> dict:
        if self._num_examples == 0:
            raise NotComputableError("TopKCategoricalAccuracy must have at"
                                     "least one example before it can be computed.")
        if prefix is None:
            return {f"top_k@{k}": self._num_correct[k] / self._num_examples for k in self.k_s}
        else:
            return {f"{prefix}top_k@{k}": self._num_correct[k] / self._num_examples for k in self.k_s}
