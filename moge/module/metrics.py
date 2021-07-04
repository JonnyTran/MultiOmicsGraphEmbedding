from typing import Optional, Any, Callable

import numpy as np
import torch
from ogb.graphproppred import Evaluator as GraphEvaluator
from ogb.linkproppred import Evaluator as LinkEvaluator
from ogb.nodeproppred import Evaluator as NodeEvaluator

from ignite.exceptions import NotComputableError
from ignite.metrics import Precision, Recall, TopKCategoricalAccuracy

import torchmetrics
from torchmetrics import F1, AUROC, AveragePrecision, MeanSquaredError, Accuracy

from .utils import filter_samples


class Metrics(torch.nn.Module):
    def __init__(self, prefix, loss_type: str, threshold=0.5, top_k=[1, 5, 10], n_classes: int = None,
                 multilabel: bool = None, metrics=["precision", "recall", "top_k", "accuracy"]):
        super().__init__()

        self.loss_type = loss_type.upper()
        self.threshold = threshold
        self.n_classes = n_classes
        self.multilabel = multilabel
        self.top_ks = top_k
        self.prefix = prefix


        self.metrics = {}
        for metric in metrics:
            if "precision" == metric:
                self.metrics[metric] = Precision(average=True, is_multilabel=multilabel)
            elif "recall" == metric:
                self.metrics[metric] = Recall(average=True, is_multilabel=multilabel)

            elif "top_k" in metric:
                if n_classes:
                    top_k = [k for k in top_k if k < n_classes]

                if multilabel:
                    self.metrics[metric] = TopKMultilabelAccuracy(k_s=top_k)
                else:
                    self.metrics[metric] = TopKCategoricalAccuracy(k=max(int(np.log(n_classes)), 1),
                                                                   output_transform=None)
            elif "macro_f1" in metric:
                self.metrics[metric] = F1(num_classes=n_classes, average="macro")
            elif "micro_f1" in metric:
                self.metrics[metric] = F1(num_classes=n_classes, average="micro")
            elif "mse" == metric:
                self.metrics[metric] = MeanSquaredError()
            elif "auroc" == metric:
                self.metrics[metric] = AUROC(num_classes=n_classes)
            elif "avg_precision" in metric:
                self.metrics[metric] = AveragePrecision(num_classes=n_classes, )


            elif "accuracy" in metric:
                self.metrics[metric] = Accuracy(top_k=int(metric.split("@")[-1]) if "@" in metric else None)

            elif "ogbn" in metric:
                self.metrics[metric] = OGBNodeClfMetrics(NodeEvaluator(metric))
            elif "ogbg" in metric:
                self.metrics[metric] = OGBNodeClfMetrics(GraphEvaluator(metric))
            elif "ogbl" in metric:
                self.metrics[metric] = OGBLinkPredMetrics(LinkEvaluator(metric))
            else:
                print(f"WARNING: metric {metric} doesn't exist")

            # Needed to add the PytorchGeometric methods as Modules, so they'll be on the correct CUDA device during training
            if isinstance(self.metrics[metric], torchmetrics.metric.Metric):
                setattr(self, metric, self.metrics[metric])

        self.reset_metrics()

    def update_metrics(self, y_hat: torch.Tensor, y: torch.Tensor, weights=None):
        """
        :param y_pred:
        :param y_true:
        :param weights:
        """
        y_pred = y_hat.detach()
        y_true = y.detach()
        y_pred, y_true = filter_samples(y_pred, y_true, weights=weights, max_mode=True)

        # Apply softmax/sigmoid activation if needed
        if "LOGITS" in self.loss_type or "FOCAL" in self.loss_type:
            if "SOFTMAX" in self.loss_type:
                y_pred = torch.softmax(y_pred, dim=1)
            else:
                y_pred = torch.sigmoid(y_pred)
        elif "NEGATIVE_LOG_LIKELIHOOD" == self.loss_type or "SOFTMAX_CROSS_ENTROPY" in self.loss_type:
            y_pred = torch.softmax(y_pred, dim=1)

        for metric in self.metrics:
            # torchmetrics metrics
            if isinstance(self.metrics[metric], torchmetrics.metric.Metric):
                self.metrics[metric].update(y_pred, y_true)

            # Torch ignite metrics
            elif "precision" in metric or "recall" in metric or "accuracy" in metric:
                if not self.multilabel and y_true.dim() == 1:
                    self.metrics[metric].update((self.hot_encode(y_pred.argmax(1, keepdim=False), type_as=y_true),
                                                 self.hot_encode(y_true, type_as=y_pred)))
                else:
                    self.metrics[metric].update(((y_pred > self.threshold).type_as(y_true), y_true))

            # Torch ignite metrics
            elif metric == "top_k":
                self.metrics[metric].update((y_pred, y_true))

            # OGB metrics
            elif "ogb" in metric:
                if metric in ["ogbl-ddi", "ogbl-collab"]:
                    y_true = y_true[:, 0]
                elif "ogbg-mol" in metric:
                    # print(tensor_sizes({"y_pred": y_pred, "y_true": y_true}))
                    pass

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
                if "ogb" in metric:
                    logs.update(self.metrics[metric].compute(prefix=self.prefix))

                elif metric == "top_k" and isinstance(self.metrics[metric], TopKMultilabelAccuracy):
                    logs.update(self.metrics[metric].compute(prefix=self.prefix))

                elif metric == "top_k" and isinstance(self.metrics[metric], TopKCategoricalAccuracy):
                    metric_name = (metric if self.prefix is None else \
                                       self.prefix + metric) + f"@{self.metrics[metric]._k}"
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


class OGBNodeClfMetrics(torchmetrics.Metric):
    def __init__(self, evaluator, compute_on_step: bool = True, dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None, dist_sync_fn: Callable = None):
        super().__init__(compute_on_step, dist_sync_on_step, process_group, dist_sync_fn)
        self.evaluator = evaluator
        self.y_pred = []
        self.y_true = []

    def reset(self):
        self.y_pred = []
        self.y_true = []

    def update(self, y_pred, y_true):
        if isinstance(self.evaluator, (NodeEvaluator, GraphEvaluator)):
            assert y_pred.dim() == 2
            if y_true.dim() == 1 or y_true.size(1) == 1:
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

        elif isinstance(self.evaluator, GraphEvaluator):
            input_shape = {"y_true": torch.cat(self.y_pred, dim=0),
                           "y_pred": torch.cat(self.y_true, dim=0)}
            output = self.evaluator.eval(input_shape)
        else:
            raise Exception(f"implement eval for {self.evaluator}")

        if prefix is None:
            return {f"{k}": v for k, v in output.items()}
        else:
            return {f"{prefix}{k}": v for k, v in output.items()}


class OGBLinkPredMetrics(torchmetrics.Metric):
    def __init__(self, evaluator: LinkEvaluator, compute_on_step: bool = True, dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None, dist_sync_fn: Callable = None):
        super().__init__(compute_on_step, dist_sync_on_step, process_group, dist_sync_fn)
        self.evaluator = evaluator
        self.outputs = {}

    def reset(self):
        self.outputs = {}

    def update(self, e_pred_pos, e_pred_neg):

        if e_pred_pos.dim() > 1:
            e_pred_pos = e_pred_pos.squeeze(-1)

        # if e_pred_neg.dim() <= 1:
        #     e_pred_neg = e_pred_neg.unsqueeze(-1)

        # print("e_pred_pos", e_pred_pos.shape)
        # print("e_pred_neg", e_pred_neg.shape)
        output = self.evaluator.eval({"y_pred_pos": e_pred_pos,
                                      "y_pred_neg": e_pred_neg})
        for k, v in output.items():
            if isinstance(v, float):
                score = torch.tensor([v])
                self.outputs.setdefault(k.strip("_list"), []).append(score)
            else:
                self.outputs.setdefault(k.strip("_list"), []).append(v.mean())

    def compute(self, prefix=None):
        output = {k: torch.stack(v, dim=0).mean().item() for k, v in self.outputs.items()}

        if prefix is None:
            return {f"{k}": v for k, v in output.items()}
        else:
            return {f"{prefix}{k}": v for k, v in output.items()}


class TopKMultilabelAccuracy(torchmetrics.Metric):
    """
    Calculates the top-k categorical accuracy.

    - `update` must receive output of the form `(y_pred, y)` or `{'y_pred': y_pred, 'y': y}` Tensors of size (batch_size, n_classes).
    """

    def __init__(self, k_s=[5, 10, 50, 100, 200], compute_on_step: bool = True, dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None, dist_sync_fn: Callable = None):
        super().__init__(compute_on_step, dist_sync_on_step, process_group, dist_sync_fn)
        self.k_s = k_s

    def reset(self):
        self._num_correct = {k: 0 for k in self.k_s}
        self._num_examples = 0

    def update(self, y_pred, y_true):
        batch_size, n_classes = y_true.size()
        _, top_indices = y_pred.topk(k=max(self.k_s), dim=1, largest=True, sorted=True)

        for k in self.k_s:
            y_true_select = torch.gather(y_true, 1, top_indices[:, :k])
            corrects_in_k = y_true_select.sum(1) * 1.0 / k
            corrects_in_k = corrects_in_k.sum(0)  # sum across all samples to get # of true positives
            self._num_correct[k] += corrects_in_k.item()
        self._num_examples += batch_size

    def compute(self, prefix=None) -> dict:
        if self._num_examples == 0:
            raise NotComputableError("TopKCategoricalAccuracy must have at"
                                     "least one example before it can be computed.")
        if prefix is None:
            return {f"top_k@{k}": self._num_correct[k] / self._num_examples for k in self.k_s}
        else:
            return {f"{prefix}top_k@{k}": self._num_correct[k] / self._num_examples for k in self.k_s}
