import torch
from ignite.exceptions import NotComputableError
from ignite.metrics import Precision, Recall, Accuracy
from ignite.metrics.metric import Metric
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced
from ogb.graphproppred import Evaluator as GraphEvaluator
from ogb.nodeproppred import Evaluator as NodeEvaluator
from ogb.linkproppred import Evaluator as LinkEvaluator

from .utils import filter_samples


class Metrics():
    def __init__(self, loss_type, threshold=0.5, k_s=[1, 5, 10], n_classes=None, multilabel=None,
                 metrics=["precision", "recall", "top_k", "accuracy"], prefix=None):
        self.loss_type = loss_type
        self.threshold = threshold
        self.n_classes = n_classes
        is_multilabel = False if "SOFTMAX" in loss_type else True
        if multilabel is not None:
            is_multilabel = multilabel
        self.is_multilabel = is_multilabel

        if n_classes:
            k_s = [k for k in k_s if k < n_classes]

        self.prefix = prefix
        self.metrics = {}
        for metric in metrics:
            if "precision" in metric:
                self.metrics[metric] = Precision(average=True, is_multilabel=is_multilabel)
            elif "recall" in metric:
                self.metrics[metric] = Recall(average=True, is_multilabel=is_multilabel)
            elif "top_k" in metric:
                self.metrics[metric] = TopKMulticlassAccuracy(k_s=k_s)
            elif "accuracy" in metric:
                self.metrics[metric] = Accuracy(is_multilabel=is_multilabel)
            elif "ogbn" in metric:
                self.metrics[metric] = NodeEvaluator(metric)
            elif "ogbg" in metric:
                self.metrics[metric] = GraphEvaluator(metric)
            elif "ogbl" in metric:
                self.metrics[metric] = LinkEvaluator(metric)
            else:
                print(f"WARNING: metric {metric} doesn't exist")

    def update_metrics(self, Y_hat: torch.Tensor, Y: torch.Tensor, weights):
        Y_hat, Y = filter_samples(Y_hat, Y, weights)

        if "LOGITS" in self.loss_type or "FOCAL" in self.loss_type:
            Y_hat = torch.softmax(Y_hat, dim=-1) if "SOFTMAX" in self.loss_type else torch.sigmoid(Y_hat)

        if not self.is_multilabel or "SOFTMAX" in self.loss_type:
            if Y.dim() >= 2:
                Y = Y.squeeze(1)
            Y = torch.eye(self.n_classes)[Y].type_as(Y_hat)

        for metric in self.metrics:
            if "precision" in metric or "recall" in metric:
                self.metrics[metric].update(((Y_hat > self.threshold).type_as(Y), Y))
            elif "accuracy" in metric:
                self.metrics[metric].update(((Y_hat > self.threshold).type_as(Y), Y))
            elif metric == "top_k":
                self.metrics[metric].update((Y_hat, Y))

    def evaluate_metric(self, Y_hat: torch.Tensor, Y: torch.Tensor, metric):
        if "ogbn" in metric:
            Y_hat = Y_hat.argmax(axis=1)
            if Y_hat.dim() <= 1:
                Y_hat = Y_hat.unsqueeze(-1)
            if Y.dim() <= 1:
                Y = Y.unsqueeze(-1)
            return self.metrics[metric].eval({"y_pred": Y_hat, "y_true": Y})
        else:
            return {}

    def compute_metrics(self):
        logs = {}
        for metric in self.metrics:
            if metric == "top_k":
                logs.update(self.metrics[metric].compute(prefix=self.prefix))
            elif "ogb" in metric:
                continue
            else:
                metric_name = metric if self.prefix is None else self.prefix + metric
                logs[metric_name] = self.metrics[metric].compute()

        return logs

    def reset_metrics(self):
        for metric in self.metrics:
            if "ogb" in metric:
                continue
            self.metrics[metric].reset()


class TopKMulticlassAccuracy(Metric):
    """
    Calculates the top-k categorical accuracy.

    - `update` must receive output of the form `(y_pred, y)` or `{'y_pred': y_pred, 'y': y}` Tensors of size (batch_size, n_classes).
    """

    def __init__(self, k_s=[5, 10, 50, 100, 200], output_transform=lambda x: x, device=None):
        self.k_s = k_s
        super(TopKMulticlassAccuracy, self).__init__(output_transform, device=device)

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
