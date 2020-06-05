import copy
from collections import OrderedDict
import torch
from ignite.exceptions import NotComputableError
from ignite.metrics import Precision, Recall
from ignite.metrics.metric import Metric
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced
from .utils import filter_samples

class Metrics():
    def __init__(self, loss_type):
        self.loss_type = loss_type
        self.precision = Precision(average=True, is_multilabel=True)
        self.recall = Recall(average=True, is_multilabel=True)
        self.top_k_train = TopKMulticlassAccuracy(k_s=[1, 5, 10])
        self.precision_val = Precision(average=True, is_multilabel=True)
        self.recall_val = Recall(average=True, is_multilabel=True)
        self.top_k_val = TopKMulticlassAccuracy(k_s=[1, 5, 10])

    def update_metrics(self, y_pred: torch.Tensor, y_true: torch.Tensor, weights, training: bool):
        y_true, y_pred = filter_samples(y_true, y_pred, weights)
        if "LOGITS" in self.loss_type or "FOCAL" in self.loss_type:
            y_pred = torch.softmax(y_pred, dim=-1) if "SOFTMAX" in self.loss_type else torch.sigmoid(y_pred)

        if training:
            self.precision.update(((y_pred > 0.5).type_as(y_true), y_true))
            self.recall.update(((y_pred > 0.5).type_as(y_true), y_true))
            self.top_k_train.update((y_pred, y_true))
        else:
            self.precision_val.update(((y_pred > 0.5).type_as(y_true), y_true))
            self.recall_val.update(((y_pred > 0.5).type_as(y_true), y_true))
            self.top_k_val.update((y_pred, y_true))

    def compute_metrics(self, training: bool):
        if training:
            logs = {
                "precision": self.precision.compute(),
                "recall": self.recall.compute(),
                **self.top_k_train.compute(training)}
        else:
            logs = {
                "val_precision": self.precision_val.compute(),
                "val_recall": self.recall_val.compute(),
                **self.top_k_val.compute(training)}
        return logs

    def reset_metrics(self, training: bool):
        if training:
            self.precision.reset()
            self.recall.reset()
            self.top_k_train.reset()
        else:
            self.precision_val.reset()
            self.recall_val.reset()
            self.top_k_val.reset()


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
    def compute(self, training=True) -> dict:
        if self._num_examples == 0:
            raise NotComputableError("TopKCategoricalAccuracy must have at"
                                     "least one example before it can be computed.")
        if training:
            return {f"top_k@{k}": self._num_correct[k] / self._num_examples for k in self.k_s}
        else:
            return {f"val_top_k@{k}": self._num_correct[k] / self._num_examples for k in self.k_s}
