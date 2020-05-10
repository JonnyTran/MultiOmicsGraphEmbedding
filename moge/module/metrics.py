import torch

from ignite.metrics.metric import Metric
from ignite.exceptions import NotComputableError
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced

from ignite.metrics import Precision, Recall


class Metrics():
    def __init__(self):
        self.precision = Precision(average=True, is_multilabel=True)
        self.recall = Recall(average=True, is_multilabel=True)
        self.top_k_train = TopKMulticlassAccuracy(k=107)
        self.precision_val = Precision(average=True, is_multilabel=True)
        self.recall_val = Recall(average=True, is_multilabel=True)
        self.top_k_val = TopKMulticlassAccuracy(k=107)

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor, training: bool):
        if training:
            self.precision.update(((y_pred > 0.5).type_as(y_true), y_true))
            self.recall.update(((y_pred > 0.5).type_as(y_true), y_true))
            self.top_k_train.update((y_pred, y_true))
        else:
            self.precision_val.update(((y_pred > 0.5).type_as(y_true), y_true))
            self.recall_val.update(((y_pred > 0.5).type_as(y_true), y_true))
            self.top_k_val.update((y_pred, y_true))

    def compute(self, training: bool):
        if training:
            logs = {
                "precision": self.precision.compute(),
                "recall": self.recall.compute(),
                "top_k": self.top_k_train.compute()}
        else:
            logs = {
                "val_precision": self.precision_val.compute(),
                "val_recall": self.recall_val.compute(),
                "val_top_k": self.top_k_val.compute()}
        return logs

    def reset(self, training: bool):
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

    def __init__(self, k=5, output_transform=lambda x: x, device=None):
        super(TopKMulticlassAccuracy, self).__init__(output_transform, device=device)
        self._k = k

    @reinit__is_reduced
    def reset(self):
        self._num_correct = 0
        self._num_examples = 0

    @reinit__is_reduced
    def update(self, outputs):
        y_pred, y_true = outputs
        batch_size, n_classes = y_true.size()
        _, top_indices = y_pred.topk(self._k, 1, True, True)

        y_true_select = torch.gather(y_true, 1, top_indices)
        corrects_in_k = y_true_select.sum(1) * 1.0 / self._k
        corrects_in_k = corrects_in_k.sum(0)  # sum across all samples (to average at .compute())
        self._num_correct += corrects_in_k.item()
        self._num_examples += batch_size

    @sync_all_reduce("_num_correct", "_num_examples")
    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError("TopKCategoricalAccuracy must have at"
                                     "least one example before it can be computed.")
        return self._num_correct / self._num_examples

