import torch

from ignite.metrics.metric import Metric
from ignite.exceptions import NotComputableError
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced


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

