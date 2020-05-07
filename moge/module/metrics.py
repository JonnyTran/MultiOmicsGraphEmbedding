import torch

from ignite.metrics.metric import Metric
from ignite.exceptions import NotComputableError
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced


class TopKMulticlassAccuracy(Metric):
    """
    Calculates the top-k categorical accuracy.

    - `update` must receive output of the form `(y_pred, y)` or `{'y_pred': y_pred, 'y': y}`.
    """

    def __init__(self, k=5, output_transform=lambda x: x, device=None):
        super(TopKMulticlassAccuracy, self).__init__(output_transform, device=device)
        self._k = k

    @reinit__is_reduced
    def reset(self):
        self._num_correct = 0
        self._num_examples = 0

    @reinit__is_reduced
    def update(self, output: torch.Tensor, target: torch.Tensor):
        # output, target = outputs
        batch_size, n_classes = target.size()

        _, top_indices = output.topk(self._k, 1, True, True)

        targets_select = target.index_select(1, top_indices)
        print("targets_select", targets_select.shape, targets_select)

        for i in range(0, batch_size):
            corrects_sample = target[i, top_indices[i]].float().sum(0) * 1.0 / self._k
            self._num_correct += corrects_sample.item()

        self._num_examples += batch_size

    @sync_all_reduce("_num_correct", "_num_examples")
    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError("TopKCategoricalAccuracy must have at"
                                     "least one example before it can be computed.")
        return self._num_correct / self._num_examples

