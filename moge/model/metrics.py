from typing import Optional, Any, Callable, List, Dict, Union, Tuple

import numpy as np
import torch
import torchmetrics
from ignite.exceptions import NotComputableError
from ignite.metrics import Precision, Recall, TopKCategoricalAccuracy
from logzero import logger
from ogb.graphproppred import Evaluator as GraphEvaluator
from ogb.linkproppred import Evaluator as LinkEvaluator
from ogb.nodeproppred import Evaluator as NodeEvaluator
from sklearn.metrics import average_precision_score
from torch import Tensor
from torchmetrics import Metric, F1Score, AUROC, MeanSquaredError, Accuracy, AveragePrecision
from torchmetrics.utilities.data import METRIC_EPS, to_onehot

from .utils import filter_samples, tensor_sizes, activation


class Metrics(torch.nn.Module):
    def __init__(self, prefix: str, loss_type: str, threshold: float = 0.5,
                 top_k: List[int] = [5, 10, 50], n_classes: int = None,
                 multilabel: bool = None,
                 metrics: List[Union[str, Tuple[str]]] = ["precision", "recall", "top_k", "accuracy"]):
        super().__init__()

        self.loss_type = loss_type.upper()
        self.threshold = threshold
        self.n_classes = n_classes
        self.multilabel = multilabel
        self.top_ks = top_k
        self.prefix = prefix if isinstance(prefix, str) else ""

        self.metrics: Dict[Union[str, Tuple[str]], Metric] = {}
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
                self.metrics[metric] = F1Score(num_classes=n_classes, average="macro",
                                               top_k=int(metric.split("@")[-1]) if "@" in metric else None,
                                               # MacroF1@TopK
                                               )
            elif "micro_f1" in metric:
                self.metrics[metric] = F1Score(num_classes=n_classes, average="micro",
                                               top_k=int(metric.split("@")[-1]) if "@" in metric else None, )
            elif "fmax" in metric:
                self.metrics[metric] = FMax(average="macro")

            elif "mse" == metric:
                self.metrics[metric] = MeanSquaredError()
            elif "auroc" == metric:
                self.metrics[metric] = AUROC(num_classes=n_classes, average="micro")
            elif "aupr" == metric:
                self.metrics[metric] = AveragePrecision(average="micro")

            elif "accuracy" in metric:
                self.metrics[metric] = Accuracy(top_k=int(metric.split("@")[-1]) if "@" in metric else None,
                                                subset_accuracy=multilabel)

            elif "ogbn" in metric or any("ogbn" in s for s in metric):
                self.metrics[metric] = OGBNodeClfMetrics(
                    NodeEvaluator(metric[0] if isinstance(metric, (list, tuple)) else metric))
            elif "ogbl" in metric or any("ogbl" in s for s in metric):
                self.metrics[metric] = OGBLinkPredMetrics(
                    LinkEvaluator(metric[0] if isinstance(metric, (list, tuple)) else metric))
            elif "ogbg" in metric or any("ogbg" in s for s in metric):
                self.metrics[metric] = OGBNodeClfMetrics(
                    GraphEvaluator(metric[0] if isinstance(metric, (list, tuple)) else metric))
            else:
                print(f"WARNING: metric {metric} doesn't exist")
                continue

            # Needed to add the PytorchGeometric methods as Modules, so they'll be on the correct CUDA device during training
            if isinstance(self.metrics[metric], torchmetrics.metric.Metric):
                setattr(self, str(metric), self.metrics[metric])

        self.reset_metrics()

    def hot_encode(self, labels, type_as):
        if labels.dim() == 2:
            return labels
        elif labels.dim() == 1:
            labels = torch.eye(self.n_classes)[labels].type_as(type_as)
            return labels

    def update_metrics(self, y_pred: Tensor, y_true: Tensor,
                       weights=Optional[Tensor], subset: List[str] = None):
        """
        Args:
            y_pred:
            y_true:
            weights:
        """
        y_pred = y_pred.detach()
        y_true = y_true.detach()

        y_pred, y_true = filter_samples(y_pred, y_true, weights=weights)
        y_pred_act = activation(y_pred, loss_type=self.loss_type)

        if subset is None:
            metrics = self.metrics.keys()
        else:
            metrics = [name for name in subset if name in self.metrics]

        for name in metrics:
            # Torch ignite metrics
            if "precision" in name or "recall" in name or "accuracy" in name:
                if not self.multilabel and y_true.dim() == 1:
                    self.metrics[name].update((self.hot_encode(y_pred_act.argmax(1, keepdim=False), type_as=y_true),
                                               self.hot_encode(y_true, type_as=y_pred)))
                elif name in self.metrics:
                    self.metrics[name].update(((y_pred_act > self.threshold).type_as(y_true), y_true))

            # Torch ignite metrics
            elif name == "top_k":
                self.metrics[name].update(y_pred_act, y_true)
            elif name == "avg_precision":
                self.metrics[name].update(y_pred_act, y_true)

            # OGB metrics
            elif "ogbn" in name:
                if name in ["ogbl-ddi", "ogbl-collab"]:
                    y_true = y_true[:, 0]
                elif "ogbg-mol" in name:
                    # print(tensor_sizes({"y_pred": y_pred, "y_true": y_true}))
                    pass

                self.metrics[name].update((y_pred_act, y_true))

            elif "ogbl" in name:
                # Both y_pred, y_true must have activation func applied, not with `y_pred_act`
                edge_pos = y_pred
                edge_neg = y_true
                self.metrics[name].update(edge_pos, edge_neg)

            # torchmetrics metrics
            elif isinstance(self.metrics[name], torchmetrics.metric.Metric):
                try:
                    self.metrics[name].update(y_pred_act, y_true)
                except Exception as e:
                    print(e, "\n", name, tensor_sizes({"y_pred": y_pred_act, "y_true": y_true}))
                    raise e
                    # self.metrics[metric].update(y_pred_full, y_true_full)


    def compute_metrics(self) -> Dict[str, Tensor]:
        logs = {}
        for metric in self.metrics:
            try:
                if isinstance(metric, (list, tuple)):
                    prefix = self.prefix + "_".join(metric[1:])
                else:
                    prefix = self.prefix

                if "ogb" in metric:
                    logs.update(self.metrics[metric].compute(prefix=prefix))

                elif metric == "top_k" and isinstance(self.metrics[metric], TopKMultilabelAccuracy):
                    logs.update(self.metrics[metric].compute(prefix=prefix))

                elif metric == "top_k" and isinstance(self.metrics[metric], TopKCategoricalAccuracy):
                    metric_name = (str(metric) if prefix is None else prefix + str(metric)) + \
                                  f"@{self.metrics[metric]._k}"
                    logs[metric_name] = self.metrics[metric].compute()

                else:
                    metric_name = str(metric) if prefix is None else prefix + str(metric)
                    logs[metric_name] = self.metrics[metric].compute()

            except Exception as e:
                print(f"Had problem with metric {metric}, {str(e)}\r")
                # traceback.print_exc()

        # Needed for Precision(average=False) metrics
        logs = {k: v.mean() if isinstance(v, Tensor) and v.numel() > 1 else v for k, v in logs.items()}

        return logs

    def reset_metrics(self):
        for metric in self.metrics:
            self.metrics[metric].reset()


class OGBNodeClfMetrics(torchmetrics.Metric):
    def __init__(self, evaluator, compute_on_step: bool = True, dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None, dist_sync_fn: Callable = None):
        super().__init__()
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
    def __init__(self, evaluator: LinkEvaluator, compute_on_step: bool = True):
        super().__init__()
        self.evaluator = evaluator
        self.outputs = {}

    def reset(self):
        self.outputs = {}

    def update(self, e_pred_pos: Tensor, e_pred_neg: Tensor):
        if e_pred_pos.dim() > 1:
            e_pred_pos = e_pred_pos.squeeze(-1)

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

    def __init__(self, k_s=[5, 10, 50, 100, 200], compute_on_step: bool = True, **kwargs):
        super().__init__()
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


class FMax(Metric):
    def __init__(self, num_classes=None, thresholds: Union[int, Tensor, List[float]] = 100,
                 average="macro", **kwargs) -> None:
        assert average == "macro"
        super().__init__(**kwargs)

        self.num_classes = num_classes
        if isinstance(thresholds, int):
            self.num_thresholds = thresholds
            self.thresholds = torch.linspace(0, 1.0, thresholds)

        elif thresholds is not None:
            if not isinstance(thresholds, (list, Tensor)):
                raise ValueError("Expected argument `thresholds` to either be an integer, list of floats or a tensor")
            self.thresholds = torch.tensor(thresholds) if isinstance(thresholds, list) else thresholds
            self.num_thresholds = self.thresholds.numel()

        if self.num_classes:
            for name in ("TPs", "FPs", "FNs"):
                self.add_state(
                    name=name,
                    default=torch.zeros(self.num_classes, self.num_thresholds, dtype=torch.float32),
                    dist_reduce_fx="sum",
                )

    def update(self, preds: Tensor, target: Tensor) -> None:
        if self.num_classes is None:
            self.num_classes = target.size(1) if target.dim() > 1 else 1
            for name in ("TPs", "FPs", "FNs"):
                self.add_state(
                    name=name,
                    default=torch.zeros(self.num_classes, self.num_thresholds, dtype=torch.float32,
                                        device=preds.device),
                    dist_reduce_fx="sum",
                )

        if len(preds.shape) == len(target.shape) == 1:
            preds = preds.reshape(-1, 1)
            target = target.reshape(-1, 1)

        if len(preds.shape) == len(target.shape) + 1:
            target = to_onehot(target, num_classes=self.num_classes)

        target = target == 1
        # Iterate one threshold at a time to conserve memory
        for i in range(self.num_thresholds):
            predictions = preds >= self.thresholds[i]
            self.TPs[:, i] += (target & predictions).sum(dim=0)
            self.FPs[:, i] += ((~target) & predictions).sum(dim=0)
            self.FNs[:, i] += (target & (~predictions)).sum(dim=0)

    def compute(self) -> Tensor:
        """Returns float tensor of size n_classes."""
        if not hasattr(self, "TPs"):
            raise NotComputableError("FMax must have at"
                                     "least one example before it can be computed.")

        precisions = (self.TPs + METRIC_EPS) / (self.TPs + self.FPs + METRIC_EPS)
        recalls = self.TPs / (self.TPs + self.FNs + METRIC_EPS)

        # Need to guarantee that last precision=1 and recall=0, similar to precision_recall_curve
        t_ones = torch.ones(self.num_classes, 1, dtype=precisions.dtype, device=precisions.device)
        precisions = torch.cat([precisions, t_ones], dim=1).cpu()
        t_zeros = torch.zeros(self.num_classes, 1, dtype=recalls.dtype, device=recalls.device)
        recalls = torch.cat([recalls, t_zeros], dim=1).cpu()

        numerator = 2 * recalls * precisions
        denom = recalls + precisions

        f1_scores = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom != 0))
        max_f1s = np.max(f1_scores, axis=1)

        # thresholds = torch.stack([self.thresholds for _ in range(self.num_classes)], dim=0)
        # max_f1_thresh = thresholds[np.argmax(f1_scores, axis=1)]

        return max_f1s.mean()


class AveragePrecision(torchmetrics.Metric):
    def __init__(self, average="macro", ):
        """

        Args:
            average : {'micro', 'samples', 'weighted', 'macro'} or None,             default='macro'
                If ``None``, the scores for each class are returned. Otherwise,
                this determines the type of averaging performed on the data:

                ``'micro'``:
                    Calculate metrics globally by considering each element of the label
                    indicator matrix as a label.
                ``'macro'``:
                    Calculate metrics for each label, and find their unweighted
                    mean.  This does not take label imbalance into account.
                ``'weighted'``:
                    Calculate metrics for each label, and find their average, weighted
                    by support (the number of true instances for each label).
                ``'samples'``:
                    Calculate metrics for each instance, and find their average.
        """
        super().__init__()
        self.average = average

    def reset(self):
        self._scores = []
        self._n_samples = []

    def update(self, y_pred, y_true):
        score = average_precision_score(y_true.detach().cpu().numpy(),
                                        y_pred.detach().cpu().numpy(), average=self.average)

        self._scores.append(score)
        self._n_samples.append(y_true.size(0))

    def compute(self, prefix=None) -> Union[float, Dict[str, float]]:
        if len(self._scores) == 0:
            logger.warn("AveragePrecision must have at"
                        "least one example before it can be computed.")

        weighted_avg_score = np.average(self._scores, weights=self._n_samples)
        return weighted_avg_score if prefix is None else {f"{prefix}avg_precision": weighted_avg_score}
