import traceback
import warnings
from typing import Optional, Any, Callable, List, Dict, Union, Tuple

import numpy as np
import scipy.sparse as ssp
import torch
import torchmetrics
from ignite.exceptions import NotComputableError
from ignite.metrics import Precision, Recall, TopKCategoricalAccuracy
from logzero import logger
from ogb.graphproppred import Evaluator as GraphEvaluator
from ogb.linkproppred import Evaluator as LinkEvaluator
from ogb.nodeproppred import Evaluator as NodeEvaluator
from torch import Tensor
from torch_sparse import SparseTensor
from torchmetrics import F1Score, AUROC, MeanSquaredError, Accuracy, Metric
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.data import METRIC_EPS, to_onehot

from moge.model.tensor import filter_samples, tensor_sizes, activation


@torch.no_grad()
def add_aggregated_metrics(metrics: Dict[str, Any],
                           prefix: str = '',
                           suffixes: List[str] = ['aupr', 'fmax', 'smin'],
                           reduce='mean') -> Dict[str, Any]:
    """
    Group values from the `metrics` dict whose keys contains the same `suffixes` and reduce with mean. The new reduced
    values are added to `metrics` with `prefix` string prepended to the metric name.

    Args:
        metrics (): a dict of metric name keys and scalar values for keys
        prefix (): a string for the prefix of the new
        suffixes (): A list of metric names to aggregate.

    Returns:
        metrics (): metric dict with aggregated metrics added.
    """
    if not isinstance(suffixes, list) or not isinstance(prefix, str):
        return metrics

    for suffix in suffixes:
        name = prefix + suffix
        if name in metrics:
            continue
        groupby = [val for key, val in metrics.items() if key.endswith(suffix)]

        # Depending on the metric, the groupby list may contain float or tensor values.
        if not groupby:
            continue

        elif any(isinstance(val, Tensor) for val in groupby):
            groupby = torch.stack(groupby, dim=0)

            # Reduce function for tensors
            if reduce == 'mean':
                agg = torch.mean
            elif reduce == 'sum':
                agg = torch.sum
            elif reduce == 'max':
                agg = torch.max
            elif reduce == 'min':
                agg = torch.min
            else:
                raise ValueError(f'Invalid reduce value: {reduce}')

        elif any(isinstance(val, (np.ndarray, float, int)) for val in groupby):
            groupby = np.stack(groupby, axis=0)

            # Reduce function for numpy arrays
            if reduce == 'mean':
                agg = np.mean
            elif reduce == 'sum':
                agg = np.sum
            elif reduce == 'max':
                agg = np.max
            elif reduce == 'min':
                agg = np.min
            else:
                raise ValueError(f'Invalid reduce value: {reduce}')
        else:
            continue

        metrics[name] = agg(groupby)

    return metrics


class Metrics(torch.nn.Module):
    def __init__(self, prefix: str,
                 metrics: List[str] = ["precision", "recall", "top_k", "accuracy"],
                 loss_type: str = "BCE_WITH_LOGITS", threshold: float = 0.5, top_k: List[int] = [5, 10, 50],
                 n_classes: int = None, multilabel: bool = None):
        """
        Create a group of metrics for either training, validation, or testing with a given prefix.
        Args:
            prefix (str): A string for the prefix of the metric name.
            metrics (List[str]): A list of metric names to compute.
            loss_type (str): A string for the loss type. One of ["BCE_WITH_LOGITS", "BCE", "MSE", "CE"].
            threshold (): A float for the threshold to use for binary classification.
            top_k (): A list of integers for the top k values to use for top k accuracy.
            n_classes (): An integer for the number of classes.
            multilabel (): A boolean for whether the task is multilabel.
        """
        super().__init__()

        self.loss_type = loss_type.upper()
        self.threshold = threshold
        self.n_classes = n_classes
        self.multilabel = multilabel
        self.top_ks = top_k
        self.prefix = prefix if isinstance(prefix, str) else ""

        self.metrics = {}
        if metrics is None:
            metrics = []

        for name in metrics:
            top_k = int(name.split("@")[-1]) if "@" in name else None

            if "precision" in name:
                self.metrics[name] = Precision(average=True, is_multilabel=multilabel)
            elif "recall" in name:
                self.metrics[name] = Recall(average=True, is_multilabel=multilabel)

            elif "top_k" in name:
                if n_classes:
                    top_k = [k for k in top_k if k < n_classes]

                if multilabel:
                    self.metrics[name] = TopKMultilabelAccuracy(k_s=top_k)
                else:
                    self.metrics[name] = TopKCategoricalAccuracy(
                        k=max(int(np.log(n_classes)), 1), output_transform=None
                    )
            elif "macro_f1" in name:
                self.metrics[name] = F1Score(num_classes=n_classes, average="macro", top_k=top_k)
            elif "micro_f1" in name:
                self.metrics[name] = F1Score(num_classes=n_classes, average="micro", top_k=top_k)

            elif "fmax" in name:
                self.metrics[name] = FMax_Slow()
            elif "smin" in name:
                self.metrics[name] = Smin(num_classes=n_classes, thresholds=100)

            elif "auroc" in name:
                self.metrics[name] = AUROC(num_classes=n_classes, average="micro")
            elif "aupr" in name:
                # self.metrics[name] = AveragePrecision_(average="pairwise")
                self.metrics[name] = AveragePrecisionPairwise(average="none", shrink=1e-2 if prefix == '' else None)

            elif "mse" in name:
                self.metrics[name] = MeanSquaredError()

            elif "acc" in name:
                self.metrics[name] = Accuracy(top_k=top_k, subset_accuracy=multilabel)

            elif "ogbn" in name or any("ogbn" in s for s in name):
                self.metrics[name] = OGBNodeClfMetrics(
                    NodeEvaluator(name[0] if isinstance(name, (list, tuple)) else name)
                )
            elif "ogbl" in name or any("ogbl" in s for s in name):
                self.metrics[name] = OGBLinkPredMetrics(
                    LinkEvaluator(name[0] if isinstance(name, (list, tuple)) else name)
                )
            elif "ogbg" in name or any("ogbg" in s for s in name):
                self.metrics[name] = OGBNodeClfMetrics(
                    GraphEvaluator(name[0] if isinstance(name, (list, tuple)) else name)
                )
            else:
                logger.warn(
                    f"metric name {name} not supported. Must containing a substring in "
                    f"['precision', 'recall', 'top_k', 'macro_f1', 'micro_f1', 'fmax', 'mse', "
                    f"'auroc', 'aupr', 'acc', 'ogbn', 'ogbl', 'ogbg', ]"
                )
                continue

            # Needed to add the torchmetrics as Modules, so they'll be on the correct CUDA device during training
            if name in self.metrics and isinstance(self.metrics[name], torchmetrics.metric.Metric):
                setattr(self, str(name), self.metrics[name])

        self.reset_metrics()

    def hot_encode(self, labels, type_as):
        if labels.dim() == 2:
            return labels
        elif labels.dim() == 1:
            labels = torch.eye(self.n_classes)[labels].type_as(type_as)
            return labels

    @torch.no_grad()
    def update_metrics(
            self, y_pred: Tensor, y_true: Tensor, weights=Optional[Tensor], subset: Union[List[str], str] = None
    ):
        """
        Update the metrics with the given predictions and labels, for a subset of metrics if specified.
        Args:
            y_pred (Tensor): Predicted scores that can be logits. It'll have an activation internally corresponding to the loss function.
            y_true (Tensor): Ground truth class labels.
            weights (Tensor, optional): Sample-level weights.
            subset (str, list): Used to only update a subset of metrics. If `subset` is a string, then only update
                metrics with names containing the substring. If `subset` is a list of string, then select the metrics
                with names in the list.
        """
        y_pred = y_pred.clone()
        y_true = y_true.clone()

        y_pred, y_true = filter_samples(y_pred, y_true, weights=weights)
        y_pred_act = activation(y_pred, loss_type=self.loss_type)

        if subset is None:
            metrics = self.metrics.keys()
        elif isinstance(subset, (list, set, tuple)):
            metrics = [name for name in subset if name in self.metrics]
        elif isinstance(subset, str):
            metrics = [name for name in self.metrics if subset in name]

        for name in metrics:
            # Torch ignite metrics
            if "precision" in name or "recall" in name or "accuracy" in name:
                if not self.multilabel and y_true.dim() == 1:
                    self.metrics[name].update(
                        (
                            self.hot_encode(y_pred_act.argmax(1, keepdim=False), type_as=y_true),
                            self.hot_encode(y_true, type_as=y_pred),
                        )
                    )
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
                except NotComputableError as nce:
                    print(nce)

                except Exception as e:
                    print(e, "\n", name, tensor_sizes({"y_pred": y_pred_act, "y_true": y_true}))
                    raise e

    @torch.no_grad()
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

                elif isinstance(self.metrics[metric], TopKMultilabelAccuracy):
                    logs.update(self.metrics[metric].compute(prefix=prefix))

                elif isinstance(self.metrics[metric], TopKCategoricalAccuracy):
                    metric_name = f"{metric if prefix is None else prefix + str(metric)}@{self.metrics[metric]._k}"
                    logs[metric_name] = self.metrics[metric].compute()

                else:
                    metric_name = str(metric) if prefix is None else prefix + str(metric)
                    logs[metric_name] = self.metrics[metric].compute()

            except NotComputableError as nce:
                # logger.warn(nce)
                pass

            except ValueError as ve:
                logger.warn(ve) if "No samples to concatenate" in ve.__str__() else None
                pass

            except Exception as e:
                logger.error(f"Metric: {metric}, {type(e)}:{str(e)}\r")
                traceback.print_exc()

        # Needed for Precision(average=False) metrics
        logs = {k: v.mean() if isinstance(v, Tensor) and v.numel() > 1 else v for k, v in logs.items()}

        return logs

    def reset_metrics(self):
        for metric in self.metrics:
            self.metrics[metric].reset()


class OGBNodeClfMetrics(torchmetrics.Metric):
    def __init__(
            self,
            evaluator,
            compute_on_step: bool = True,
            dist_sync_on_step: bool = False,
            process_group: Optional[Any] = None,
            dist_sync_fn: Callable = None,
    ):
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
            output = self.evaluator.eval(
                {"y_pred": torch.cat(self.y_pred, dim=0), "y_true": torch.cat(self.y_true, dim=0)}
            )

        elif isinstance(self.evaluator, LinkEvaluator):
            y_pred_pos = torch.cat(self.y_pred, dim=0).squeeze(-1)
            y_pred_neg = torch.cat(self.y_true, dim=0)

            output = self.evaluator.eval({"y_pred_pos": y_pred_pos, "y_pred_neg": y_pred_neg})
            output = {k.strip("_list"): v.mean().item() for k, v in output.items()}

        elif isinstance(self.evaluator, GraphEvaluator):
            input_shape = {"y_true": torch.cat(self.y_pred, dim=0), "y_pred": torch.cat(self.y_true, dim=0)}
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

        output = self.evaluator.eval({"y_pred_pos": e_pred_pos, "y_pred_neg": e_pred_neg})
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
    def __init__(self, k_s=[5, 10, 50, 100, 200], compute_on_step: bool = True, **kwargs):
        """
        Calculates the top-k categorical accuracy
        Args:
            k_s ():
            compute_on_step ():
            **kwargs ():
        """
        super().__init__()
        self.k_s = k_s

    def reset(self):
        self._num_correct = {k: 0 for k in self.k_s}
        self._num_examples = 0

    @torch.no_grad()
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
            raise NotComputableError(
                "TopKCategoricalAccuracy must have at" "least one example before it can be computed."
            )
        if prefix is None:
            return {f"top_k@{k}": self._num_correct[k] / self._num_examples for k in self.k_s}
        else:
            return {f"{prefix}top_k@{k}": self._num_correct[k] / self._num_examples for k in self.k_s}


class FMax_Slow(torchmetrics.Metric):
    def __init__(self, thresholds: Union[int, Tensor, List[float]] = 100):
        """
        Fmax at argmax_t F(t) = 2 * avg_pr * avg_rc / (avg_pr + avg_rc) where averaged pr's and rc's are sample-centric.
        Args:
            thresholds ():
        """
        super().__init__()
        self.thresholds = np.linspace(0, 1.0, thresholds)
        self.shrink = torch.nn.Hardshrink(lambd=1e-2)
        self.reset()

    def reset(self):
        self._scores = []
        self._threshs = []
        self._n_samples = []

    def update(self, scores: Tensor, targets: Tensor):
        n_samples = scores.shape[0]
        with torch.no_grad():
            if isinstance(scores, Tensor):
                scores = self.shrink(scores).cpu().numpy()

            if isinstance(targets, SparseTensor):
                targets = targets.to_scipy()
            elif targets.layout == torch.sparse_csr:
                targets = ssp.csr_matrix(
                    (targets.values().numpy(), targets.col_indices().numpy(), targets.crow_indices().numpy()),
                    shape=targets.shape,
                )
            elif isinstance(targets, Tensor):
                targets = targets.cpu().numpy()

        fmax, thresh = self.get_fmax(scores=scores, targets=targets, thresholds=self.thresholds)

        self._scores.append(fmax)
        self._threshs.append(thresh)
        self._n_samples.append(n_samples)

    def get_fmax(
            self,
            scores: Union[np.ndarray, ssp.csr_matrix],
            targets: Union[np.ndarray, ssp.csr_matrix],
            thresholds: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Find the threshold with the max F1 score using sparse binary CSR matrices.

        Args:
            scores:
            targets:
            thresholds:

        Returns:

        """
        fmax_t = 0.0, 0.0

        if not isinstance(targets, ssp.csr_matrix):
            targets = ssp.csr_matrix(targets)
        targets_sum = targets.sum(axis=1)

        for thresh in thresholds:
            cut_sc = ssp.csr_matrix(scores > thresh).astype(np.int32)
            correct = cut_sc.multiply(targets).sum(axis=1)

            p = correct / cut_sc.sum(axis=1)
            r = correct / targets_sum
            p, r = np.average(p[np.invert(np.isnan(p))]), np.average(r)
            if np.isnan(p):
                continue
            try:
                fmax_t = max(fmax_t, (2 * p * r / (p + r) if p + r > 0.0 else 0.0, thresh))
            except ZeroDivisionError:
                pass
        return fmax_t

    def compute(self, prefix=None) -> Union[float, Dict[str, float]]:
        if len(self._scores) == 0:
            raise NotComputableError("AveragePrecision must have at" "least one example before it can be computed.")

        weighted_avg_score = np.average(self._scores, weights=self._n_samples)
        return weighted_avg_score


class BinnedPrecisionRecallCurve(Metric):
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = False
    TPs: Tensor
    FPs: Tensor
    FNs: Tensor

    def __init__(
            self,
            num_classes: int,
            thresholds: Union[int, Tensor, List[float]] = 100,
            **kwargs: Any,
    ) -> None:
        """
        Computes precision and recall for each class at different thresholds.
        Args:
            num_classes ():
            thresholds ():
            **kwargs ():
        """
        rank_zero_warn(
            "Metric `BinnedPrecisionRecallCurve` has been deprecated in v0.10 and will be completly removed in v0.11."
            " Instead, use the refactored version of `PrecisionRecallCurve` by specifying the `thresholds` argument.",
            DeprecationWarning,
        )
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

        for name in ("TPs", "FPs", "FNs"):
            self.add_state(
                name=name,
                default=torch.zeros(num_classes, self.num_thresholds, dtype=torch.float32),
                dist_reduce_fx="sum",
            )

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """
        Args
            preds: (n_samples, n_classes) tensor
            target: (n_samples, n_classes) tensor
        """
        # binary case
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

    def compute(self) -> Union[Tuple[Tensor, Tensor, Tensor], Tuple[List[Tensor], List[Tensor], List[Tensor]]]:
        """Returns float tensor of size n_classes."""
        precisions = (self.TPs + METRIC_EPS) / (self.TPs + self.FPs + METRIC_EPS)
        recalls = self.TPs / (self.TPs + self.FNs + METRIC_EPS)

        # Need to guarantee that last precision=1 and recall=0, similar to precision_recall_curve
        t_ones = torch.ones(self.num_classes, 1, dtype=precisions.dtype, device=precisions.device)
        precisions = torch.cat([precisions, t_ones], dim=1)
        t_zeros = torch.zeros(self.num_classes, 1, dtype=recalls.dtype, device=recalls.device)
        recalls = torch.cat([recalls, t_zeros], dim=1)
        if self.num_classes == 1:
            return precisions[0, :], recalls[0, :], self.thresholds
        return list(precisions), list(recalls), [self.thresholds for _ in range(self.num_classes)]


class Smin(BinnedPrecisionRecallCurve):
    def __init__(self, num_classes: int, thresholds: Union[int, Tensor, List[float]] = 100, **kwargs: Any) -> None:
        """
        A metric that considered the unbalanced information content (IC) of GO terms. The IC is defined as the negative
        of the base 2 logarithm of the frequency of the term in the reference set. The Smin metric is the minimum IC
        of all the terms in the predicted set that are also in the reference set.
        Args:
            num_classes ():
            thresholds ():
            **kwargs ():
        """
        super().__init__(num_classes, thresholds, **kwargs)
        self.reset()

    def reset(self) -> None:
        super().reset()
        self.occurence_counts: Optional[Tensor] = None
        self.num_samples: int = 0

    @torch.no_grad()
    def update(self, preds: Tensor, target: Tensor) -> None:
        """
        Args
            preds: (n_samples, n_classes) tensor
            target: (n_samples, n_classes) tensor
        """
        if target.size(1) != self.num_classes:
            self.num_classes = target.size(1)
            for name in ("FPs", "FNs"):
                self.add_state(
                    name=name,
                    default=torch.zeros(self.num_classes, self.num_thresholds, dtype=torch.float32, device=self.device),
                    dist_reduce_fx="sum",
                )
            self.reset()

        # binary case
        if len(preds.shape) == len(target.shape) == 1:
            preds = preds.reshape(-1, 1)
            target = target.reshape(-1, 1)

        if len(preds.shape) == len(target.shape) + 1:
            target = to_onehot(target, num_classes=self.num_classes)

        target = target == 1
        # Iterate one threshold at a time to conserve memory
        for i in range(self.num_thresholds):
            predictions = preds >= self.thresholds[i]
            self.FPs[:, i] += ((~target) & predictions).sum(dim=0)
            self.FNs[:, i] += (target & (~predictions)).sum(dim=0)

        # Update class counts
        if self.occurence_counts is None:
            self.occurence_counts = target.sum(0)
        else:
            self.occurence_counts = self.occurence_counts.to(target.device) + target.sum(0)

        self.num_samples += preds.shape[0]

    def compute(self) -> Union[List[Tensor], Tensor]:
        if self.num_samples == 0:
            raise NotComputableError(
                "Smin must have at least one example before it can be computed."
            )

        information_content = -torch.log10(self.occurence_counts / self.occurence_counts.sum())
        information_content = torch.nan_to_num(information_content, nan=0, posinf=0, neginf=0)

        remaining_uncertainty = (self.FNs * information_content[:, None]).sum(axis=0) / self.num_samples
        misinformation = (self.FPs * information_content[:, None]).sum(axis=0) / self.num_samples

        s_values = torch.sqrt(remaining_uncertainty ** 2 + misinformation ** 2)
        s_min = s_values.min(axis=0)

        return s_min.values


class BinnedAveragePrecision(BinnedPrecisionRecallCurve):
    def __init__(
            self,
            num_classes: int,
            thresholds: Union[int, Tensor, List[float]] = 100,
            **kwargs: Any,
    ) -> None:
        rank_zero_warn(
            "Metric `BinnedAveragePrecision` has been deprecated in v0.10 and will be completly removed in v0.11."
            " Instead, use the refactored version of `AveragePrecision` by specifying the `thresholds` argument.",
            DeprecationWarning,
        )
        super().__init__(num_classes=num_classes, thresholds=thresholds, **kwargs)

    def compute(self) -> Union[List[Tensor], Tensor]:  # type: ignore
        precisions, recalls, _ = super().compute()
        return self.average_precision_compute_with_precision_recall(precisions, recalls, self.num_classes, average=None)

    def average_precision_compute_with_precision_recall(
            self,
            precision: Tensor,
            recall: Tensor,
            num_classes: int,
            average: Optional[str] = "macro",
            weights: Optional[Tensor] = None,
    ) -> Union[List[Tensor], Tensor]:
        """Computes the average precision score from precision and recall.

        Args:
            precision: precision values
            recall: recall values
            num_classes: integer with number of classes. Not nessesary to provide
                for binary problems.
            average: reduction method for multi-class or multi-label problems
            weights: weights to use when average='weighted'
        """
        # Return the step function integral
        # The following works because the last entry of precision is
        # guaranteed to be 1, as returned by precision_recall_curve
        if num_classes == 1:
            return -torch.sum((recall[1:] - recall[:-1]) * precision[:-1])

        res = []
        for p, r in zip(precision, recall):
            res.append(-torch.sum((r[1:] - r[:-1]) * p[:-1]))

        # Reduce
        if average in ("macro", "weighted"):
            res = torch.stack(res)
            if torch.isnan(res).any():
                warnings.warn(
                    "Average precision score for one or more classes was `nan`. Ignoring these classes in average",
                    UserWarning,
                )
            if average == "macro":
                return res[~torch.isnan(res)].mean()
            weights = torch.ones_like(res) if weights is None else weights
            return (res * weights)[~torch.isnan(res)].sum()
        if average is None or average == "none":
            return res
        allowed_average = ("micro", "macro", "weighted", "none", None)
        raise ValueError(f"Expected argument `average` to be one of {allowed_average}" f" but got {average}")


class AveragePrecisionPairwise(BinnedAveragePrecision):

    def __init__(self, num_classes: int = 1, thresholds: Union[int, Tensor, List[float]] = 100, shrink=1e-2,
                 **kwargs: Any) -> None:
        super().__init__(num_classes, thresholds, **kwargs)

        if shrink:
            self.shrink = torch.nn.Hardshrink(lambd=shrink)
        self.reset()

    def reset(self) -> None:
        super().reset()
        self.num_samples = 0

    @torch.no_grad()
    def update(self, preds: Tensor, target: Tensor) -> None:
        # assume values in [0, 1] range
        if hasattr(self, 'shrink') and self.shrink is not None:
            row, col = (self.shrink(preds) + target).nonzero().T
            preds = preds[row, col]
            target = target[row, col]
        else:
            preds = preds.ravel()
            target = target.ravel()

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

        self.num_samples += preds.shape[0]

    def compute(self) -> Union[List[Tensor], Tensor]:
        if self.num_samples == 0:
            raise NotComputableError(
                "AveragePrecisionPairwise must have at least one example before it can be computed."
            )
        return super().compute()


class FMax_Micro(BinnedPrecisionRecallCurve):

    def __init__(self, num_classes: int = 1, thresholds: Union[int, Tensor, List[float]] = 100, **kwargs: Any) -> None:
        super(BinnedPrecisionRecallCurve, self).__init__(**kwargs)

        self.num_classes = num_classes
        if isinstance(thresholds, int):
            self.num_thresholds = thresholds
            self.thresholds = torch.linspace(0, 1.0, thresholds)

        elif thresholds is not None:
            if not isinstance(thresholds, (list, Tensor)):
                raise ValueError("Expected argument `thresholds` to either be an integer, list of floats or a tensor")
            self.thresholds = torch.tensor(thresholds) if isinstance(thresholds, list) else thresholds
            self.num_thresholds = self.thresholds.numel()

        self.reset()

    def reset(self) -> None:
        super().reset()
        self.TPs = [[] for i in range(self.num_thresholds)]
        self.FPs = [[] for i in range(self.num_thresholds)]
        self.FNs = [[] for i in range(self.num_thresholds)]

    @torch.no_grad()
    def update(self, preds: Tensor, target: Union[SparseTensor, Tensor]) -> None:
        """
        Args
            preds: (n_samples, n_classes) tensor
            target: (n_samples, n_classes) tensor
        """
        if self.num_classes is None:
            self.num_classes = target.size(1) if target.dim() > 1 else 1

        if isinstance(target, SparseTensor):
            target = target.to_dense()

        # binary case
        if len(preds.shape) == len(target.shape) == 1:
            preds = preds.reshape(-1, 1)
            target = target.reshape(-1, 1)

        if len(preds.shape) == len(target.shape) + 1:
            target = to_onehot(target, num_classes=self.num_classes)

        target = target == 1
        # Iterate one threshold at a time to conserve memory
        for i in range(self.num_thresholds):
            predictions = preds >= self.thresholds[i]
            TPs = (target & predictions).sum(dim=1)
            FPs = ((~target) & predictions).sum(dim=1)
            FNs = (target & (~predictions)).sum(dim=1)

            self.TPs[i].append(TPs)
            self.FPs[i].append(FPs)
            self.FNs[i].append(FNs)
    def compute(self) -> Tensor:
        """Returns a float scalar."""
        if not hasattr(self, "TPs") or len(self.TPs[0]) == 0:
            raise NotComputableError("FMax must have at" "least one example before it can be computed.")

        TPs = torch.stack([torch.cat(li) for li in self.TPs])
        FPs = torch.stack([torch.cat(li) for li in self.FPs])
        FNs = torch.stack([torch.cat(li) for li in self.FNs])

        precisions = (TPs + METRIC_EPS) / (TPs + FPs + METRIC_EPS)
        recalls = TPs / (TPs + FNs + METRIC_EPS)

        numerator = 2 * recalls * precisions
        denom = recalls + precisions

        f1_scores = torch.div(numerator, denom)  # shape: n_thresholds x n_samples
        f1_means = f1_scores.nanmean(axis=1)  # shape: n_thresholds x 1
        max_f1_thresh = f1_means.max(axis=0)

        return max_f1_thresh.values



@torch.no_grad()
def precision_recall_curve(y_true: Tensor, y_pred: Tensor, n_thresholds=100, average="micro"):
    assert average == "micro"
    if not isinstance(y_true, Tensor):
        targets = torch.from_numpy(y_true)
    else:
        targets = y_true
    if not isinstance(y_pred, Tensor):
        scores = torch.from_numpy(y_pred)
    else:
        scores = y_pred

    pr_curve = BinnedPrecisionRecallCurve(num_classes=1, thresholds=n_thresholds)
    precision, recall, thresholds = pr_curve(preds=scores, target=targets)
    return precision, recall, thresholds
