import traceback
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
from torch import Tensor
from torchmetrics import Metric, F1Score, AUROC, MeanSquaredError, Accuracy, BinnedPrecisionRecallCurve, \
    BinnedAveragePrecision
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
                    self.metrics[name] = TopKCategoricalAccuracy(k=max(int(np.log(n_classes)), 1),
                                                                 output_transform=None)
            elif "macro_f1" in name:
                self.metrics[name] = F1Score(num_classes=n_classes, average="macro", top_k=top_k, )
            elif "micro_f1" in name:
                self.metrics[name] = F1Score(num_classes=n_classes, average="micro", top_k=top_k, )

            # elif "fmax_macro" in name:
            #     self.metrics[name] = FMax()

            elif "fmax" in name:
                self.metrics[name] = FMax_Micro()


            elif "auroc" in name:
                self.metrics[name] = AUROC(num_classes=n_classes, average="micro")
            elif "aupr" in name:
                # self.metrics[name] = AveragePrecision_(average="pairwise")
                self.metrics[name] = AveragePrecisionPairwise(average='none')

            elif "mse" in name:
                self.metrics[name] = MeanSquaredError()

            elif "acc" in name:
                self.metrics[name] = Accuracy(top_k=top_k, subset_accuracy=multilabel)

            elif "ogbn" in name or any("ogbn" in s for s in name):
                self.metrics[name] = OGBNodeClfMetrics(
                    NodeEvaluator(name[0] if isinstance(name, (list, tuple)) else name))
            elif "ogbl" in name or any("ogbl" in s for s in name):
                self.metrics[name] = OGBLinkPredMetrics(
                    LinkEvaluator(name[0] if isinstance(name, (list, tuple)) else name))
            elif "ogbg" in name or any("ogbg" in s for s in name):
                self.metrics[name] = OGBNodeClfMetrics(
                    GraphEvaluator(name[0] if isinstance(name, (list, tuple)) else name))
            else:
                logger.warn(f"metric name {name} not supported. Must containing a substring in "
                            f"['precision', 'recall', 'top_k', 'macro_f1', 'micro_f1', 'fmax', 'mse', "
                            f"'auroc', 'aupr', 'acc', 'ogbn', 'ogbl', 'ogbg', ]")
                continue

            # Needed to add the torchmetrics as Modules, so they'll be on the correct CUDA device during training
            if isinstance(self.metrics[name], torchmetrics.metric.Metric):
                setattr(self, str(name), self.metrics[name])

        self.reset_metrics()

    def hot_encode(self, labels, type_as):
        if labels.dim() == 2:
            return labels
        elif labels.dim() == 1:
            labels = torch.eye(self.n_classes)[labels].type_as(type_as)
            return labels

    @torch.no_grad()
    def update_metrics(self, y_pred: Tensor, y_true: Tensor,
                       weights=Optional[Tensor], subset: Union[List[str], str] = None):
        """
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
                    metric_name = (str(metric) if prefix is None else prefix + str(metric)) + \
                                  f"@{self.metrics[metric]._k}"
                    logs[metric_name] = self.metrics[metric].compute()

                else:
                    metric_name = str(metric) if prefix is None else prefix + str(metric)
                    logs[metric_name] = self.metrics[metric].compute()

            except NotComputableError as nce:
                # logger.warn(nce)
                pass

            except ValueError as ve:
                logger.warn(ve) if 'No samples to concatenate' in ve.__str__() else None
                pass

            except Exception as e:
                logger.error(f"Metric: {metric}, {type(e)}:{str(e)}\r")
                traceback.print_exc()

        # Needed for Precision(average=False) metrics
        logs = {k: v.mean() if isinstance(v, Tensor) and v.numel() > 1 else v \
                for k, v in logs.items()}

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
            raise NotComputableError("TopKCategoricalAccuracy must have at"
                                     "least one example before it can be computed.")
        if prefix is None:
            return {f"top_k@{k}": self._num_correct[k] / self._num_examples for k in self.k_s}
        else:
            return {f"{prefix}top_k@{k}": self._num_correct[k] / self._num_examples for k in self.k_s}


class AveragePrecisionPairwise(BinnedAveragePrecision):
    def __init__(self, num_classes: int = 1, thresholds: Union[int, Tensor, List[float]] = 100, **kwargs: Any) -> None:
        super().__init__(num_classes, thresholds, **kwargs)

        self.shrink = torch.nn.Hardshrink(lambd=1e-2)

    @torch.no_grad()
    def update(self, preds: Tensor, target: Tensor) -> None:
        # assume values in [0, 1] range
        row, col = (self.shrink(preds) + target).nonzero().T
        preds = preds[row, col]
        target = target[row, col]

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


class FMax_Micro(BinnedPrecisionRecallCurve):
    def __init__(self, num_classes: int = 1, thresholds: Union[int, Tensor, List[float]] = 100,
                 **kwargs: Any) -> None:
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
    def update(self, preds: Tensor, target: Tensor) -> None:
        """
        Args
            preds: (n_samples, n_classes) tensor
            target: (n_samples, n_classes) tensor
        """
        if self.num_classes is None:
            self.num_classes = target.size(1) if target.dim() > 1 else 1

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
        """Returns float tensor of size n_classes."""
        if not hasattr(self, "TPs") or len(self.TPs[0]) == 0:
            raise NotComputableError("FMax must have at"
                                     "least one example before it can be computed.")

        TPs = torch.stack([torch.cat(li) for li in self.TPs])
        FPs = torch.stack([torch.cat(li) for li in self.FPs])
        FNs = torch.stack([torch.cat(li) for li in self.FNs])

        precisions = (TPs + METRIC_EPS) / (TPs + FPs + METRIC_EPS)
        recalls = TPs / (TPs + FNs + METRIC_EPS)

        numerator = 2 * recalls * precisions
        denom = recalls + precisions + METRIC_EPS

        f1_scores = torch.div(numerator, denom)  # shape: n_thresholds x n_samples
        f1_means = f1_scores.mean(axis=1)  # shape: n_thresholds x 1
        max_f1 = f1_means.max(axis=0).values

        return max_f1


class FMax_Slow(torchmetrics.Metric):
    def __init__(self, thresholds: Union[int, Tensor, List[float]] = 100):
        """
        Fmax at argmax_t F(t) = 2 * avg_pr * avg_rc / (avg_pr + avg_rc) where averaged pr's and rc's are sample-centric.
        Args:
            thresholds ():
        """
        super().__init__()
        self.thresholds = np.linspace(0, 1.0, thresholds)
        self.reset()

    def reset(self):
        self._scores = []
        self._threshs = []
        self._n_samples = []

    def update(self, scores: Tensor, targets: Tensor):
        n_samples = scores.shape[0]
        with torch.no_grad():
            if isinstance(scores, Tensor):
                scores = scores.cpu().numpy()
            if isinstance(targets, Tensor):
                targets = targets.cpu().numpy()

        fmax, thresh = get_fmax(scores, targets, thresholds=self.thresholds)

        self._scores.append(fmax)
        self._threshs.append(thresh)
        self._n_samples.append(n_samples)

    # def update(self, scores: Tensor, targets: Tensor):
    #     n_samples = scores.shape[0]
    #     if isinstance(scores, Tensor):
    #         scores = scores.detach().cpu().numpy()
    #     if isinstance(targets, Tensor):
    #         targets = targets.detach().cpu().numpy()
    #
    #     fmax_t = 0.0, 0.0
    #     for thresh in self.thresholds:
    #         pred = ssp.csr_matrix((scores >= thresh).astype(np.int32))
    #         TP = pred.multiply(targets).sum(axis=1)
    #
    #         with warnings.catch_warnings():
    #             warnings.simplefilter('ignore')
    #             precision = TP / pred.sum(axis=1)
    #             recall = TP / targets.sum(axis=1)
    #             precision, recall = np.average(precision[np.invert(np.isnan(precision))]), np.average(recall)
    #         if np.isnan(precision):
    #             continue
    #
    #         try:
    #             fmax = 2 * precision * recall / (precision + recall) if precision + recall > 0.0 else 0.0
    #             fmax_t = max(fmax_t, (fmax, thresh))
    #         except ZeroDivisionError:
    #             pass
    #
    #     self._scores.append(fmax_t[0])
    #     self._n_samples.append(n_samples)

    def compute(self, prefix=None) -> Union[float, Dict[str, float]]:
        if len(self._scores) == 0:
            raise NotComputableError("AveragePrecision must have at"
                                     "least one example before it can be computed.")

        weighted_avg_score = np.average(self._scores, weights=self._n_samples)
        return weighted_avg_score

def get_fmax(scores: np.ndarray, targets: np.ndarray, thresholds: np.ndarray) -> Tuple[float, float]:
    fmax_t = 0.0, 0.0

    for thresh in thresholds:
        preds = (scores >= thresh).astype(np.int32)
        TPs = (preds * targets).sum(axis=1).ravel()

        precisions = np.true_divide(TPs, preds.sum(axis=1).ravel())
        recalls = np.true_divide(TPs, targets.sum(axis=1).ravel())

        avg_pr = np.average(precisions[~np.isnan(precisions)])
        avg_rc = np.average(recalls)

        if np.isnan(avg_pr): continue

        try:
            if avg_pr + avg_rc > 0.0:
                fmax = (2 * avg_pr * avg_rc) / (avg_pr + avg_rc)
            else:
                fmax = 0.0

            if fmax > fmax_t[0]:
                # Updates higher fmax_t
                fmax_t = (fmax, thresh)
        except Exception:  # ZeroDivisionError
            continue

    return fmax_t

@torch.no_grad()
def precision_recall_curve(y_true: Tensor, y_pred: Tensor, n_thresholds=100, average='micro'):
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
