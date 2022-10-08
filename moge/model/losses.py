import codecs as cs
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_sparse import SparseTensor


class ClassificationLoss(nn.Module):
    def __init__(self, loss_type: str, n_classes: int = None, class_weight: Tensor = None, pos_weight: Tensor = None,
                 multilabel: bool = False, reduction: str = "mean"):
        super().__init__()
        self.n_classes = n_classes
        self.loss_type = loss_type
        self.multilabel = multilabel
        self.reduction = reduction

        print(f"INFO: Using {loss_type}")
        print(f"class_weight for {class_weight.shape} classes") if class_weight is not None else None
        print(f"pos_weight for {pos_weight.shape} classes") if pos_weight is not None else None

        if loss_type == "SOFTMAX_CROSS_ENTROPY":
            self.criterion = torch.nn.CrossEntropyLoss(weight=class_weight, reduction=reduction)
        elif loss_type == "NEGATIVE_LOG_LIKELIHOOD":
            self.criterion = torch.nn.NLLLoss(class_weight, reduction=reduction)
        elif loss_type == "SOFTMAX_FOCAL_CROSS_ENTROPY":
            self.criterion = FocalLoss(n_classes, "SOFTMAX")
        elif loss_type == "SIGMOID_FOCAL_CROSS_ENTROPY":
            self.criterion = FocalLoss(n_classes, "SIGMOID")
        elif loss_type == "BCE_WITH_LOGITS":
            self.criterion = torch.nn.BCEWithLogitsLoss(weight=class_weight, reduction=reduction, pos_weight=pos_weight)
        elif loss_type == "BCE":
            self.criterion = torch.nn.BCELoss(weight=class_weight, reduction=reduction)
        elif loss_type == "MULTI_LABEL_MARGIN":
            self.criterion = torch.nn.MultiLabelMarginLoss(weight=class_weight, reduction=reduction)
        elif loss_type == "KL_DIVERGENCE":
            self.criterion = torch.nn.KLDivLoss(reduction=reduction)
        elif loss_type == "PU_LOSS_WITH_LOGITS":
            self.criterion = PULoss(prior=torch.tensor(1 / 1000))
        elif loss_type == "LINK_PRED_WITH_LOGITS":
            self.criterion = LinkPredLoss()
        elif "CONTRASTIVE" in loss_type:
            assert "LOGITS" not in loss_type
            self.criterion = ContrastiveLoss()
        else:
            raise TypeError(f"Unsupported loss type: {loss_type}")

    def forward(self, logits: Tensor, targets: Tensor, weights: Optional[Tensor] = None) -> Tensor:
        """

        Args:
            logits (torch.Tensor): predicted labels or logits
            targets (torch.Tensor): true labels
            weights (): Sample weights.

        Returns:

        """
        if isinstance(targets, SparseTensor):
            targets = targets.to_dense()

        if self.multilabel:
            assert self.loss_type in ["BCE_WITH_LOGITS", "BCE", "PU_LOSS_WITH_LOGITS",
                                      "SIGMOID_FOCAL_CROSS_ENTROPY", "MULTI_LABEL_MARGIN"], \
                f"Multilabel loss in compatible with loss type: {self.loss_type}"
            targets = targets.type_as(logits)
        else:
            if self.loss_type in ["SOFTMAX_CROSS_ENTROPY", "NEGATIVE_LOG_LIKELIHOOD", "SOFTMAX_FOCAL_CROSS_ENTROPY"] \
                    and targets.dim() == 1:
                targets = torch.eye(self.n_classes, device=logits.device, dtype=torch.long)[targets]

        loss = self.criterion.forward(logits, targets)

        if isinstance(weights, Tensor) and weights.numel() and self.reduction == "none":
            if loss.dim() > 1 and loss.size(1) > 1:
                loss = loss.sum(dim=1)
            loss = (weights * loss).sum() / weights.sum()

        return loss


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07) -> None:
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, pos_edges: Tensor, neg_batch: Tensor):
        pos_logits = torch.div(pos_edges, self.temperature)
        neg_logits = torch.div(neg_batch, self.temperature)

        # For numerical stability
        logits_max, _ = torch.max(torch.cat([pos_logits.unsqueeze(1), neg_logits], dim=1),
                                  dim=1, keepdim=False)
        pos_logits = pos_logits - logits_max.detach()
        neg_logits = neg_logits - logits_max.unsqueeze(1).detach()

        mean_log_prob_pos = pos_logits - torch.log(torch.exp(neg_logits).sum(1))

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss


class PULoss(nn.Module):
    """wrapper of loss function for PU learning"""

    def __init__(self, prior, loss=(lambda x: torch.sigmoid(x)),
                 gamma=1, beta=0, nnPU=True):
        super(PULoss, self).__init__()
        if not 0 < prior < 1:
            raise NotImplementedError("The class prior should be in (0, 1)")
        self.prior = prior
        print("PULoss Prior: ", prior)
        self.gamma = gamma
        self.beta = beta
        self.loss_func = loss  # lambda x: (torch.tensor(1., device=x.device) - torch.sign(x))/torch.tensor(2, device=x.device)
        self.nnPU = nnPU
        self.positive = 1
        self.unlabeled = -1
        self.min_count = torch.tensor(1.)

    def forward(self, y_pred: Tensor, y_true: Tensor, weights=None):
        assert (y_pred.shape == y_true.shape)
        positive, unlabeled = y_true == self.positive, y_true == self.unlabeled
        positive, unlabeled = positive.type(torch.float), unlabeled.type(torch.float)
        if y_pred.is_cuda:
            self.min_count = self.min_count.type_as(y_pred)
            self.prior = self.prior.type_as(y_pred)
        n_positive, n_unlabeled = torch.max(self.min_count, torch.sum(positive)), \
                                  torch.max(self.min_count, torch.sum(unlabeled))

        y_positive = self.loss_func(positive * y_pred) * positive
        y_positive_inv = self.loss_func(-positive * y_pred) * positive
        y_unlabeled = self.loss_func(-unlabeled * y_pred) * unlabeled

        positive_risk = self.prior * torch.sum(y_positive) / n_positive
        negative_risk = - self.prior * torch.sum(y_positive_inv) / n_positive + torch.sum(y_unlabeled) / n_unlabeled

        if negative_risk < -self.beta and self.nnPU:
            return -self.gamma * negative_risk
        else:
            return positive_risk + negative_risk

class LinkPredLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pos_pred: Tensor, neg_pred: Tensor, pos_weights: Tensor = None, neg_weights=None) -> Tensor:
        if pos_weights is None:
            pos_loss = -torch.mean(F.logsigmoid(pos_pred), dim=-1)
            neg_loss = -torch.mean(F.logsigmoid(-neg_pred.view(-1)), dim=-1)
            loss = (pos_loss + neg_loss) / 2
        else:
            pos_loss = - (pos_weights * F.logsigmoid(pos_pred)).sum() / pos_weights.sum()
            neg_loss = - torch.mean(F.logsigmoid(-neg_pred.view(-1)), dim=-1)
            loss = (pos_loss + neg_loss) / 2

        # scores_pred = torch.cat([pos_pred, neg_pred.view(-1)])
        # scores_true = torch.cat([torch.ones_like(pos_pred, requires_grad=False),
        #                          torch.zeros_like(neg_pred.view(-1), requires_grad=False)])
        #
        # loss = F.binary_cross_entropy_with_logits(scores_pred, scores_true)
        return loss


class FocalLoss(nn.Module):
    """Softmax focal loss
    references: Focal Loss for Dense Object Detection
                https://github.com/Hsuxu/FocalLoss-PyTorch
    """

    def __init__(self, label_size, activation_type="SOFTMAX",
                 gamma=2.0, alpha=0.25, epsilon=1.e-9):
        super(FocalLoss, self).__init__()
        self.num_cls = label_size
        self.activation_type = activation_type
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon

    def forward(self, logits, target, weight=None):
        """
        Args:
            logits: model's output, shape of [batch_size, num_cls]
            target: ground truth labels, shape of [batch_size]
        Returns:
            shape of [batch_size]
        """
        if self.activation_type == "SOFTMAX":
            idx = target.view(-1, 1).long()
            one_hot_key = torch.zeros(idx.size(0), self.num_cls,
                                      dtype=torch.float, device=logits.device)
            one_hot_key = one_hot_key.scatter_(1, idx, 1)
            logits = torch.softmax(logits, dim=-1)
            loss = -self.alpha * one_hot_key * \
                   torch.pow((1 - logits), self.gamma) * \
                   (logits + self.epsilon).log()
            loss = loss.sum(1)

        elif self.activation_type == "SIGMOID":
            multi_hot_key = target
            logits = torch.sigmoid(logits)
            zero_hot_key = 1 - multi_hot_key
            loss = -self.alpha * multi_hot_key * \
                   torch.pow((1 - logits), self.gamma) * \
                   (logits + self.epsilon).log()
            loss += -(1 - self.alpha) * zero_hot_key * \
                    torch.pow(logits, self.gamma) * \
                    (1 - logits + self.epsilon).log()
        else:
            raise TypeError("Unknown activation type: " + self.activation_type
                            + "Supported activation types: ")
        return loss.mean()


def get_hierarchical_relations(hierarchy_file, label_map):
    """ get parent-children relationships from given hierar_taxonomy
        hierar_taxonomy: parent_label \t child_label_0 \t child_label_1 \n
    """
    hierar_relations = {}
    with cs.open(hierarchy_file, "r", "utf8") as f:
        for line in f:
            line_split = line.strip("\n").split("\t")
            parent_label, children_label = line_split[0], line_split[1:]
            if parent_label not in label_map:
                continue
            parent_label_id = label_map[parent_label]
            children_label_ids = [label_map[child_label] \
                                  for child_label in children_label if child_label in label_map]
            hierar_relations[parent_label_id] = children_label_ids
    return hierar_relations


