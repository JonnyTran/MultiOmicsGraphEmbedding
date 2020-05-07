import torch

import pytorch_lightning as pl


class ClassificationLoss(torch.nn.Module):
    def __init__(self, hparams) -> None:
        super(ClassificationLoss, self).__init__()

        self.label_size = hparams.n_classes
        self.loss_type = hparams.loss_type

        if self.loss_type == "SOFTMAX_CROSS_ENTROPY":
            self.criterion = torch.nn.CrossEntropyLoss(hparams.class_weight)
        elif self.loss_type == "BCE_WITH_LOGITS":
            self.criterion = torch.nn.BCEWithLogitsLoss()
        else:
            raise TypeError(
                "Unsupported loss type: %s" % (
                    self.loss_type))

    def forward(self, y_pred, y_true):
        pass
