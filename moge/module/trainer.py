import pytorch_lightning as pl
import torch
from ignite.metrics import Precision, Recall


class LightningModel(pl.LightningModule):
    def __init__(self, model):
        super(LightningModel, self).__init__()

        self._model = model
        self.init_metrics()

    def forward(self, X):
        return self._model(X)

    def training_step(self, batch, batch_nb):
        X, y, weights = batch

        Y_hat = self.forward(X)
        loss = self._model.loss(Y_hat, y, weights)

        self.update_metrics(Y_hat, y, training=True)
        progress_bar = {
            "precision": self.precision.compute(),
            "recall": self.recall.compute()
        }

        return {'loss': loss,
                'progress_bar': progress_bar,
                }

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()

        tensorboard_logs = {
            "loss": avg_loss,
            "precision": self.precision.compute(),
            "recall": self.recall.compute(),
        }
        self.reset_metrics(training=True)
        return {"loss": avg_loss,
                "progress_bar": tensorboard_logs,
                "log": tensorboard_logs,
                }

    def validation_step(self, batch, batch_nb):
        X, y, weights = batch

        Y_hat = self._model.forward(X)
        loss = self._model.loss(Y_hat, y, None)

        self.update_metrics(Y_hat, y, training=False)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {
            "val_loss": avg_loss,
            "val_precision": self.precision_val.compute(),
            "val_recall": self.recall_val.compute(),
        }

        results = {"progress_bar": tensorboard_logs,
                   "log": tensorboard_logs}
        self.reset_metrics(training=False)
        print(tensorboard_logs)
        return results

    def init_metrics(self):
        self.precision = Precision(average=True, is_multilabel=True)
        self.recall = Recall(average=True, is_multilabel=True)
        self.precision_val = Precision(average=True, is_multilabel=True)
        self.recall_val = Recall(average=True, is_multilabel=True)

    def update_metrics(self, y_pred, y_true, training):
        if training:
            self.precision.update(((y_pred > 0.5).type_as(y_true), y_true))
            self.recall.update(((y_pred > 0.5).type_as(y_true), y_true))
        else:
            self.precision_val.update(((y_pred > 0.5).type_as(y_true), y_true))
            self.recall_val.update(((y_pred > 0.5).type_as(y_true), y_true))

    def reset_metrics(self, training):
        if training:
            self.precision.reset()
            self.recall.reset()
        else:
            self.precision_val.reset()
            self.recall_val.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self._model.hparams.lr,
                                     weight_decay=self._model.hparams.nb_weight_decay
                                     )
        return optimizer

