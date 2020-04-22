import argparse
import os
import pickle
import random
import shutil

import optuna
import pytorch_lightning as pl
import torch
from ignite.metrics import Precision, Recall
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.loggers import LightningLoggerBase

from .encoder import EncoderLSTM
from ..generator.subgraph_generator import SubgraphGenerator

EPOCHS = 10
DIR = os.getcwd()
MODEL_DIR = os.path.join(DIR, "result")

with open('../MultiOmicsGraphEmbedding/moge/data/gtex_string_network.pickle', 'rb') as file:
    network = pickle.load(file)
variables = []
targets = ['go_id']
network.process_feature_tranformer(min_count=100, verbose=True)
classes = network.feature_transformer[targets[0]].classes_
batch_size = 2000
max_length = 1000
test_frac = 0.05
n_steps = int(400000 / batch_size)
directed = False
seed = random.randint(0, 1000)
network.split_stratified(directed=directed, stratify_label=targets[0], stratify_omic=False,
                         n_splits=int(1 / test_frac), dropna=True, seed=seed, verbose=False)

dataset_train = network.get_train_generator(
    SubgraphGenerator, variables=variables, targets=targets,
    sampling="bfs", batch_size=batch_size, agg_mode=None,
    method="GAT", adj_output="coo",
    compression="log", n_steps=n_steps, directed=directed,
    maxlen=max_length, padding='post', truncating='post', variable_length=False,
    seed=seed, verbose=False)

dataset_test = network.get_test_generator(
    SubgraphGenerator, variables=variables, targets=targets,
    sampling='all', batch_size=batch_size, agg_mode=None,
    method="GAT", adj_output="coo",
    compression="log", n_steps=1, directed=directed,
    maxlen=max_length, padding='post', truncating='post', variable_length=False,
    seed=seed, verbose=False)


class LightningEncoderLSTM(pl.LightningModule):
    def __init__(self, trial):
        super(LightningEncoderLSTM, self).__init__()

        self._model = EncoderLSTM(trial)

    def forward(self, X):
        return self._model(X)

    @pl.data_loader
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset_train,
            batch_size=None,
            num_workers=10
        )

    @pl.data_loader
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset_test,
            batch_size=None,
            num_workers=10
        )

    def training_step(self, batch, batch_nb):
        X, y, train_weights = batch

        Y_hat = self.forward(X)
        loss = self._model.loss(Y_hat, y, None)

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
        X, y, train_weights = batch

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
                                     lr=self.hparams.lr,
                                     # weight_decay=self.hparams.nb_weight_decay
                                     )
        return optimizer


class DictLogger(LightningLoggerBase):
    """PyTorch Lightning `dict` logger."""

    def __init__(self, version):
        super(DictLogger, self).__init__()
        self.metrics = []
        self._version = version

    def log_metrics(self, metric, step=None):
        self.metrics.append(metric)

    @property
    def version(self):
        return self._version


def objective(trial):
    # PyTorch Lightning will try to restore model parameters from previous trials if checkpoint
    # filenames match. Therefore, the filenames for each trial must be made unique.
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        os.path.join(MODEL_DIR, "trial_{}".format(trial.number)), monitor="precision"
    )

    # The default logger in PyTorch Lightning writes to event files to be consumed by
    # TensorBoard. We create a simple logger instead that holds the log in memory so that the
    # final accuracy can be obtained after optimization. When using the default logger, the
    # final accuracy could be stored in an attribute of the `Trainer` instead.
    logger = DictLogger(trial.number)

    trainer = pl.Trainer(
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        max_epochs=EPOCHS,
        gpus=0 if torch.cuda.is_available() else None,
        early_stop_callback=PyTorchLightningPruningCallback(trial, monitor="accuracy"),
    )

    model = LightningEncoderLSTM(trial)
    trainer.fit(model)

    return logger.metrics[-1]["precision"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Lightning example.")
    parser.add_argument(
        "--pruning",
        "-p",
        action="store_true",
        help="Activate the pruning feature. `MedianPruner` stops unpromising "
             "trials at the early stages of training.",
    )
    args = parser.parse_args()

    pruner = optuna.pruners.MedianPruner() if args.pruning else optuna.pruners.NopPruner()

    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=100, timeout=600)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    shutil.rmtree(MODEL_DIR)
