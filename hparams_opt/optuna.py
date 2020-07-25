import argparse
import os
import pickle
import random
import shutil

import sys
from typing import Any, Dict, Optional

sys.path.insert(0, "../MultiOmicsGraphEmbedding/")

import optuna
import pytorch_lightning as pl
import torch
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.logging import LightningLoggerBase
from pytorch_lightning.loggers import WandbLogger

from moge.generator.subgraph_generator import SubgraphGenerator
from moge.module.trainer import ModelTrainer
from moge.module.encoder import ConvLSTM

import wandb

# wandb.init(project="multiplex-rna-embedding")

DATASET = '../MultiOmicsGraphEmbedding/moge/data/gtex_string_network.pickle'
EPOCHS = 10
DIR = os.getcwd()
MODEL_DIR = os.path.join(DIR, "result")

with open(DATASET, 'rb') as file:
    network = pickle.load(file)
variables = []
targets = ['go_id']
network.process_feature_tranformer(filter_label=targets[0], min_count=100, verbose=False)
classes = network.feature_transformer[targets[0]].classes_
n_classes = len(classes)
batch_size = 1000
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

train_dataloader = torch.utils.dataset.DataLoader(
    dataset_train,
    batch_size=None,
    num_workers=10
)

test_dataloader = torch.utils.dataset.DataLoader(
    dataset_test,
    batch_size=None,
    num_workers=10
)
vocab = dataset_train.tokenizer.word_index


def objective(trial):
    trial.encoding_dim = trial.suggest_categorical("encoding_dim", [64, 128, 256])
    trial.embedding_dim = trial.suggest_categorical("embedding_dim", [128, 256, 386])
    trial.n_classes = n_classes
    trial.vocab_size = len(vocab)
    trial.word_embedding_size = None

    trial.nb_conv1d_filters = trial.suggest_int("nb_conv1d_filters", 100, 320)
    trial.nb_conv1d_kernel_size = trial.suggest_int("nb_conv1d_kernel_size", 5, 20)
    trial.nb_max_pool_size = trial.suggest_int("nb_max_pool_size", 5, 20)
    trial.nb_conv1d_dropout = trial.suggest_float("nb_conv1d_dropout", 0.0, 0.5)
    trial.nb_conv1d_layernorm = trial.suggest_categorical("nb_conv1d_layernorm", [True, False])

    trial.nb_lstm_layers = 1,  # trial.suggest_int("nb_lstm_layers", 1, 2)
    trial.nb_lstm_units = trial.suggest_int("nb_lstm_units", 100, 320)
    trial.nb_lstm_dropout = trial.suggest_float("nb_lstm_dropout", 0.0, 0.5)
    trial.nb_lstm_hidden_dropout = trial.suggest_float("nb_lstm_hidden_dropout", 0.0, 0.5)
    trial.nb_lstm_bidirectional = trial.suggest_categorical("nb_lstm_bidirectional", [True, False])
    trial.nb_lstm_layernorm = trial.suggest_categorical("nb_lstm_layernorm", [True, False])

    trial.nb_attn_heads = trial.suggest_categorical("nb_attn_heads", [1, 2, 4, 8])
    trial.nb_attn_dropout = trial.suggest_float("nb_attn_dropout", 0.0, 0.8)

    trial.nb_cls_dense_size = trial.suggest_categorical("nb_cls_dense_size", [128, 256, 512, 768])
    trial.nb_cls_dropout = trial.suggest_float("nb_cls_dropout", 0.0, 0.5)

    trial.nb_weight_decay = trial.suggest_float("nb_weight_decay", 1e-5, 1e-2)
    trial.lr = trial.suggest_float("lr", 1e-4, 5e-2)

    # PyTorch Lightning will try to restore model parameters from previous trials if checkpoint
    # filenames match. Therefore, the filenames for each trial must be made unique.
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        os.path.join(MODEL_DIR, "trial_{}".format(trial.number)), monitor="precision"
    )

    # The default logger in PyTorch Lightning writes to event files to be consumed by
    # TensorBoard. We create a simple logger instead that holds the log in memory so that the
    # final accuracy can be obtained after optimization. When using the default logger, the
    # final accuracy could be stored in an attribute of the `Trainer` instead.
    # logger = DictLogger(trial.number)
    logger = WandbLogger(name="trial_{}".format(trial.number), project="multiplex-rna-embedding")
    wandb.init(config=trial)

    trainer = pl.Trainer(
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        max_epochs=EPOCHS,
        gpus=1 if torch.cuda.is_available() else None,
        early_stop_callback=PyTorchLightningPruningCallback(trial, monitor="precision"),
    )

    encoder = ConvLSTM(trial)
    model = ModelTrainer(encoder)

    trainer.fit(model, train_dataloader, test_dataloader)
    print("logger.metrics", logger.log_metrics)

    return logger.log_metrics[-1]["val_precision"]


class DictLogger(LightningLoggerBase):
    """PyTorch Lightning `dict` logger."""

    def __init__(self, name):
        super(DictLogger, self).__init__()
        self.metrics = []
        self.hparams_logged = None
        self._version = name

    def log_hyperparams(self, params):
        self.hparams_logged = params

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        self.metrics.append(metrics)
        super(DictLogger, self).log_metrics(metrics, step)

    @property
    def version(self):
        return self._version

    @property
    def name(self) -> str:
        return "trial_{}".format(self._version)

    @property
    def experiment(self) -> Any:
        return self.hparams_logged


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
