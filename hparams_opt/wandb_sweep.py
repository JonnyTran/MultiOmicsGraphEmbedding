import os
import pickle
import random

import sys

sys.path.insert(0, "../MultiOmicsGraphEmbedding/")

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger

from moge.generator.subgraph_generator import SubgraphGenerator
from moge.module.trainer import LightningModel
from moge.module.encoder import EncoderLSTM

import wandb

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
batch_size = 1800
max_length = 1000
test_frac = 0.20
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

train_dataloader = torch.utils.data.DataLoader(
    dataset_train,
    batch_size=None,
    num_workers=10
)

test_dataloader = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=None,
    num_workers=5
)
vocab = dataset_train.tokenizer.word_index


def objective():
    hparams_defaults = {
        "encoding_dim": 128,
        "embedding_dim": 256,
        "n_classes": n_classes,
        "vocab_size": len(vocab),
        "word_embedding_size": None,

        "nb_conv1d_filters": 256,
        "nb_conv1d_kernel_size": 6,
        "nb_max_pool_size": 14,
        "nb_conv1d_dropout": 0.2,
        "nb_conv1d_layernorm": True,

        "nb_lstm_layers": 1,
        "nb_lstm_bidirectional": True,
        "nb_lstm_units": 192,
        "nb_lstm_dropout": 0.0,
        "nb_lstm_hidden_dropout": 0.2,
        "nb_lstm_layernorm": False,

        "nb_attn_heads": 4,
        "nb_attn_dropout": 0.6,

        "nb_cls_dense_size": 512,
        "nb_cls_dropout": 0.5,

        "nb_weight_decay": 1e-5,
        "lr": 1e-3,
    }

    wandb.init(config=hparams_defaults, project="multiplex-rna-embedding")
    config = wandb.config
    config.n_classes = n_classes
    config.vocab_size = len(vocab)

    logger = WandbLogger(project="multiplex-rna-embedding")
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=EPOCHS,
        min_epochs=5,
        gpus=1 if torch.cuda.is_available() else None,
    )

    encoder = EncoderLSTM(config)
    model = LightningModel(encoder, data_path='../MultiOmicsGraphEmbedding/moge/data/gtex_string_network.pickle')

    trainer.fit(model, train_dataloader=train_dataloader, val_dataloaders=test_dataloader)

    wandb.log(logger.experiment._summary)
    # print("logger.metrics", logger.log_metrics)

    # return logger.log_metrics[-1]["val_precision"]


if __name__ == "__main__":
    # wandb.agent('jonnytran/multiplex-rna-embedding/z8yke4u0', function=objective)
    objective()
