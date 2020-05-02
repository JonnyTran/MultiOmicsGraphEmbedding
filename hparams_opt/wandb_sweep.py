import pickle
import random

import sys
from argparse import ArgumentParser
sys.path.insert(0, "../MultiOmicsGraphEmbedding/")

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping

from moge.generator.subgraph_generator import SubgraphGenerator
from moge.module.trainer import LightningModel
from moge.module.encoder import EncoderLSTM

DATASET = '../MultiOmicsGraphEmbedding/moge/data/gtex_string_network.pickle'
with open(DATASET, 'rb') as file:
    network = pickle.load(file)

MAX_EPOCHS = 10
min_count = 500
batch_size = 1000
max_length = 1000
n_steps = int(400000 / batch_size)
directed = False

variables = []
targets = ['go_id']
network.process_feature_tranformer(filter_label=targets[0], min_count=min_count, verbose=False)
classes = network.feature_transformer[targets[0]].classes_
n_classes = len(classes)
seed = random.randint(0, 1000)

split_idx = 0
dataset_train = network.get_train_generator(
    SubgraphGenerator, split_idx=split_idx, variables=variables, targets=targets,
    traversal="bfs", batch_size=batch_size, agg_mode=None,
    method="GAT", adj_output="coo",
    sampling="all", n_steps=n_steps, directed=directed,
    maxlen=max_length, padding='post', truncating='post', variable_length=False,
    seed=seed, verbose=False)

dataset_test = network.get_test_generator(
    SubgraphGenerator, split_idx=split_idx, variables=variables, targets=targets,
    traversal='all', batch_size=batch_size, agg_mode=None,
    method="GAT", adj_output="coo",
    sampling="log", n_steps=1, directed=directed,
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


def train(hparams):
    hparams.n_classes = n_classes
    hparams.vocab_size = len(vocab)

    logger = WandbLogger()
    # wandb.init(config=hparams, project="multiplex-rna-embedding")

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[EarlyStopping(monitor='val_loss', patience=3), EarlyStopping(monitor='loss', patience=3)],
        min_epochs=3, max_epochs=MAX_EPOCHS,
        gpus=[random.randint(0, 3)] if torch.cuda.is_available() else None,
        weights_summary='top',
    )
    encoder = EncoderLSTM(hparams)
    model = LightningModel(encoder)

    # wandb.watch(model, criterion="val_loss", log="parameters", log_freq=100)

    trainer.fit(model, train_dataloader=train_dataloader, val_dataloaders=test_dataloader)
    # trainer.run_evaluation(test_mode=False)
    # wandb.log(logger.experiment._summary)


if __name__ == "__main__":
    parser = ArgumentParser()
    # parametrize the network
    parser.add_argument('--encoding_dim', type=int, default=128)
    parser.add_argument('--embedding_dim', type=int, default=256)
    parser.add_argument('--word_embedding_size', type=int, default=None)

    parser.add_argument('--nb_conv1_filters', type=int, default=192)
    parser.add_argument('--nb_conv1_kernel_size', type=int, default=10)
    parser.add_argument('--nb_conv1_dropout', type=float, default=0.2)
    parser.add_argument('--nb_conv1_batchnorm', type=bool, default=True)

    parser.add_argument('--nb_conv2_filters', type=int, default=128)
    parser.add_argument('--nb_conv2_kernel_size', type=int, default=3)
    parser.add_argument('--nb_conv2_batchnorm', type=bool, default=True)

    parser.add_argument('--nb_max_pool_size', type=int, default=13)

    parser.add_argument('--nb_lstm_units', type=int, default=100)
    parser.add_argument('--nb_lstm_bidirectional', type=bool, default=False)
    parser.add_argument('--nb_lstm_hidden_dropout', type=float, default=0.0)
    parser.add_argument('--nb_lstm_layernorm', type=bool, default=False)

    parser.add_argument('--nb_attn_heads', type=int, default=4)
    parser.add_argument('--nb_attn_dropout', type=float, default=0.5)

    parser.add_argument('--nb_cls_dense_size', type=int, default=512)
    parser.add_argument('--nb_cls_dropout', type=float, default=0.2)

    parser.add_argument('--nb_weight_decay', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=1e-3)

    # add all the available options to the trainer
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    train(args)
