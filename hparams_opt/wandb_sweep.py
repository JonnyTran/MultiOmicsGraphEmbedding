import pickle
import random

import sys
from argparse import ArgumentParser

import logging

logger = logging.getLogger("wandb")
logger.setLevel(logging.ERROR)

sys.path.insert(0, "../MultiOmicsGraphEmbedding/")

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping

from moge.generator.subgraph_generator import SubgraphGenerator
from moge.module.trainer import LightningModel, EncoderEmbedderClassifier

DATASET = '../MultiOmicsGraphEmbedding/moge/data/gtex_string_network.pickle'

def train(hparams):
    print(hparams)
    with open(DATASET, 'rb') as file:
        network = pickle.load(file)

    MAX_EPOCHS = 20
    min_count = hparams.classes_min_count
    batch_size = hparams.batch_size
    max_length = hparams.max_length
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

    hparams.n_classes = n_classes
    hparams.vocab_size = len(vocab)

    logger = WandbLogger()
    # wandb.init(config=hparams, project="multiplex-rna-embedding")

    eec = EncoderEmbedderClassifier(hparams)
    model = LightningModel(eec)

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[EarlyStopping(monitor='val_loss', patience=5), EarlyStopping(monitor='loss', patience=3)],
        min_epochs=3, max_epochs=MAX_EPOCHS,
        gpus=[random.randint(0, 3)] if torch.cuda.is_available() else None,
        weights_summary='top',
        amp_level='O1', precision=16,
    )
    trainer.fit(model, train_dataloader=train_dataloader, val_dataloaders=test_dataloader)


if __name__ == "__main__":
    parser = ArgumentParser()
    # parametrize the network
    parser.add_argument('--encoder', type=str, default="Albert")
    parser.add_argument('--encoding_dim', type=int, default=128)
    parser.add_argument('--embedding_dim', type=int, default=256)
    parser.add_argument('--word_embedding_size', type=int, default=22)
    parser.add_argument('--max_length', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=1000)

    parser.add_argument('--num_hidden_layers', type=int, default=1)
    parser.add_argument('--num_hidden_groups', type=int, default=1)
    parser.add_argument('--hidden_dropout_prob', type=float, default=0.2)
    parser.add_argument('--attention_probs_dropout_prob', type=float, default=0.2)
    parser.add_argument('--num_attention_heads', type=int, default=4)
    parser.add_argument('--intermediate_size', type=int, default=512)

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

    parser.add_argument('--embedder', type=str, default="GAT")
    parser.add_argument('--nb_attn_heads', type=int, default=4)
    parser.add_argument('--nb_attn_dropout', type=float, default=0.5)

    parser.add_argument('--classifier', type=str, default="Dense")
    parser.add_argument('--nb_cls_dense_size', type=int, default=512)
    parser.add_argument('--nb_cls_dropout', type=float, default=0.2)
    parser.add_argument('--classes_min_count', type=int, default=100)

    parser.add_argument('--nb_weight_decay', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=1e-3)

    parser.add_argument('--optimizer', type=str, default="adam")

    # add all the available options to the trainer
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    train(args)
