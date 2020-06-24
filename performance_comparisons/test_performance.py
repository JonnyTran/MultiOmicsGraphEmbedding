import logging
import pickle
import random
import sys
from argparse import ArgumentParser, Namespace
import wandb

logger = logging.getLogger("wandb")
logger.setLevel(logging.ERROR)

sys.path.insert(0, "../MultiOmicsGraphEmbedding/")

import pytorch_lightning as pl
from pytorch_lightning.trainer import Trainer
import torch
from torch_geometric.datasets import PPI, CoraFull, AMiner

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping

from cogdl.datasets.han_data import ACM_HANDataset, DBLP_HANDataset, IMDB_HANDataset
from cogdl.datasets.matlab_matrix import BlogcatalogDataset

from moge.module.multiplex import MultiplexEmbedder, HeterogeneousMultiplexEmbedder
from moge.module.trainer import ModelTrainer
from moge.module.methods import MetaPath2Vec, HAN, GTN
from moge.generator.datasets import HeterogeneousNetworkDataset
from pytorch_lightning.loggers import WandbLogger


def train(hparams):
    if hparams.dataset == "ACM":
        dataset = HeterogeneousNetworkDataset(ACM_HANDataset(),
                                              node_types=["P"], metapath=["PAP", "PLP"],
                                              train_ratio=hparams.train_ratio)
    elif hparams.dataset == "DBLP":
        dataset = HeterogeneousNetworkDataset(DBLP_HANDataset(),
                                              node_types=["A"], metapath=["APA", "ACA", "ATA"],
                                              train_ratio=hparams.train_ratio)
    elif hparams.dataset == "IMDB":
        dataset = HeterogeneousNetworkDataset(IMDB_HANDataset(),
                                              node_types=["M"], metapath=["MAM", "MDM", "MYM"],
                                              train_ratio=hparams.train_ratio)
    elif hparams.dataset == "AMiner":
        dataset = HeterogeneousNetworkDataset(AMiner("datasets/aminer"),
                                              node_types=None, head_node_type="author",
                                              train_ratio=hparams.train_ratio)
    elif hparams.dataset == "BlogCatalog":
        dataset = HeterogeneousNetworkDataset("/home/jonny/Downloads/blogcatalog6k.mat",
                                              node_types=["user", "tag"],
                                              train_ratio=hparams.train_ratio)
        dataset.name = "BlogCatalog3"

    NUM_GPUS = 1
    METRICS = ["accuracy", "precision", "recall"]

    if hparams.method == "HAN":
        USE_AMP = False
        model_hparams = {
            "embedding_dim": 128,
            "batch_size": 128 * NUM_GPUS,
            "train_ratio": dataset.train_ratio,
            "loss_type": "SOFTMAX_CROSS_ENTROPY",
            "n_classes": dataset.n_classes,
            "lr": 0.001 * NUM_GPUS,
        }
        model = HAN(Namespace(**model_hparams), dataset=dataset, metrics=METRICS)
    elif hparams.method == "GTN":
        USE_AMP = True
        model_hparams = {
            "embedding_dim": 128,
            "num_channels": 1,
            "batch_size": 128 * NUM_GPUS,
            "train_ratio": dataset.train_ratio,
            "loss_type": "SOFTMAX_CROSS_ENTROPY",
            "n_classes": dataset.n_classes,
            "lr": 0.001 * NUM_GPUS,
        }
        model = GTN(Namespace(**model_hparams), dataset=dataset, metrics=METRICS)
    elif hparams.method == "MetaPath2Vec":
        USE_AMP = False
        model_hparams = {
            "embedding_dim": 128,
            "walk_length": 50,
            "context_size": 7,
            "walks_per_node": 5,
            "num_negative_samples": 5,
            "sparse": True,
            "batch_size": 128 * NUM_GPUS,
            "train_ratio": dataset.train_ratio,
            "n_classes": dataset.n_classes,
            "lr": 0.001 * NUM_GPUS,
        }
        model = MetaPath2Vec(Namespace(**model_hparams), dataset=dataset, metrics=METRICS)

    MAX_EPOCHS = 250
    wandb_logger = WandbLogger(name=model.__class__.__name__,
                               tags=[dataset.name],
                               project="multiplex-comparison")
    wandb_logger.log_hyperparams(model_hparams)

    trainer = Trainer(
        gpus=NUM_GPUS,
        distributed_backend='dp' if NUM_GPUS > 1 else None,
        #     auto_lr_find=True,
        max_epochs=MAX_EPOCHS,
        callbacks=[EarlyStopping(monitor='loss', patience=2, min_delta=0.0001),
                   EarlyStopping(monitor='val_loss', patience=5, min_delta=0.0001), ],
        logger=wandb_logger,
        #     regularizers=regularizers,
        weights_summary='top',
        use_amp=USE_AMP,
        amp_level='O1' if USE_AMP else None, precision=16 if USE_AMP else 32
    )

    trainer.fit(model)


if __name__ == "__main__":
    parser = ArgumentParser()
    # parametrize the network
    parser.add_argument('--dataset', type=str, default="ACM_HANDataset")
    parser.add_argument('--method', type=str, default="MetaPath2Vec")
    parser.add_argument('--train_ratio', type=float, default=0.7)

    # add all the available options to the trainer
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    train(args)
