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
        dataset = HeterogeneousNetworkDataset(ACM_HANDataset(), node_types=["P"], metapath=["PAP", "PLP"])
    elif hparams.dataset == "DBLP":
        dataset = HeterogeneousNetworkDataset(DBLP_HANDataset(), node_types=["A"], metapath=["APA", "ACA", "ATA"])
    elif hparams.dataset == "IMDB":
        dataset = HeterogeneousNetworkDataset(IMDB_HANDataset(), node_types=["M"], metapath=["MAM", "MDM", "MYM"])
    elif hparams.dataset == "AMiner":
        dataset = HeterogeneousNetworkDataset(AMiner("datasets/aminer"), node_types=None, head_node_type="author")
    elif hparams.dataset == "BlogCatalog":
        dataset = HeterogeneousNetworkDataset("/home/jonny/Downloads/blogcatalog6k.mat", node_types=["user", "tag"])
        dataset.name = "BlogCatalog3"

    num_gpus = 1
    metrics = ["accuracy", "precision", "recall"]

    if hparams.method == "HAN":
        model_hparams = {
            "embedding_dim": 32,
            "batch_size": 64 * num_gpus,
            "train_ratio": dataset.train_ratio,
            "loss_type": "SOFTMAX_CROSS_ENTROPY",
            "n_classes": dataset.n_classes,
            "lr": 0.001 * num_gpus,
        }
        model = HAN(Namespace(**model_hparams), dataset, metrics=metrics)
    elif hparams.method == "GTN":
        model_hparams = {
            "embedding_dim": 128,
            "num_channels": 1,
            "batch_size": 128 * num_gpus,
            "train_ratio": dataset.train_ratio,
            "loss_type": "SOFTMAX_CROSS_ENTROPY",
            "n_classes": dataset.n_classes,
            "lr": 0.001 * num_gpus,
        }
        model = GTN(Namespace(**model_hparams), dataset, metrics=metrics)
    elif hparams.method == "MetaPath2Vec":
        model_hparams = {
            "embedding_dim": 128,
            "walk_length": 50,
            "context_size": 7,
            "walks_per_node": 5,
            "num_negative_samples": 5,
            "sparse": True,
            "batch_size": 512 * num_gpus,
            "train_ratio": dataset.train_ratio,
            "n_classes": dataset.n_classes,
            "lr": 0.01 * num_gpus,
        }
        model = MetaPath2Vec(Namespace(**model_hparams), dataset, metrics=metrics)

    max_epochs = 250
    wandb_logger = WandbLogger(name=model.__class__.__name__,
                               tags=[dataset.name],
                               project="multiplex-comparison")
    wandb_logger.log_hyperparams(hparams)

    trainer = Trainer(
        gpus=num_gpus,
        #     distributed_backend='dp',
        #     auto_lr_find=True,
        max_epochs=max_epochs,
        callbacks=[EarlyStopping(monitor='loss', patience=2, min_delta=0.0001),
                   EarlyStopping(monitor='val_loss', patience=5, min_delta=0.0001), ],
        logger=wandb_logger,
        #     regularizers=regularizers,
        weights_summary='top',
        amp_level='O1', precision=16
    )

    trainer.fit(model)


if __name__ == "__main__":
    parser = ArgumentParser()
    # parametrize the network
    parser.add_argument('--dataset', type=str, default="ACM_HANDataset")
    parser.add_argument('--method', type=str, default="MetaPath2Vec")

    # add all the available options to the trainer
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    train(args)
