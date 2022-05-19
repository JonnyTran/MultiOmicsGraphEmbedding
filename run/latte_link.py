import logging
import os
import random
import sys
from argparse import ArgumentParser, Namespace

import yaml

logger = logging.getLogger("wandb")
logger.setLevel(logging.ERROR)
sys.path.insert(0, "../MultiOmicsGraphEmbedding/")

from pytorch_lightning.trainer import Trainer

from pytorch_lightning.callbacks import EarlyStopping

from pytorch_lightning.loggers import WandbLogger

from moge.model.PyG.link_pred import LATTELinkPred
from run.load_data import load_link_dataset


def train(hparams):
    NUM_GPUS = hparams.num_gpus
    USE_AMP = False  # True if NUM_GPUS > 1 else False
    MAX_EPOCHS = 30

    if hparams.n_layers > 1:
        hparams.batch_size = hparams.batch_size // hparams.n_layers

    dataset = load_link_dataset(hparams.dataset, hparams=hparams, path="datasets")
    hparams.n_classes = dataset.n_classes

    model = LATTELinkPred(hparams, dataset, collate_fn="triples_batch", metrics=[hparams.dataset])
    wandb_logger = WandbLogger(name=model.name(), tags=[dataset.name()], project="multiplex-comparison")

    trainer = Trainer(
        gpus=random.sample(range(4), NUM_GPUS),
        distributed_backend='ddp' if NUM_GPUS > 1 else None,
        auto_lr_find=False,
        max_epochs=MAX_EPOCHS,
        callbacks=[EarlyStopping(monitor='val_loss', patience=5, min_delta=0.01, strict=False)],
        logger=wandb_logger,
        weights_summary='top',
        amp_level='O1' if USE_AMP else None,
        precision=16 if USE_AMP else 32
    )

    trainer.fit(model)
    trainer.test(model)


def parse_yaml(parser: ArgumentParser) -> Namespace:
    parser.add_argument('-y', '--config', help="configuration file *.yml", type=str, required=False)
    args = parser.parse_args()
    # yaml priority is higher than args
    if isinstance(args.config, str) and os.path.exists(args.config):
        opt = yaml.load(open(args.config), Loader=yaml.FullLoader)
        args_dict = args.__dict__
        args_dict.update(opt)
        args = Namespace(**args_dict)
        print("\n", args, end="\n\n\n")

    return args


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--num_gpus', type=int, default=1)
    # parametrize the network
    parser.add_argument('--dataset', type=str, default="ogbl-biokg")
    parser.add_argument('-d', '--embedding_dim', type=int, default=128)
    parser.add_argument('-t', '--n_layers', type=int, default=1)
    parser.add_argument('-b', '--batch_size', type=int, default=12000)
    parser.add_argument('--activation', type=str, default="relu")
    parser.add_argument('--attn_heads', type=int, default=1)
    parser.add_argument('--attn_activation', type=str, default="sharpening")
    parser.add_argument('--attn_dropout', type=float, default=0.5)

    parser.add_argument('--n_neighbors_1', type=int, default=50)
    parser.add_argument('--use_proximity', type=bool, default=False)
    parser.add_argument('--neg_sampling_ratio', type=float, default=64.0)

    parser.add_argument('--use_reverse', type=bool, default=False)

    parser.add_argument('--loss_type', type=str, default="KL_DIVERGENCE")
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-2)

    args = parse_yaml(parser)

    train(args)
