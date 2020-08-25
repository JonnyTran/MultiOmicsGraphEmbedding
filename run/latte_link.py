import logging
import sys
from argparse import ArgumentParser, Namespace

logger = logging.getLogger("wandb")
logger.setLevel(logging.ERROR)
sys.path.insert(0, "../MultiOmicsGraphEmbedding/")

import pytorch_lightning as pl
from pytorch_lightning.trainer import Trainer

from pytorch_lightning.callbacks import EarlyStopping

from ogb.nodeproppred import PygNodePropPredDataset
from ogb.linkproppred import PygLinkPropPredDataset
from moge.generator import HeteroNeighborSampler, TripletSampler, EdgeSampler
from pytorch_lightning.loggers import WandbLogger

from moge.methods.node_clf import LATTENodeClassifier
from moge.methods.link_pred import LATTELinkPredictor
from run.utils import load_link_dataset

def train(hparams):
    NUM_GPUS = hparams.num_gpus
    USE_AMP = False  # True if NUM_GPUS > 1 else False
    MAX_EPOCHS = 50

    dataset = load_link_dataset(hparams.dataset, hparams=hparams)
    hparams.n_classes = dataset.n_classes

    model = LATTELinkPredictor(hparams, dataset, collate_fn="triples_batch", metrics=[hparams.dataset])
    wandb_logger = WandbLogger(name=model.name(), tags=[dataset.name()], project="multiplex-comparison")

    trainer = Trainer(
        gpus=NUM_GPUS,
        distributed_backend='ddp' if NUM_GPUS > 1 else None,
        auto_lr_find=False,
        max_epochs=MAX_EPOCHS,
        early_stop_callback=EarlyStopping(monitor='val_loss', patience=10, min_delta=0.01, strict=False),
        logger=wandb_logger,
        # regularizers=regularizers,
        weights_summary='top',
        use_amp=USE_AMP,
        amp_level='O1' if USE_AMP else None,
        precision=16 if USE_AMP else 32
    )

    trainer.fit(model)
    trainer.test(model)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--num_gpus', type=int, default=4)
    # parametrize the network
    parser.add_argument('--dataset', type=str, default="ogbl-biokg")
    parser.add_argument('-d', '--embedding_dim', type=int, default=128)
    parser.add_argument('-t', '--t_order', type=int, default=1)
    parser.add_argument('-b', '--batch_size', type=int, default=32000)
    parser.add_argument('--n_neighbors_1', type=int, default=30, help="Not used - only for compatibility")
    parser.add_argument('--activation', type=str, default="relu")
    parser.add_argument('--attn_heads', type=int, default=32)
    parser.add_argument('--attn_activation', type=str, default="sharpening")
    parser.add_argument('--attn_dropout', type=float, default=0.2)

    parser.add_argument('--nb_cls_dense_size', type=int, default=0)
    parser.add_argument('--nb_cls_dropout', type=float, default=0.2)

    parser.add_argument('--use_proximity_loss', type=bool, default=True)
    parser.add_argument('--neg_sampling_ratio', type=float, default=10.0)

    parser.add_argument('--use_class_weights', action='store_true')
    parser.add_argument('--use_reverse', action='store_true')

    parser.add_argument('--loss_type', type=str, default="KL_DIVERGENCE")
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-5)

    # add all the available options to the trainer
    # parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    train(args)
