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
from moge.generator import HeteroNeighborSampler, LinkSampler
from pytorch_lightning.loggers import WandbLogger

from moge.methods.node_classification import LATTENodeClassifier


def train(hparams):
    NUM_GPUS = 4
    USE_AMP = True if NUM_GPUS > 1 else False
    MAX_EPOCHS = 50

    if hparams.t_order > 1:
        hparams.n_neighbors_2 = max(1, 153600 // (hparams.n_neighbors_1 * hparams.batch_size))
        neighbor_sizes = [hparams.n_neighbors_1, hparams.n_neighbors_2]
    else:
        neighbor_sizes = [hparams.n_neighbors_1]
        hparams.batch_size = int(2 * hparams.batch_size)

    if hparams.embedding_dim > 128:
        hparams.batch_size = hparams.batch_size // 2

    mag = PygNodePropPredDataset(name="ogbn-mag", root="datasets")
    dataset = HeteroNeighborSampler(mag, directed=True, neighbor_sizes=neighbor_sizes,
                                    node_types=['paper', 'author', 'field_of_study', 'institution'],
                                    head_node_type="paper",
                                    add_reverse_metapaths=True)

    hparams.loss_type = "BCE" if dataset.multilabel else "SOFTMAX_CROSS_ENTROPY"
    hparams.n_classes = dataset.n_classes
    METRICS = ["accuracy" if dataset.multilabel else "ogbn-mag", "top_k"]

    model = LATTENodeClassifier(hparams, dataset, collate_fn="neighbor_sampler",
                                metrics=METRICS)
    logger = WandbLogger()

    trainer = Trainer(
        gpus=NUM_GPUS,
        distributed_backend='ddp' if NUM_GPUS > 1 else None,
        # auto_lr_find=True,
        max_epochs=MAX_EPOCHS,
        early_stop_callback=EarlyStopping(monitor='val_loss', patience=3, min_delta=0.001, strict=False),
        logger=logger,
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
    # parametrize the network
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--t_order', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--n_neighbors_1', type=int, default=30)
    parser.add_argument('--activation', type=str, default="tanh")
    parser.add_argument('--attn_activation', type=str, default="LeakyReLU")
    parser.add_argument('--attn_dropout', type=float, default=0.2)

    parser.add_argument('--nb_cls_dense_size', type=int, default=1024)
    parser.add_argument('--nb_cls_dropout', type=float, default=0.2)

    parser.add_argument('--use_proximity_loss', type=bool, default=False)
    parser.add_argument('--neg_sampling_ratio', type=float, default=2.0)
    parser.add_argument('--use_class_weights', type=bool, default=False)

    parser.add_argument('--loss_type', type=str, default="SOFTMAX_CROSS_ENTROPY")
    parser.add_argument('--lr', type=float, default=0.01)

    # add all the available options to the trainer
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    train(args)
