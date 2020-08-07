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
    NUM_GPUS = 1
    MAX_EPOCHS = 100

    mag = PygNodePropPredDataset(name="ogbn-mag", root="datasets")

    if hparams.t_order > 1:
        hparams.n_neighbors_2 = int(153600 / (hparams.n_neighbors_1 * hparams.batch_size))
        neighbor_sizes = [hparams.n_neighbors_1, hparams.n_neighbors_2]
    else:
        neighbor_sizes = [hparams.n_neighbors_1]
    dataset = HeteroNeighborSampler(mag, directed=True, neighbor_sizes=neighbor_sizes,
                                    node_types=['paper', 'author', 'field_of_study', 'institution'],
                                    head_node_type="paper",
                                    add_reverse_metapaths=True)

    METRICS = ["precision", "recall", "accuracy" if dataset.multilabel else "ogbn-mag", "top_k"]
    hparams.loss_type = "BCE" if dataset.multilabel else "SOFTMAX_CROSS_ENTROPY"
    hparams.n_classes = dataset.n_classes
    model = LATTENodeClassifier(hparams, dataset, collate_fn="neighbor_sampler",
                                metrics=METRICS)

    wandb_logger = WandbLogger(name=model.name(),
                               tags=[dataset.name()],
                               project="multiplex-comparison")
    wandb_logger.log_hyperparams(hparams)

    trainer = Trainer(
        gpus=NUM_GPUS,
        distributed_backend='dp' if NUM_GPUS > 1 else None,
        # auto_lr_find=True,
        max_epochs=MAX_EPOCHS,
        early_stop_callback=EarlyStopping(monitor='val_loss', patience=4, min_delta=0.001),
        # callbacks=[EarlyStopping(monitor='loss', patience=1, min_delta=0.0001),
        #            EarlyStopping(monitor='val_loss', patience=2, min_delta=0.0001), ],
        logger=wandb_logger,
        # regularizers=regularizers,
        weights_summary='top',
        # use_amp=USE_AMP,
        # amp_level='O1' if USE_AMP else None,
        # precision=16 if USE_AMP else 32
    )

    trainer.fit(model)


if __name__ == "__main__":
    parser = ArgumentParser()
    # parametrize the network
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--t_order', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=2024)
    parser.add_argument('--n_neighbors_1', type=int, default=2024)
    parser.add_argument('--activation', type=str, default="sigmoid")

    parser.add_argument('--nb_cls_dense_size', type=int, default=0)
    parser.add_argument('--nb_cls_dropout', type=float, default=0.2)

    parser.add_argument('--use_proximity_loss', type=bool, default=False)
    parser.add_argument('--neg_sampling_ratio', type=float, default=2.0)
    parser.add_argument('--use_class_weights', type=bool, default=False)

    parser.add_argument('--lr', type=float, default=0.01)

    # add all the available options to the trainer
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    train(args)
