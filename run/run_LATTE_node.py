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
from cogdl.datasets.han_data import ACM_HANDataset, DBLP_HANDataset, IMDB_HANDataset
from cogdl.datasets.gtn_data import ACM_GTNDataset, DBLP_GTNDataset, IMDB_GTNDataset
from torch_geometric.datasets import AMiner

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping

from moge.generator import HeteroNeighborSampler, TripletSampler
from moge.methods.node_clf import LATTENodeClassifier
from moge.methods.link_pred import LATTELinkPredictor
from .utils import load_node_dataset


def train(hparams: Namespace):
    NUM_GPUS = hparams.num_gpus
    USE_AMP = False  # True if NUM_GPUS > 1 else False
    MAX_EPOCHS = 250

    if hparams.t_order > 1:
        hparams.n_neighbors_2 = max(1, 153600 // (hparams.n_neighbors_1 * hparams.batch_size))
        neighbor_sizes = [hparams.n_neighbors_1, hparams.n_neighbors_2]
    else:
        neighbor_sizes = [hparams.n_neighbors_1]
        hparams.batch_size = int(2 * hparams.batch_size)
        hparams.neighbor_sizes = neighbor_sizes

    if hparams.embedding_dim > 128:
        hparams.batch_size = hparams.batch_size // 2

    dataset = load_node_dataset(hparams.dataset, method=LATTENodeClassifier.name(), train_ratio=None, hparams=hparams)

    METRICS = ["precision", "recall", "f1", "accuracy" if dataset.multilabel else hparams.dataset, "top_k"]
    hparams.loss_type = "BCE" if dataset.multilabel else "SOFTMAX_CROSS_ENTROPY"
    hparams.n_classes = dataset.n_classes
    model = LATTENodeClassifier(hparams, dataset, collate_fn="neighbor_sampler", metrics=METRICS)

    logger = WandbLogger(name=model.name(),
                         tags=[dataset.name()],
                         project="multiplex-comparison")

    trainer = Trainer(
        gpus=NUM_GPUS,
        distributed_backend='ddp' if NUM_GPUS > 1 else None,
        # auto_lr_find=True,
        max_epochs=MAX_EPOCHS,
        early_stop_callback=EarlyStopping(monitor='val_loss', patience=5, min_delta=0.001, strict=False),
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
    parser.add_argument('--num_gpus', type=int, default=1)
    # parametrize the network
    parser.add_argument('--dataset', type=str, default="ogbn-mag")
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--t_order', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=2000)
    parser.add_argument('--n_neighbors_1', type=int, default=20)
    parser.add_argument('--activation', type=str, default="relu")
    parser.add_argument('--attn_heads', type=int, default=64)
    parser.add_argument('--attn_activation', type=str, default="sharpening")
    parser.add_argument('--attn_dropout', type=float, default=0.2)

    parser.add_argument('--nb_cls_dense_size', type=int, default=0)
    parser.add_argument('--nb_cls_dropout', type=float, default=0.3)

    parser.add_argument('--use_proximity_loss', type=bool, default=True)
    parser.add_argument('--neg_sampling_ratio', type=float, default=5.0)
    parser.add_argument('--use_class_weights', type=bool, default=True)
    parser.add_argument('--use_reverse', type=bool, default=True)

    parser.add_argument('--loss_type', type=str, default="SOFTMAX_CROSS_ENTROPY")
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-5)

    # add all the available options to the trainer
    # parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    train(args)
