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
from moge.generator import HeteroNeighborSampler, LinkSampler
from pytorch_lightning.loggers import WandbLogger

from moge.methods.node_classification import LATTENodeClassifier
from moge.methods.link_prediction import LATTELinkPredictor


def train(hparams):
    NUM_GPUS = 4
    USE_AMP = True if NUM_GPUS > 1 else False
    MAX_EPOCHS = 50

    if hparams.embedding_dim > 128:
        hparams.batch_size = hparams.batch_size // 2

    if "ogbl" in hparams.dataset:
        ogbl = PygLinkPropPredDataset(name=hparams.dataset, root="datasets")
        dataset = LinkSampler(ogbl, directed=True,
                              node_types=list(ogbl[0].num_nodes_dict.keys()) if hasattr(ogbl[0],
                                                                                        "num_nodes_dict") else None,
                              head_node_type=None,
                              add_reverse_metapaths=hparams.use_reverse)
        hparams.n_classes = dataset.n_classes
        METRICS = [hparams.dataset]

        model = LATTELinkPredictor(hparams, dataset, collate_fn="triples_batch", metrics=METRICS)
    else:
        raise Exception(f"Dataset `{hparams.dataset}` not found")

    # logger = WandbLogger()
    wandb_logger = WandbLogger(name=model.name(),
                               tags=[dataset.name()],
                               project="multiplex-comparison")

    trainer = Trainer(
        gpus=NUM_GPUS,
        distributed_backend='ddp' if NUM_GPUS > 1 else None,
        auto_lr_find=False,
        max_epochs=MAX_EPOCHS,
        early_stop_callback=EarlyStopping(monitor='val_loss', patience=10, min_delta=0.001, strict=False),
        logger=wandb_logger,
        # regularizers=regularizers,
        weights_summary='top',
        use_amp=USE_AMP,
        amp_level='O1' if USE_AMP else None,
        precision=16 if USE_AMP else 32
    )

    trainer.fit(model)
    trainer.test(model)
    return None

if __name__ == "__main__":
    parser = ArgumentParser()
    # parametrize the network
    parser.add_argument('--dataset', type=str, default="ogbl-biokg")
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--t_order', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=524288)
    parser.add_argument('--n_neighbors_1', type=int, default=30)
    parser.add_argument('--activation', type=str, default="tanh")
    parser.add_argument('--attn_activation', type=str, default="LeakyReLU")
    parser.add_argument('--attn_dropout', type=float, default=0.2)

    parser.add_argument('--nb_cls_dense_size', type=int, default=0)
    parser.add_argument('--nb_cls_dropout', type=float, default=0.2)

    parser.add_argument('--use_proximity_loss', type=bool, default=True)
    parser.add_argument('--neg_sampling_ratio', type=float, default=5.0)
    parser.add_argument('--neg_sampling_test_size', type=int, default=10)

    parser.add_argument('--use_class_weights', type=bool, default=False)
    parser.add_argument('--use_reverse', type=bool, default=True)

    parser.add_argument('--loss_type', type=str, default="KL_DIVERGENCE")
    parser.add_argument('--lr', type=float, default=0.01)

    # add all the available options to the trainer
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    train(args)
