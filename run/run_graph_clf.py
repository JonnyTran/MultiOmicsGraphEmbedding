import logging
import sys
from argparse import ArgumentParser, Namespace

logger = logging.getLogger("wandb")
logger.setLevel(logging.ERROR)

sys.path.insert(0, "../MultiOmicsGraphEmbedding/")

from pytorch_lightning.trainer import Trainer

from pytorch_lightning.callbacks import EarlyStopping

from moge.model.dgl.graph_clf import LATTEGraphClassifier
from pytorch_lightning.loggers import WandbLogger

from run.utils import load_graph_dataset

def train(hparams):
    EMBEDDING_DIM = 128
    USE_AMP = None
    NUM_GPUS = hparams.num_gpus
    MAX_EPOCHS = 1000
    batch_order = 11

    dataset = load_graph_dataset(hparams.dataset, hparams=hparams, )
    METRICS = ["precision", "recall", "micro_f1", "macro_f1",
               dataset.name() if "ogb" in dataset.name() else "accuracy"]

    if "LATTE" in hparams.method:
        USE_AMP = False

        model_hparams = {
            "embedding_dim": EMBEDDING_DIM,
            "n_layers": 1,
            "batchnorm": True,
            "readout": "sum",
            "activation": "relu",

            "batch_size": 400,
            "attn_heads": 4,
            "attn_activation": "LeakyReLU",
            "attn_dropout": 0.5,

            "nb_cls_dense_size": 0,
            "nb_cls_dropout": 0.3,

            "use_proximity": False,
            "neg_sampling_ratio": 5.0,

            "n_classes": dataset.n_classes,
            "use_class_weights": True,

            "loss_type": "BCE" if dataset.multilabel else "SOFTMAX_CROSS_ENTROPY",
            "lr": 0.01,
            "momentum": 0.9,
            "weight_decay": 1e-4,
        }

        model_hparams.update(hparams.__dict__)

        model = LATTEGraphClassifier(Namespace(**model_hparams), dataset, collate_fn="neighbor_sampler",
                                     metrics=METRICS)

    wandb_logger = WandbLogger(name=model.name(),
                               tags=[dataset.name()],
                               project="multiplex-comparison")
    wandb_logger.log_hyperparams(model_hparams)

    trainer = Trainer(
        gpus=NUM_GPUS,
        auto_select_gpus=True,
        distributed_backend='dp' if NUM_GPUS > 1 else None,
        max_epochs=MAX_EPOCHS,
        callbacks=[EarlyStopping(monitor='val_loss', patience=10, min_delta=0.0001, strict=False)],
        logger=wandb_logger,
        weights_summary='top',
        amp_level='O1' if USE_AMP else None,
        precision=16 if USE_AMP else 32
    )

    trainer.fit(model)
    trainer.test(model)


if __name__ == "__main__":
    parser = ArgumentParser()
    # parametrize the network
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--run', type=int, default=0)
    parser.add_argument('--inductive', type=bool, default=True)

    parser.add_argument('--dataset', type=str, default="ogbg-ppa")
    parser.add_argument('--method', type=str, default="LATTE")
    parser.add_argument('--train_ratio', type=float, default=None)

    parser.add_argument('--disable_alpha', type=bool, default=False)
    parser.add_argument('--disable_beta', type=bool, default=False)
    parser.add_argument('--disable_concat', type=bool, default=False)
    parser.add_argument('--attn_activation', type=str, default=None)

    parser.add_argument('--num_gpus', type=int, default=1)

    # add all the available options to the trainer
    args = parser.parse_args()
    train(args)
