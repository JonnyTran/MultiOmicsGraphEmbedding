import logging
import sys
from argparse import ArgumentParser, Namespace

logger = logging.getLogger("wandb")
logger.setLevel(logging.ERROR)

sys.path.insert(0, "../MultiOmicsGraphEmbedding/")

from pytorch_lightning.trainer import Trainer

from pytorch_lightning.callbacks import EarlyStopping

from moge.module.PyG.node_clf import MetaPath2Vec, LATTENodeClf
from moge.module.cogdl.node_clf import HGT, GTN, HAN
from pytorch_lightning.loggers import WandbLogger

from run.utils import load_node_dataset


def train(hparams):
    EMBEDDING_DIM = 128
    USE_AMP = None
    NUM_GPUS = hparams.num_gpus
    MAX_EPOCHS = 1000
    batch_order = 11

    dataset = load_node_dataset(hparams.dataset, hparams.method, hparams=hparams, train_ratio=hparams.train_ratio)

    METRICS = ["precision", "recall", "f1", "accuracy", "top_k" if dataset.multilabel else "ogbn-mag", ]

    if hparams.method == "HAN":
        USE_AMP = True
        model_hparams = {
            "embedding_dim": EMBEDDING_DIM,
            "batch_size": 2 ** batch_order * NUM_GPUS,
            "num_layers": 2,
            "collate_fn": "HAN_batch",
            "train_ratio": dataset.train_ratio,
            "loss_type": "BINARY_CROSS_ENTROPY" if dataset.multilabel else "SOFTMAX_CROSS_ENTROPY",
            "n_classes": dataset.n_classes,
            "lr": 0.0005 * NUM_GPUS,
        }
        model = HAN(Namespace(**model_hparams), dataset=dataset, metrics=METRICS)
    elif hparams.method == "GTN":
        USE_AMP = True
        model_hparams = {
            "embedding_dim": EMBEDDING_DIM,
            "num_channels": len(dataset.metapaths),
            "num_layers": 2,
            "batch_size": 2 ** batch_order * NUM_GPUS,
            "collate_fn": "HAN_batch",
            "train_ratio": dataset.train_ratio,
            "loss_type": "BINARY_CROSS_ENTROPY" if dataset.multilabel else "SOFTMAX_CROSS_ENTROPY",
            "n_classes": dataset.n_classes,
            "lr": 0.0005 * NUM_GPUS,
        }
        model = GTN(Namespace(**model_hparams), dataset=dataset, metrics=METRICS)

    elif hparams.method == "MetaPath2Vec":
        USE_AMP = True
        model_hparams = {
            "embedding_dim": EMBEDDING_DIM,
            "walk_length": 50,
            "context_size": 7,
            "walks_per_node": 5,
            "num_negative_samples": 5,
            "sparse": True,
            "batch_size": 400 * NUM_GPUS,
            "train_ratio": dataset.train_ratio,
            "n_classes": dataset.n_classes,
            "lr": 0.01 * NUM_GPUS,
        }
        model = MetaPath2Vec(Namespace(**model_hparams), dataset=dataset, metrics=METRICS)

    elif hparams.method == "HGT":
        USE_AMP = False
        model_hparams = {
            "embedding_dim": EMBEDDING_DIM,
            "num_channels": len(dataset.metapaths),
            "n_layers": 2,
            "attn_heads": 8,
            "attn_dropout": 0.2,
            "prev_norm": True,
            "last_norm": True,
            "nb_cls_dense_size": 0, "nb_cls_dropout": 0.0,
            "use_class_weights": False,
            "batch_size": 2 ** batch_order,
            "n_epoch": MAX_EPOCHS,
            "train_ratio": dataset.train_ratio,
            "loss_type": "BCE" if dataset.multilabel else "SOFTMAX_CROSS_ENTROPY",
            "n_classes": dataset.n_classes,
            "collate_fn": "collate_HGT_batch",
            "lr": 0.001,  # Not used here, defaults to 1e-3
        }
        model = HGT(Namespace(**model_hparams), dataset, metrics=METRICS)

    elif "LATTE" in hparams.method:
        USE_AMP = False
        num_gpus = 1

        if "-1" in hparams.method:
            t_order = 1
        elif "-2" in hparams.method:
            t_order = 2
        elif "-3" in hparams.method:
            t_order = 3
        else:
            t_order = 2

        model_hparams = {
            "embedding_dim": EMBEDDING_DIM,
            "n_layers": len(dataset.neighbor_sizes),
            "t_order": t_order,
            "batch_size": 2 ** batch_order * max(num_gpus, 1),
            "nb_cls_dense_size": 0,
            "nb_cls_dropout": 0.4,
            "activation": "relu",
            "attn_heads": 2,
            "attn_activation": "sharpening",
            "attn_dropout": 0.2,
            "loss_type": "BCE" if dataset.multilabel else "SOFTMAX_CROSS_ENTROPY",
            "use_proximity": True if "proximity" in hparams.method else False,
            "neg_sampling_ratio": 2.0,
            "n_classes": dataset.n_classes,
            "use_class_weights": False,
            "lr": 0.001 * num_gpus,
            "momentum": 0.9,
            "weight_decay": 1e-2,
        }

        model_hparams.update(hparams.__dict__)

        metrics = ["precision", "recall", "micro_f1",
                   "accuracy" if dataset.multilabel else "ogbn-mag", "top_k"]

        model = LATTENodeClf(Namespace(**model_hparams), dataset, collate_fn="neighbor_sampler", metrics=metrics)

    wandb_logger = WandbLogger(name=model.name(),
                               tags=[dataset.name()],
                               project="multiplex-comparison")
    wandb_logger.log_hyperparams(model_hparams)

    trainer = Trainer(
        gpus=NUM_GPUS, auto_select_gpus=True,
        distributed_backend='dp' if NUM_GPUS > 1 else None,
        max_epochs=MAX_EPOCHS,
        callbacks=[EarlyStopping(monitor='val_loss', patience=10, min_delta=0.0001, strict=False)],
        logger=wandb_logger,
        weights_summary='top',
        amp_level='O1' if USE_AMP else None,
        precision=16 if USE_AMP else 32
    )

    trainer.fit(model)

    # model.register_hooks()
    trainer.test(model)
    # wandb_logger.log_metrics(model.clustering_metrics(n_runs=10, compare_node_types=True))


if __name__ == "__main__":
    parser = ArgumentParser()
    # parametrize the network
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--run', type=int, default=0)
    parser.add_argument('--inductive', type=bool, default=True)

    parser.add_argument('--dataset', type=str, default="ACM")
    parser.add_argument('--method', type=str, default="HAN")
    parser.add_argument('--train_ratio', type=float, default=None)

    parser.add_argument('--disable_alpha', type=bool, default=False)
    parser.add_argument('--disable_beta', type=bool, default=False)
    parser.add_argument('--disable_concat', type=bool, default=False)
    parser.add_argument('--attn_activation', type=str, default=None)

    parser.add_argument('--num_gpus', type=int, default=1)

    parser.add_argument('--use_emb', type=str,
                        default="~/PycharmProjects/MultiOmicsGraphEmbedding/moge/module/dgl/NARS/")

    # add all the available options to the trainer
    args = parser.parse_args()
    train(args)
