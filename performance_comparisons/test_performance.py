import logging
import sys
from argparse import ArgumentParser, Namespace

logger = logging.getLogger("wandb")
logger.setLevel(logging.ERROR)

sys.path.insert(0, "../MultiOmicsGraphEmbedding/")

import pytorch_lightning as pl
from pytorch_lightning.trainer import Trainer
from torch_geometric.datasets import AMiner

from pytorch_lightning.callbacks import EarlyStopping

from cogdl.datasets.han_data import ACM_HANDataset, DBLP_HANDataset, IMDB_HANDataset
from cogdl.datasets.gtn_data import ACM_GTNDataset, DBLP_GTNDataset, IMDB_GTNDataset

from moge.methods.node_clf import MetaPath2Vec, HAN, GTN, LATTENodeClassifier
from moge.generator import HeteroNetDataset, HeteroNeighborSampler
from pytorch_lightning.loggers import WandbLogger


def train(hparams):
    EMBEDDING_DIM = 128
    NUM_GPUS = 1
    METRICS = ["precision", "recall", "f1", "top_k" if hparams.dataset.multilabel else "ogbn-mag", ]

    if hparams.dataset == "ACM":
        if hparams.method == "HAN":
            dataset = HeteroNetDataset(ACM_HANDataset(), node_types=["P"], metapaths=["PAP", "PLP"],
                                       train_ratio=hparams.train_ratio)
        else:
            dataset = HeteroNeighborSampler(ACM_GTNDataset(), node_types=["P"], metapaths=["PAP", "PLP"],
                                            head_node_type="P",
                                            train_ratio=hparams.train_ratio)

    elif hparams.dataset == "DBLP":
        if hparams.method == "HAN":
            dataset = HeteroNetDataset(DBLP_HANDataset(), node_types=["A"], metapaths=["APA", "ACA", "ATA"],
                                       train_ratio=hparams.train_ratio)
        else:
            dataset = HeteroNeighborSampler(DBLP_GTNDataset(), node_types=["A"], metapaths=["APA", "ACA", "ATA", "AGA"],
                                            head_node_type="A",
                                            train_ratio=hparams.train_ratio)

    elif hparams.dataset == "IMDB":
        if hparams.method == "HAN":
            dataset = HeteroNetDataset(IMDB_HANDataset(), node_types=["M"], metapaths=["MAM", "MDM", "MYM"],
                                       train_ratio=hparams.train_ratio)
        else:
            dataset = HeteroNeighborSampler(IMDB_GTNDataset(), node_types=["M"], metapaths=["MAM", "MDM", "MYM"],
                                            head_node_type="M",
                                            train_ratio=hparams.train_ratio)
    elif hparams.dataset == "AMiner":
        dataset = HeteroNeighborSampler(AMiner("datasets/aminer"), node_types=None,
                                        metapaths=[('paper', 'written by', 'author'),
                                                   ('venue', 'published', 'paper')],
                                        head_node_type="author",
                                        train_ratio=hparams.train_ratio)
    elif hparams.dataset == "BlogCatalog":
        dataset = HeteroNeighborSampler("dataset/blogcatalog6k.mat", node_types=["user", "tag"], head_node_type="user",
                                        train_ratio=hparams.train_ratio)
        dataset.name = "BlogCatalog3"

    if hparams.method == "HAN":
        USE_AMP = False
        model_hparams = {
            "embedding_dim": EMBEDDING_DIM,
            "batch_size": 512 * NUM_GPUS,
            "collate_fn": "HAN_batch",
            "val_collate_fn": "HAN_batch",
            "train_ratio": dataset.train_ratio,
            "loss_type": "BINARY_CROSS_ENTROPY" if dataset.multilabel else "SOFTMAX_CROSS_ENTROPY",
            "n_classes": dataset.n_classes,
            "lr": 0.001 * NUM_GPUS,
        }
        model = HAN(Namespace(**model_hparams), dataset=dataset, metrics=METRICS)
    elif hparams.method == "GTN":
        USE_AMP = True
        model_hparams = {
            "embedding_dim": EMBEDDING_DIM,
            "num_channels": len(dataset.metapaths),
            "batch_size": 512 * NUM_GPUS,
            "collate_fn": "HAN_batch",
            "val_collate_fn": "HAN_batch",
            "train_ratio": dataset.train_ratio,
            "loss_type": "BINARY_CROSS_ENTROPY" if dataset.multilabel else "SOFTMAX_CROSS_ENTROPY",
            "n_classes": dataset.n_classes,
            "lr": 0.0005 * NUM_GPUS,
        }
        model = GTN(Namespace(**model_hparams), dataset=dataset, metrics=METRICS)
    elif hparams.method == "MetaPath2Vec":
        USE_AMP = False
        model_hparams = {
            "embedding_dim": EMBEDDING_DIM,
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
    elif hparams.method == "LATTE":
        num_gpus = 1
        batch_order = 11
        model_hparams = {
            "embedding_dim": 128,
            "t_order": 2,
            "batch_size": 2 ** batch_order * max(num_gpus, 1),
            "nb_cls_dense_size": 0,
            "nb_cls_dropout": 0.3,
            "activation": "relu",
            "attn_heads": 64,
            "attn_activation": "LeakyReLU",
            "attn_dropout": 0.2,
            "loss_type": "BCE" if dataset.multilabel else "SOFTMAX_CROSS_ENTROPY",
            "use_proximity_loss": False,
            "neg_sampling_ratio": 2.0,
            "n_classes": dataset.n_classes,
            "use_class_weights": False,
            "lr": 0.001 * num_gpus,
            "momentum": 0.9,
            "weight_decay": 1e-5,
        }

        metrics = ["precision", "recall", "f1",
                   "accuracy" if dataset.multilabel else "ogbn-mag", "top_k"]

        model = LATTENodeClassifier(Namespace(**model_hparams), dataset, collate_fn="neighbor_sampler", metrics=metrics)

    MAX_EPOCHS = 250
    wandb_logger = WandbLogger(name=model.name(),
                               tags=[dataset.name()],
                               project="multiplex-comparison")
    wandb_logger.log_hyperparams(model_hparams)

    trainer = Trainer(
        gpus=NUM_GPUS,
        distributed_backend='dp' if NUM_GPUS > 1 else None,
        max_epochs=MAX_EPOCHS,
        callbacks=[EarlyStopping(monitor='loss', patience=2, min_delta=0.0001, strict=False),
                   EarlyStopping(monitor='val_loss', patience=5, min_delta=0.0001, strict=False)],
        logger=wandb_logger,
        weights_summary='top',
        use_amp=USE_AMP,
        amp_level='O1' if USE_AMP else None,
        precision=16 if USE_AMP else 32
    )

    trainer.fit(model)


if __name__ == "__main__":
    parser = ArgumentParser()
    # parametrize the network
    parser.add_argument('--embedding_dim', type=int, default=128)

    parser.add_argument('--dataset', type=str, default="ACM")
    parser.add_argument('--method', type=str, default="MetaPath2Vec")
    parser.add_argument('--train_ratio', type=float, default=None)

    # add all the available options to the trainer
    args = parser.parse_args()
    train(args)
