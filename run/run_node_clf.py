import logging
import sys
from argparse import ArgumentParser, Namespace
import random


logger = logging.getLogger("wandb")
logger.setLevel(logging.ERROR)

sys.path.insert(0, "../MultiOmicsGraphEmbedding/")

from pytorch_lightning.trainer import Trainer

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from moge.module.PyG.node_clf import MetaPath2Vec, LATTENodeClf
from moge.module.cogdl.node_clf import GTN
from moge.module.dgl.node_clf import HAN, HGT, NARS, HGConv, R_HGNN
from moge.data.dgl.node_generator import DGLNodeSampler

from pytorch_lightning.loggers import WandbLogger

from run.load_data import load_node_dataset


def train(hparams):
    USE_AMP = False
    CALLBACKS = None
    NUM_GPUS = hparams.num_gpus

    dataset = load_node_dataset(hparams.dataset, hparams.method, hparams=hparams, train_ratio=hparams.train_ratio,
                                dataset_path=hparams.root_path)

    METRICS = ["micro_f1", "macro_f1", dataset.name() if "ogb" in dataset.name() else "accuracy"]

    if hparams.method == "HAN":
        args = {
            'num_neighbors': 20,
            'hidden_units': 32,
            'num_heads': [8],
            'dropout': 0.6,
            'head_node_type': dataset.head_node_type,
            'batch_size': 5000,
            'epochs': 1000,
            'patience': 10,
            'loss_type': "BCE_WITH_LOGITS" if dataset.multilabel else "SOFTMAX_CROSS_ENTROPY",
            'lr': 0.001,
            'weight_decay': 0.001,
        }

        model = HAN(args, dataset, metrics=METRICS)

    elif hparams.method == "GTN":
        USE_AMP = True
        args = {
            "embedding_dim": 128,
            "num_channels": len(dataset.metapaths),
            "num_layers": 2,
            "batch_size": 2 ** 11 * NUM_GPUS,
            "collate_fn": "HAN_batch",
            "train_ratio": dataset.train_ratio,
            "loss_type": "BCE" if dataset.multilabel else "SOFTMAX_CROSS_ENTROPY",
            "n_classes": dataset.n_classes,
            "lr": 0.005 * NUM_GPUS,
            "epochs": 40,
        }
        model = GTN(Namespace(**args), dataset=dataset, metrics=METRICS)

    elif hparams.method == "MetaPath2Vec":
        USE_AMP = True
        args = {
            "embedding_dim": 128,
            "walk_length": 50,
            "context_size": 7,
            "walks_per_node": 5,
            "num_negative_samples": 5,
            "sparse": True,
            "batch_size": 400 * NUM_GPUS,
            "train_ratio": dataset.train_ratio,
            "n_classes": dataset.n_classes,
            "lr": 0.01 * NUM_GPUS,
            "epochs": 100
        }
        model = MetaPath2Vec(Namespace(**args), dataset=dataset, metrics=METRICS)

    elif hparams.method == "HGT":
        args = {
            "embedding_dim": 128,
            "fanouts": [10, 10],
            "batch_size": 2 ** 11,
            "activation": "relu",
            "attn_heads": 4,
            "attn_activation": "sharpening",
            "attn_dropout": 0.2,
            "nb_cls_dense_size": 0,
            "nb_cls_dropout": 0.2,
            "loss_type": "BCE" if dataset.multilabel else "SOFTMAX_CROSS_ENTROPY",
            "n_classes": dataset.n_classes,
            "use_norm": True,
            "use_class_weights": False,
            "lr": 0.001,
            "momentum": 0.9,
            "weight_decay": 1e-2,
            'epochs': 100,
        }
        model = HGT(Namespace(**args), dataset, metrics=METRICS)

    elif hparams.method == "NARS":
        args = {
            'R': 2,
            'ff_layer': 2,
            'num_subsets': 8,
            'num_hidden': 256,
            #     'use_relation_subsets': "../MultiOmicsGraphEmbedding/moge/module/dgl/NARS/sample_relation_subsets/examples/mag",
            'input_dropout': True,
            'dropout': 0.5,
            'head_node_type': dataset.head_node_type,
            'batch_size': 50000,
            'epochs': 1000,
            'patience': 10,
            'lr': 0.001,
            'weight_decay': 0.0,
        }
        model = NARS(Namespace(**args), dataset, metrics=METRICS)

    elif hparams.method == "HGConv":
        args = {
            'seed': hparams.run,
            'cuda': 0,
            'num_heads': 8,  # Number of attention heads
            'hidden_units': 32,
            'dropout': 0.5,
            'n_layers': 2,
            'batch_size': 3000,  # the number of graphs to train in each batch
            'node_neighbors_min_num': 10,  # number of sampled edges for each type for each GNN layer
            'optimizer': 'adam',
            'weight_decay': 0.0,
            'residual': True,
            'epochs': 200,
            'patience': 50,
            'learning_rate': 0.001,
            'loss_type': "BCE_WITH_LOGITS" if dataset.multilabel else "SOFTMAX_CROSS_ENTROPY",
        }
        model = HGConv(args, dataset, metrics=METRICS)

    elif hparams.method == "R_HGNN":
        args = {
            'model_name': 'R_HGNN_lr0.001_dropout0.5_seed_0',
            "head_node_type": dataset.head_node_type,
            'seed': hparams.run,
            'learning_rate': 0.001,
            'num_heads': 8,
            'hidden_units': 64,
            'relation_hidden_units': 8,
            'dropout': 0.5,
            'n_layers': 2,
            'residual': True,
            'batch_size': 1280,  # the number of nodes to train in each batch
            'node_neighbors_min_num': 10,  # number of sampled edges for each type for each GNN layer
            'optimizer': 'adam',
            'weight_decay': 0.0,
            'epochs': 200,
            'patience': 50,
            'loss_type': "BCE_WITH_LOGITS" if dataset.multilabel else "SOFTMAX_CROSS_ENTROPY",
        }
        model = R_HGNN(args, dataset, metrics=METRICS)

    elif "LATTE" in hparams.method:
        USE_AMP = True

        if "-1" in hparams.method:
            t_order = 1
            batch_order = 12
        elif "-2" in hparams.method:
            t_order = 2
            batch_order = 11
        elif "-3" in hparams.method:
            t_order = 3
            batch_order = 10

            dataset.neighbor_sizes = [10, 10, 10]
            if isinstance(dataset, DGLNodeSampler):
                dataset.neighbor_sampler.fanouts = [10, 10, 10]
                dataset.neighbor_sampler.num_layers = len(dataset.neighbor_sizes)
        else:
            t_order = 2

        args = {
            "embedding_dim": 128,
            "layer_pooling": "concat",

            "n_layers": len(dataset.neighbor_sizes),
            "t_order": t_order,
            "batch_size": int(2 ** batch_order),

            "attn_heads": 4,
            "attn_activation": "LeakyReLU",
            "attn_dropout": 0.3,

            "batchnorm": False,
            "layernorm": False,
            "activation": "relu",
            "dropout": 0.5,
            "input_dropout": True,

            "nb_cls_dense_size": 0,
            "nb_cls_dropout": 0.0,

            "edge_threshold": 0.0,
            "edge_sampling": False,

            "head_node_type": dataset.head_node_type,

            "n_classes": dataset.n_classes,
            "use_class_weights": False,
            "loss_type": "BCE_WITH_LOGITS" if dataset.multilabel else "SOFTMAX_CROSS_ENTROPY",
            "lr": 0.01,
            "epochs": 50,
            "patience": 10,
            "weight_decay": 0.0,
        }

        args.update(hparams.__dict__)
        model = LATTENodeClf(Namespace(**args), dataset, collate_fn="neighbor_sampler", metrics=METRICS)
        CALLBACKS = [EarlyStopping(monitor='val_moving_loss', patience=10, min_delta=0.0001, strict=False),
                     ModelCheckpoint(monitor='val_micro_f1', mode="max",
                                     filename=model.name() + '-' + dataset.name() + '-{epoch:02d}-{val_micro_f1:.3f}')]

    if CALLBACKS is None and "patience" in args:
        CALLBACKS = [EarlyStopping(monitor='val_loss', patience=10, min_delta=0.0001, strict=False)]

    wandb_logger = WandbLogger(name=model.name(), tags=[dataset.name()], project="ogb_nodepred")
    wandb_logger.log_hyperparams(args)

    trainer = Trainer(
        gpus=random.sample([0, 1, 2], NUM_GPUS),
        auto_select_gpus=True,
        distributed_backend='ddp' if NUM_GPUS > 1 else None,
        max_epochs=args["epochs"],
        callbacks=CALLBACKS,
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

    parser.add_argument('--method', type=str, default="HAN")

    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--run', type=int, default=0)
    parser.add_argument('--inductive', type=bool, default=True)

    parser.add_argument('--dataset', type=str, default="ACM")
    parser.add_argument('--use_emb', type=str,
                        default="/home/jonny/PycharmProjects/MultiOmicsGraphEmbedding/moge/module/dgl/NARS/")
    parser.add_argument('--root_path', type=str,
                        default="/home/jonny/Bioinformatics_ExternalData/OGB/")

    parser.add_argument('--train_ratio', type=float, default=None)

    # Ablation study
    parser.add_argument('--disable_alpha', type=bool, default=False)
    parser.add_argument('--disable_beta', type=bool, default=False)
    parser.add_argument('--disable_concat', type=bool, default=False)
    parser.add_argument('--attn_activation', type=str, default=None)

    parser.add_argument('--num_gpus', type=int, default=1)


    # add all the available options to the trainer
    args = parser.parse_args()
    train(args)
