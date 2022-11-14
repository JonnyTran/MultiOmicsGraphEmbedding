import datetime
import glob
import logging
import os
import random
import sys
from argparse import ArgumentParser, Namespace

from run.datasets.deepgraphgo import build_deepgraphgo_model
from run.utils import parse_yaml_config, select_empty_gpus

logger = logging.getLogger("wandb")
logger.setLevel(logging.ERROR)

sys.path.insert(0, "../MultiOmicsGraphEmbedding/")

import pytorch_lightning
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import EarlyStopping

from moge.model.PyG.node_clf import MetaPath2Vec, LATTEFlatNodeClf
from moge.model.cogdl.node_clf import GTN
from moge.model.dgl.node_clf import HANNodeClf, HGTNodeClf, HGConv, R_HGNN

from pytorch_lightning.loggers import WandbLogger

from run.load_data import load_node_dataset


def train(hparams):
    pytorch_lightning.seed_everything(hparams.seed)

    NUM_GPUS = hparams.num_gpus
    USE_AMP = True
    MAX_EPOCHS = 1000
    MIN_EPOCHS = getattr(hparams, 'min_epochs', None)

    if hasattr(hparams, "gpu") and isinstance(hparams.gpu, int):
        GPUS = [hparams.gpu]
    elif hparams.num_gpus == 1:
        GPUS = select_empty_gpus()
    else:
        GPUS = random.sample([0, 1, 2], NUM_GPUS)

    ### Dataset
    dataset = load_node_dataset(hparams.dataset, hparams.method, hparams=hparams, train_ratio=hparams.train_ratio,
                                dataset_path=hparams.root_path)
    if dataset is not None:
        hparams.n_classes = dataset.n_classes
        hparams.head_node_type = dataset.head_node_type

    hparams.neighbor_sizes = [hparams.n_neighbors, ] * hparams.n_layers

    ### Callbacks
    callbacks = []
    if "GO" in hparams.dataset or 'uniprot' in hparams.dataset.lower():
        METRICS = ["BPO_aupr", "BPO_fmax", "CCO_aupr", "CCO_fmax", "MFO_aupr", "MFO_fmax"]
        early_stopping_args = dict(monitor='val_aupr', mode='max')
    else:
        METRICS = ["micro_f1", "macro_f1", dataset.name() if "ogb" in dataset.name() else "accuracy"]
        early_stopping_args = dict(monitor='val_loss', mode='min')

    # Path to set MultiLabelBinarizer
    mlb_path = os.path.expanduser('~/Bioinformatics_ExternalData/LATTE2GO')
    if 'MULTISPECIES' in hparams.dataset:
        results = glob.glob(f'{mlb_path}/{hparams.dataset}-{hparams.pred_ntypes}/go_id.mlb')
    elif 'HUMAN_MOUSE' in hparams.dataset:
        results = glob.glob(f'{mlb_path}/{hparams.dataset}-{hparams.pred_ntypes}/go_id.mlb')
    else:
        results = []
    if results:
        hparams.mlb_path = results[0]

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
        model = HANNodeClf(args, dataset, metrics=METRICS)
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
        model = HGTNodeClf(Namespace(**args), dataset, metrics=METRICS)
    elif hparams.method == "HGConv":
        args = {
            'seed': hparams.run,
            'head_node_type': dataset.head_node_type,
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
        ModelClass = HGConv
        model = HGConv(args, dataset, metrics=METRICS)
    elif hparams.method == "R_HGNN":
        args = {
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
        USE_AMP = False

        extra_args = {}
        if "-1" in hparams.method:
            t_order = 1
            batch_order = 12
        elif "-2" in hparams.method:
            t_order = 2
            batch_order = 11

        elif "-3" in hparams.method:
            t_order = 3
            batch_order = 10
            t_order = 2

        args = {
            "embedding_dim": 128,
            "layer_pooling": "order_concat",

            "n_layers": len(dataset.neighbor_sizes),
            "t_order": t_order,
            "batch_size": int(2 ** batch_order),
            **extra_args,

            "attn_heads": 4,
            "attn_activation": "LeakyReLU",
            "attn_dropout": 0.5,

            "batchnorm": False,
            "layernorm": False,
            "activation": "relu",
            "dropout": 0.5,
            "input_dropout": False,

            "nb_cls_dense_size": 0,
            "nb_cls_dropout": 0.0,

            "edge_threshold": 0.0,
            "edge_sampling": False,

            "head_node_type": dataset.head_node_type,

            "n_classes": dataset.n_classes,
            "use_class_weights": False,
            "loss_type": "BCE_WITH_LOGITS" if dataset.multilabel else "SOFTMAX_CROSS_ENTROPY",
            "stochastic_weight_avg": False,
            "lr": 0.001,
            "epochs": 300,
            "patience": 10,
            "weight_decay": 1e-2,
            "lr_annealing": None,
        }

        args.update(hparams.__dict__)
        model = LATTEFlatNodeClf(Namespace(**args), dataset, metrics=METRICS)

    elif 'DeepGraphGO' == hparams.method:
        USE_AMP = False
        model = build_deepgraphgo_model(hparams, base_path='../DeepGraphGO')

    else:
        raise Exception(f"Unknown model {hparams.embedder}")

    tags = [] + hparams.dataset.split(" ")
    if hasattr(hparams, "namespaces"):
        tags.extend(hparams.namespaces)
    if hasattr(dataset, 'tags'):
        tags.extend(dataset.tags)

    logger = WandbLogger(name=model.name(), tags=list(set(tags)), project="LATTE2GO")
    logger.log_hyperparams(hparams)

    if hparams.early_stopping:
        callbacks.append(EarlyStopping(patience=hparams.early_stopping, strict=False, **early_stopping_args))

    trainer = Trainer(
        accelerator='cuda',
        devices=GPUS,
        # enable_progress_bar=False,
        # auto_scale_batch_size=True if hparams.n_layers > 2 else False,
        max_epochs=MAX_EPOCHS,
        min_epochs=MIN_EPOCHS,
        callbacks=callbacks,
        logger=logger,
        max_time=datetime.timedelta(hours=hparams.hours) \
            if hasattr(hparams, "hours") and isinstance(hparams.hours, (int, float)) else None,
        # plugins='deepspeed' if NUM_GPUS > 1 else None,
        # accelerator='ddp_spawn',
        # plugins='ddp_sharded'
        precision=16 if USE_AMP else 32
    )
    trainer.tune(model)
    trainer.fit(model)
    trainer.test(model)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('--method', type=str, default="LATTE")

    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--inductive', type=bool, default=False)

    parser.add_argument('--dataset', type=str, default="ACM")
    parser.add_argument('--pred_ntypes', type=str, default="biological_process")

    parser.add_argument('--use_emb', type=str,
                        default="/home/jonny/PycharmProjects/MultiOmicsGraphEmbedding/moge/module/dgl/NARS/")
    parser.add_argument('--root_path', type=str,
                        default="/home/jonny/Bioinformatics_ExternalData/OGB/")

    parser.add_argument('--train_ratio', type=float, default=None)
    parser.add_argument('--early_stopping', type=int, default=5)

    # Ablation study
    parser.add_argument('--disable_alpha', type=bool, default=False)
    parser.add_argument('--disable_beta', type=bool, default=False)
    parser.add_argument('--disable_concat', type=bool, default=False)

    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--seed', type=int, default=random.randint(0, int(1e4)))


    # add all the available options to the trainer
    args = parse_yaml_config(parser)
    train(args)
