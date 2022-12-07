import datetime
import os
import random
import sys
import warnings


warnings.filterwarnings("ignore")

from logzero import logger
from argparse import ArgumentParser, Namespace
from pytorch_lightning.loggers import WandbLogger

import pytorch_lightning
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import EarlyStopping

sys.path.insert(0, "../MultiOmicsGraphEmbedding/")

from moge.model.PyG.node_clf import MetaPath2Vec, LATTEFlatNodeClf, HGTNodeClf, MLP
from moge.model.dgl.node_clf import HANNodeClf, HGConv, R_HGNN
from moge.model.PyG.DeepGOZero import DeepGOZero
from moge.model.utils import tensor_sizes

from run.datasets.deepgraphgo import build_deepgraphgo_model
from run.utils import parse_yaml_config, select_empty_gpus
from run.load_data import load_node_dataset


def train(hparams):
    pytorch_lightning.seed_everything(hparams.seed)

    NUM_GPUS = hparams.num_gpus
    USE_AMP = True
    MAX_EPOCHS = 1000
    MIN_EPOCHS = getattr(hparams, 'min_epochs', 60)

    ### Dataset
    dataset = load_node_dataset(hparams.dataset, hparams.method, hparams=hparams, train_ratio=hparams.train_ratio,
                                dataset_path=hparams.root_path)
    if dataset is not None:
        hparams.n_classes = dataset.n_classes
        hparams.head_node_type = dataset.head_node_type

    ### Callbacks
    callbacks = []
    if hparams.dataset.upper() in ['UNIPROT', "MULTISPECIES", "HUMAN_MOUSE"]:
        METRICS = ["BPO_aupr", "BPO_fmax", "CCO_aupr", "CCO_fmax", "MFO_aupr", "MFO_fmax"]
        early_stopping_args = dict(monitor='val_aupr', mode='max', patience=hparams.early_stopping)
    else:
        METRICS = ["micro_f1", "macro_f1", dataset.name() if "ogb" in dataset.name() else "accuracy"]
        early_stopping_args = dict(monitor='val_loss', mode='min')


    if hparams.method == "HAN":
        default_args = {
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
        model = HANNodeClf(default_args, dataset, metrics=METRICS)

    elif hparams.method == "GTN":
        from moge.model.cogdl.node_clf import GTN

        USE_AMP = True
        default_args = {
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
        model = GTN(Namespace(**default_args), dataset=dataset, metrics=METRICS)

    elif hparams.method == "MetaPath2Vec":
        USE_AMP = True
        default_args = {
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
        model = MetaPath2Vec(Namespace(**default_args), dataset=dataset, metrics=METRICS)

    elif hparams.method == "HGT":
        USE_AMP = False
        default_args = {
            "embedding_dim": 128,
            "n_layers": 2,
            # "fanouts": [10, 10],
            "batch_size": 2 ** 11,
            "activation": "relu",
            "attn_heads": 4,
            "attn_activation": "sharpening",
            "attn_dropout": 0.2,
            "dropout": 0.5,
            "nb_cls_dense_size": 0,
            "nb_cls_dropout": 0,
            "loss_type": "BCE_WITH_LOGITS" if dataset.multilabel else "SOFTMAX_CROSS_ENTROPY",
            "n_classes": dataset.n_classes,
            "use_norm": True,
            "use_class_weights": False,
            "lr": 1e-3,
            "momentum": 0.9,
            "weight_decay": 1e-2,
        }

        model = HGTNodeClf(Namespace(**default_args), dataset, metrics=METRICS)

    elif hparams.method == "HGConv":
        default_args = {
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
        model = HGConv(default_args, dataset, metrics=METRICS)

    elif hparams.method == "R_HGNN":
        default_args = {
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
        model = R_HGNN(default_args, dataset, metrics=METRICS)
    elif "LATTE" in hparams.method:
        USE_AMP = False
        early_stopping_args['monitor'] = 'val_aupr_mean'

        if hparams.method.endswith("-1"):
            t_order = 1
            batch_order = 11
            # early_stopping_args['patience'] = 30

        elif hparams.method.endswith("-2"):
            t_order = 2
            batch_order = 10
            # early_stopping_args['patience'] = 25

        elif hparams.method.endswith("-3"):
            t_order = 3
            batch_order = 10
        else:
            raise Exception()

        dataset.neighbor_sizes = [2048, 2048]

        default_args = {
            "embedding_dim": 512,
            "layer_pooling": "concat",

            "n_layers": len(dataset.neighbor_sizes),
            "t_order": t_order,
            'neighbor_sizes': dataset.neighbor_sizes,
            "batch_size": int(2 ** batch_order),

            "filter_metapaths": {
                'biological_process': {('is_a', 'is_a')},
                'molecular_function': {('is_a', 'is_a')},
                'cellular_component': {('is_a', 'is_a')},
                'Protein': {('coexpression', 'coexpression'),
                            ('cooccurence', 'cooccurence'),
                            ('database', 'database'),
                            ('fusion', 'fusion'),
                            ('neighborhood', 'neighborhood'),
                            ('textmining', 'textmining'),
                            ('experimental', 'experimental')},
            },

            "attn_heads": 8,
            "attn_activation": "LeakyReLU",
            "attn_dropout": 0.2,

            "batchnorm": False,
            "layernorm": True,
            "activation": "relu",
            "dropout": 0.0 if hparams.pred_ntypes == 'biological_process' else 0.5,

            "head_node_type": dataset.head_node_type,

            "n_classes": dataset.n_classes,
            "use_class_weights": False,
            "loss_type": "BCE_WITH_LOGITS" if dataset.multilabel else "SOFTMAX_CROSS_ENTROPY",
            "lr": 1e-3,
            "weight_decay": 1e-2,
            "lr_annealing": None,
        }
        hparams.__dict__.update(default_args)
        model = LATTEFlatNodeClf(hparams, dataset, metrics=METRICS)

    elif 'DeepGraphGO' == hparams.method:
        USE_AMP = False
        model = build_deepgraphgo_model(hparams, base_path='../DeepGraphGO')

    elif 'DeepGOZero' == hparams.method:
        dataset.neighbor_sizes = [0]
        hparams.__dict__.update({
            'go_file': '../deepgozero/data/go.norm',
            "embedding_dim": 1024,
            "hidden_dim": 1024,
            'margin': 0.1,
            'batch_size': 450,
            "loss_type": "BCE_WITH_LOGITS",
            'lr': 1e-3,
        })
        model = DeepGOZero(hparams, dataset, metrics=METRICS)

    elif 'MLP' == hparams.method:
        dataset.neighbor_sizes = [0]
        hparams.__dict__.update({
            "embedding_dim": 256,
            "n_layers": len(dataset.neighbor_sizes),
            'neighbor_sizes': dataset.neighbor_sizes,
            "batch_size": 2 ** 12,
            "dropout": 0.0,
            "loss_type": "BCE_WITH_LOGITS" if dataset.multilabel else "SOFTMAX_CROSS_ENTROPY",
            "n_classes": dataset.n_classes,
            "use_class_weights": False,
            "lr": 1e-3,
        })

        model = MLP(hparams, dataset=dataset, metrics=METRICS)

    else:
        raise Exception(f"Unknown model {hparams.model}")

    model.train_metrics.metrics = {}

    tags = [] + hparams.dataset.split(" ")
    if hasattr(hparams, "namespaces"):
        tags.extend(hparams.namespaces)
    if hasattr(dataset, 'tags'):
        tags.extend(dataset.tags)

    logger = WandbLogger(name=getattr(hparams, 'method', model.name()),
                         tags=list(set(tags)),
                         project="LATTE2GO")
    logger.log_hyperparams(tensor_sizes(hparams))

    if hparams.early_stopping:
        callbacks.append(EarlyStopping(strict=False, **early_stopping_args))

    if hasattr(hparams, "gpu") and isinstance(hparams.gpu, int):
        GPUS = [hparams.gpu]
    elif NUM_GPUS:
        GPUS = select_empty_gpus(NUM_GPUS)

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
        precision=16 if USE_AMP else 32,
    )
    trainer.tune(model)
    trainer.fit(model)
    trainer.test(model)

def update_hparams_from_env(hparams: Namespace, dataset=None):
    updates = {}
    if 'batch_size'.upper() in os.environ:
        updates['batch_size'] = int(os.environ['batch_size'.upper()])
        if hasattr(dataset, 'neighbor_sizes'):
            dataset.neighbor_sizes = [int(n * (updates['batch_size'] / hparams['batch_size'])) \
                                      for n in dataset.neighbor_sizes]

    if 'n_neighbors'.upper() in os.environ:
        updates['n_neighbors'] = int(os.environ['n_neighbors'.upper()])

    logger.info(f"Hparams updates from ENV: {updates}")

    if isinstance(hparams, Namespace):
        hparams.__dict__.update(updates)
    elif isinstance(hparams, dict):
        hparams.update(updates)
    return hparams


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('--method', type=str, default="LATTE")

    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--inductive', type=bool, default=False)

    parser.add_argument('--dataset', type=str, default="ACM")
    parser.add_argument('--pred_ntypes', type=str, default="biological_process")
    parser.add_argument('--ntype_subset', type=str, default="Protein")
    parser.add_argument('--layer_pooling', type=str, default="concat")

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
    parser.add_argument('--hours', type=int, default=23)

    parser.add_argument('-y', '--config', help="configuration file *.yml", type=str, required=False)
    # add all the available options to the trainer
    args = parse_yaml_config(parser)
    train(args)
