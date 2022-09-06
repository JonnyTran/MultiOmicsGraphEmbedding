import random
import sys
from argparse import ArgumentParser, Namespace

sys.path.insert(0, "../MultiOmicsGraphEmbedding/")

from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from moge.model.PyG.node_clf import LATTEFlatNodeClf
from run.load_data import load_node_dataset
from run.utils import parse_yaml_config, select_empty_gpu


def train(hparams: Namespace):
    NUM_GPUS = hparams.num_gpus
    USE_AMP = True  # True if NUM_GPUS > 1 else False
    MAX_EPOCHS = 100
    seed_everything(seed=hparams.seed)

    hparams.neighbor_sizes = [hparams.n_neighbors, ] * hparams.n_layers

    dataset = load_node_dataset(hparams.dataset, method="LATTE", hparams=hparams, train_ratio=None,
                                dataset_path=hparams.root_path)

    if "GO" in hparams.dataset:
        METRICS = ["BPO_aupr", "BPO_fmax", "CCO_aupr", "CCO_fmax", "MFO_aupr", "MFO_fmax"]
    else:
        METRICS = ["micro_f1", "macro_f1", dataset.name() if "ogb" in dataset.name() else "accuracy"]

    hparams.loss_type = hparams.loss_type
    hparams.n_classes = dataset.n_classes
    hparams.head_node_type = dataset.head_node_type
    model = LATTEFlatNodeClf(hparams, dataset, metrics=METRICS)

    tags = [dataset.name()]
    if hasattr(hparams, "namespaces"):
        tags.extend(hparams.namespaces)
    logger = WandbLogger(name=model.name(), tags=tags, project="LATTE2GO")
    logger.log_hyperparams(hparams)

    callbacks = []
    if hparams.early_stopping:
        callbacks.append(EarlyStopping(monitor='val_loss', patience=hparams.early_stopping, strict=False))
    callbacks.append(ModelCheckpoint(monitor='val_loss',
                                     filename=model.name() + '-' + dataset.name() + '-{epoch:02d}-{val_loss:.3f}'))

    if hasattr(hparams, "gpu") and isinstance(hparams.gpu, int):
        GPUS = [hparams.gpu]
    elif hparams.num_gpus == 1:
        best_gpu = select_empty_gpu()
        GPUS = [best_gpu]
    else:
        GPUS = random.sample([0, 1, 2], NUM_GPUS)

    trainer = Trainer(
        accelerator='cuda',
        devices=GPUS,
        auto_lr_find=False,
        # auto_scale_batch_size=True if hparams.n_layers > 2 else False,
        log_every_n_steps=1,
        max_epochs=MAX_EPOCHS,
        callbacks=callbacks,
        logger=logger,
        # plugins='deepspeed' if NUM_GPUS > 1 else None,
        # accelerator='ddp_spawn',
        # plugins='ddp_sharded'
        precision=16 if USE_AMP else 32
    )
    trainer.tune(model)
    trainer.fit(model)

    try:
        if trainer.checkpoint_callback is not None:
            model = LATTEFlatNodeClf.load_from_checkpoint(trainer.checkpoint_callback.best_model_path,
                                                          hparams=hparams,
                                                          dataset=dataset,
                                                          metrics=METRICS)
            print(trainer.checkpoint_callback.best_model_path)
    except:
        pass
    finally:
        trainer.test(model)

if __name__ == "__main__":
    parser = ArgumentParser()

    # Dataset
    parser.add_argument('--dataset', type=str, default="ACM")
    parser.add_argument('--root_path', type=str, default="/home/jonny/Bioinformatics_ExternalData/OGB/")
    parser.add_argument('--use_emb', type=str,
                        default="/home/jonny/PycharmProjects/MultiOmicsGraphEmbedding/moge/module/dgl/NARS/")
    parser.add_argument('--inductive', type=bool, default=False)
    parser.add_argument('--use_reverse', type=bool, default=True)
    parser.add_argument('--freeze_embeddings', type=bool, default=True)
    parser.add_argument('--ntype_subset', type=str, default=None)

    parser.add_argument('--head_node_type', type=str, default='Protein')
    parser.add_argument('--namespaces', type=str, default='m')
    parser.add_argument('--train_date', type=str, default='2018-01-01')
    parser.add_argument('--valid_date', type=str, default='2018-07-01')
    parser.add_argument('--test_date', type=str, default='2021-04-01')

    # parametrize the network
    parser.add_argument('-g', '--num_gpus', type=int, default=1)
    parser.add_argument("-d", '--embedding_dim', type=int, default=256)
    parser.add_argument('-n', '--batch_size', type=int, default=1024)
    parser.add_argument('--neighbor_loader', type=str, default="HGTLoader")
    parser.add_argument('--n_neighbors', type=int, default=20)
    parser.add_argument("-l", '--n_layers', type=int, default=2)
    parser.add_argument("-t", '--t_order', type=int, default=2)
    parser.add_argument('--layer_pooling', type=str, default="concat")

    parser.add_argument('--activation', type=str, default="relu")
    parser.add_argument('--batchnorm', type=bool, default=False)
    parser.add_argument('--layernorm', type=bool, default=False)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--input_dropout', type=bool, default=False)

    parser.add_argument('--attn_heads', type=int, default=4)
    parser.add_argument('--attn_activation', type=str, default="LeakyReLU")
    parser.add_argument('--attn_dropout', type=float, default=0.2)

    parser.add_argument('--nb_cls_dense_size', type=float, default=0)
    parser.add_argument('--nb_cls_dropout', type=float, default=0)

    parser.add_argument('--edge_threshold', type=float, default=0.0)
    parser.add_argument('--edge_sampling', type=bool, default=False)

    parser.add_argument('--cls_graph', type=bool, default=False)

    # parser.add_argument('--reduction', type=str, default="none")
    parser.add_argument('--use_class_weights', type=bool, default=False)
    parser.add_argument('--use_pos_weights', type=bool, default=False)

    # Optimizer parameters
    parser.add_argument('--sparse', type=bool, default=False)
    parser.add_argument('-a', '--accelerator', type=str, default="cuda")
    parser.add_argument('--loss_type', type=str, default="BCE_WITH_LOGITS")
    parser.add_argument('--lr', type=float, default=0.03)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--gradient_clip_val', type=float, default=0.0)
    parser.add_argument('--early_stopping', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    # add all the available options to the trainer
    # parser = pl.Trainer.add_argparse_args(parser)

    args = parse_yaml_config(parser)
    train(args)
