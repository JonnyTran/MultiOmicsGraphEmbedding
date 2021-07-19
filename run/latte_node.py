import logging
import pprint
import sys, random
from argparse import ArgumentParser, Namespace

# logger = logging.getLogger("wandb")
# logger.setLevel(logging.INFO)
from moge.module.utils import tensor_sizes

sys.path.insert(0, "../MultiOmicsGraphEmbedding/")

from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from moge.module.PyG.node_clf import LATTENodeClf
from run.load_data import load_node_dataset


def train(hparams: Namespace):
    NUM_GPUS = hparams.num_gpus
    USE_AMP = True  # True if NUM_GPUS > 1 else False
    MAX_EPOCHS = 100

    seed_everything(seed=42)

    neighbor_sizes = [hparams.n_neighbors, ]
    for t in range(1, hparams.n_layers):
        if neighbor_sizes[-1] != -1:
            neighbor_sizes.extend([neighbor_sizes[-1] // 2])
        else:
            neighbor_sizes.extend([10])

    hparams.neighbor_sizes = list(reversed(neighbor_sizes))
    print("neighbor_sizes", hparams.neighbor_sizes, hparams.n_layers)

    dataset = load_node_dataset(hparams.dataset, method="LATTE", args=hparams, train_ratio=None,
                                dataset_path=hparams.root_path)

    METRICS = ["micro_f1", "macro_f1", dataset.name() if "ogb" in dataset.name() else "accuracy"]

    hparams.loss_type = "BCE" if dataset.multilabel else hparams.loss_type
    hparams.n_classes = dataset.n_classes
    hparams.head_node_type = dataset.head_node_type
    model = LATTENodeClf(hparams, dataset, metrics=METRICS)

    logger = WandbLogger(name=model.name(), tags=[dataset.name()], project="ogb_nodepred")

    callbacks = []
    callbacks.append(EarlyStopping(monitor='val_moving_loss', patience=5,
                                   min_delta=0.001, strict=False))
    callbacks.append(ModelCheckpoint(monitor='val_loss',
                                     filename=model.name() + '-' + dataset.name() + '-{epoch:02d}-{val_loss:.3f}'))

    trainer = Trainer(
        gpus=random.sample([0, 1, 2], NUM_GPUS),
        accelerator='ddp' if NUM_GPUS > 1 else None,
        gradient_clip_val=hparams.gradient_clip_val,
        stochastic_weight_avg=hparams.stochastic_weight_avg,
        auto_lr_find=False,
        # auto_scale_batch_size=True if hparams.n_layers > 2 else False,
        max_epochs=MAX_EPOCHS,
        callbacks=callbacks,
        logger=logger,
        # plugins='deepspeed' if NUM_GPUS > 1 else None,
        #     accelerator='ddp_spawn',
        #     plugins='ddp_sharded'
        amp_level='O1' if USE_AMP else None,
        precision=16 if USE_AMP else 32
    )
    trainer.tune(model)

    trainer.fit(model)

    if trainer.checkpoint_callback is not None:
        model = LATTENodeClf.load_from_checkpoint(trainer.checkpoint_callback.best_model_path,
                                                  hparams=hparams,
                                                  dataset=dataset,
                                                  metrics=METRICS)
        print(trainer.checkpoint_callback.best_model_path)
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

    # parametrize the network
    parser.add_argument('-g', '--num_gpus', type=int, default=1)
    parser.add_argument("-d", '--embedding_dim', type=int, default=128)
    parser.add_argument('-n', '--batch_size', type=int, default=1024)
    parser.add_argument('--n_neighbors', type=int, default=20)
    parser.add_argument("-l", '--n_layers', type=int, default=2)
    parser.add_argument("-t", '--t_order', type=int, default=2)

    parser.add_argument('--activation', type=str, default="relu")
    parser.add_argument('--batchnorm', type=bool, default=False)
    parser.add_argument('--layernorm', type=bool, default=True)
    parser.add_argument('--layer_pooling', type=str, default="last")
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--input_dropout', type=bool, default=True)

    parser.add_argument('--attn_heads', type=int, default=4)
    parser.add_argument('--attn_activation', type=str, default="LeakyReLU")
    parser.add_argument('--attn_dropout', type=float, default=0.5)

    parser.add_argument('--nb_cls_dense_size', type=int, default=0)
    parser.add_argument('--nb_cls_dropout', type=float, default=0.4)

    parser.add_argument('--edge_threshold', type=float, default=0.0)
    parser.add_argument('--edge_sampling', type=bool, default=False)

    parser.add_argument('--use_proximity', type=bool, default=False)
    parser.add_argument('--neg_sampling_ratio', type=float, default=5.0)

    # parser.add_argument('--reduction', type=str, default="none")
    parser.add_argument('--use_class_weights', type=bool, default=False)

    # Optimizer parameters
    parser.add_argument('--sparse', type=bool, default=False)
    parser.add_argument('-a', '--accelerator', type=str, default="ddp|horovod")
    parser.add_argument('--loss_type', type=str, default="SOFTMAX_CROSS_ENTROPY")
    parser.add_argument('--lr', type=float, default=0.03)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--gradient_clip_val', type=float, default=0.0)
    parser.add_argument('--stochastic_weight_avg', type=bool, default=False)
    parser.add_argument('--early_stopping', type=int, default=2)
    # add all the available options to the trainer
    # parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    train(args)
