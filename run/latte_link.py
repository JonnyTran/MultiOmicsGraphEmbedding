import datetime
import logging
import sys
import traceback
from argparse import ArgumentParser

import torch

from run.utils import parse_yaml

logger = logging.getLogger("wandb")
logger.setLevel(logging.ERROR)
sys.path.insert(0, "../MultiOmicsGraphEmbedding/")

from pytorch_lightning.trainer import Trainer

from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from moge.model.PyG.link_pred import LATTELinkPred
from run.load_data import load_link_dataset


def train(hparams):
    if hparams.dataset == 'rna_ppi_go':
        assert hasattr(hparams, "max_length") and hasattr(hparams, "bert_config")

        metrics = {"BPO": ["ogbl-biokg", 'precision', 'recall'],
                   "CCO": ["ogbl-biokg", 'precision', 'recall'],
                   "MFO": ["ogbl-biokg", 'precision', 'recall'], }
        callbacks = [EarlyStopping(monitor='val_BPO_mrr', patience=10, strict=False)]

        if hasattr(hparams, "sweep") and hparams.sweep:
            callbacks.append(EarlyStopping(monitor='val_loss', patience=5, strict=False))
    else:
        metrics = [hparams.dataset]
        callbacks = [EarlyStopping(monitor='val_loss', patience=5, min_delta=0.01, strict=False)]

    # Load dataset
    dataset = load_link_dataset(hparams.dataset, hparams=hparams, path=hparams.root_path)
    hparams.n_classes = dataset.n_classes

    hparams.batch_size = adjust_batch_size(hparams)

    # Resume from model checkpoint
    if hasattr(hparams, "load_path") and hparams.load_path:
        model = LATTELinkPred.load_from_checkpoint(hparams.load_path,
                                                   hparams=hparams, dataset=dataset, metrics=metrics)
        print(f"Loaded model from {hparams.load_path}")
    else:
        model = LATTELinkPred(hparams, dataset, metrics=metrics)

    # Logger
    logger = None if hasattr(hparams, "no_wandb") and hparams.no_wandb else \
        WandbLogger(name=model.name(), tags=[dataset.name()], project="multiplex-comparison")
    logger.log_hyperparams(hparams)

    # Trainer
    if hasattr(hparams, "gpu") and isinstance(hparams.gpu, int):
        GPUS = [hparams.gpu]
    elif hparams.num_gpus == 1:
        gpu_mem_free = {i: torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())}
        best_gpu = max(gpu_mem_free, key=gpu_mem_free.get)
        print("gpu_mem_free", gpu_mem_free, "selected GPU", best_gpu)
        GPUS = [best_gpu]
    else:
        GPUS = hparams.num_gpus

    trainer = Trainer(
        gpus=GPUS,
        strategy="fsdp" if isinstance(hparams.num_gpus, int) and hparams.num_gpus > 1 else None,
        enable_progress_bar=False,
        # auto_lr_find=False,
        # auto_scale_batch_size=True, log_every_n_steps=1,
        log_every_n_steps=len(dataset.training_idx) // hparams.batch_size,
        max_epochs=hparams.max_epochs,
        callbacks=callbacks,
        logger=logger,
        weights_summary='top',
        max_time=datetime.timedelta(hours=hparams.hours) \
            if hasattr(hparams, "hours") and isinstance(hparams.hours, (int, float)) else None,
        precision=16
    )

    trainer.tune(model)

    try:
        trainer.fit(model)
        trainer.test(model)

    except Exception as e:
        print(e)
        traceback.print_exc()

    finally:
        if trainer.node_rank == 0 and trainer.local_rank == 0 and trainer.current_epoch > 1 and \
                hasattr(hparams, "save_path") and hparams.save_path is not None:
            trainer.save_checkpoint(hparams.save_path)
            print(f"Saved model checkpoint to {hparams.save_path}")

    print()


def adjust_batch_size(hparams):
    batch_size = hparams.batch_size
    if hparams.n_neighbors > 128:
        batch_size = batch_size // (hparams.n_neighbors // 128)
    if hparams.embedding_dim > 256:
        batch_size = batch_size // (hparams.embedding_dim / 256)
    if hparams.n_layers > 2:
        batch_size = batch_size // 2

    print(f"Adjusted batch_size to", batch_size)

    return int(batch_size)


if __name__ == "__main__":
    parser = ArgumentParser()

    # parametrize the network
    parser.add_argument('--dataset', type=str, default="ogbl-biokg")
    parser.add_argument('-p', '--root_path', type=str, default="data/gtex_rna_ppi_multiplex_network.pickle")
    parser.add_argument('-d', '--embedding_dim', type=int, default=128)
    parser.add_argument('-t', '--n_layers', type=int, default=1)
    parser.add_argument('--t_order', type=int, default=1)
    parser.add_argument('-b', '--batch_size', type=int, default=12000)
    parser.add_argument('--activation', type=str, default="relu")
    parser.add_argument('--attn_heads', type=int, default=1)
    parser.add_argument('--attn_activation', type=str, default="sharpening")
    parser.add_argument('--attn_dropout', type=float, default=0.5)

    parser.add_argument('--batchnorm', type=bool, default=True)
    parser.add_argument('--layernorm', type=bool, default=False)
    parser.add_argument('--dropout', type=float, default=0.5)

    parser.add_argument('--n_neighbors', type=int, default=50)
    parser.add_argument('--use_proximity', type=bool, default=False)
    parser.add_argument('--neg_sampling_ratio', type=float, default=64.0)

    parser.add_argument('--head_node_type', type=str, default=None)  # Ignore but needed

    parser.add_argument('--use_reverse', type=bool, default=True)
    parser.add_argument('--no_wandb', action='store_true')

    parser.add_argument('--max_epochs', type=int, default=2000)
    parser.add_argument('--loss_type', type=str, default="CONTRASTIVE_LOSS")
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    parser.add_argument('-n', '--num_gpus', type=int, default=1)
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--sweep', type=bool, default=False)
    parser.add_argument('--hours', type=float, default=None)

    parser.add_argument('--train_date', type=str, default='2018-01-01')

    args = parse_yaml(parser)

    train(args)
