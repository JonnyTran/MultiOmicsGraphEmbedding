import argparse
import datetime
import os.path
import traceback
from argparse import Namespace

from moge.dataset.sequences import MaskedLMDataset
from moge.model.transformers.mlm import BertMLM
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from run.load_data import load_link_dataset
from run.utils import parse_yaml
from transformers import BertConfig


def train_mlm(hparams: Namespace):
    network_dataset = load_link_dataset(name=hparams.dataset, hparams=hparams, path=hparams.root_path)

    node_type = hparams.node_type

    tokenizer = network_dataset.seq_tokenizer[node_type]
    mlm_dataset = MaskedLMDataset(data=network_dataset.G[node_type]["sequence"],
                                  tokenizer=tokenizer,
                                  mlm_probability=hparams.mlm_probability,
                                  max_length=hparams.max_length)

    if hasattr(hparams, 'load_path') and isinstance(hparams.load_path, str) and os.path.exists(hparams.load_path):
        bert_config = hparams.load_path

    else:
        bert_config = BertConfig.from_pretrained(hparams.model)
        print("\nSet Bert Config")
        for key in bert_config.__dict__.keys():
            if key in hparams:
                bert_config.__dict__[key] = hparams.__dict__[key]
                print(key, hparams.__dict__[key])

        bert_config.gradient_checkpointing = True
        bert_config.max_position_embeddings = hparams.max_length
        print("\n", bert_config)

    model = BertMLM(bert_config, dataset=mlm_dataset, hparams=hparams)

    # os.environ["CUDA_LAUNCH_BLOCKING"] = 1

    trainer = Trainer(
        gpus=[hparams.gpu] if hasattr(hparams, "gpu") and isinstance(hparams.gpu, int) \
            else hparams.num_gpus,
        auto_select_gpus=True,
        strategy="fsdp" if isinstance(hparams.num_gpus, int) and hparams.num_gpus > 1 else None,
        # auto_lr_find=True,
        # auto_scale_batch_size=True,
        max_epochs=hparams.max_epochs,
        callbacks=[
            EarlyStopping(monitor='loss', patience=20, min_delta=0.01, check_on_train_epoch_end=True),
        ],
        limit_val_batches=0,
        max_time=datetime.timedelta(hours=hparams.hours) \
            if hasattr(hparams, "hours") and isinstance(hparams.hours, (int, float)) else None,
        weights_summary='top',
        precision=16
    )

    try:
        trainer.fit(model, val_dataloaders=None)

    except Exception as e:
        print(e)
        traceback.print_exc()

    finally:
        if trainer.node_rank == 0 and trainer.local_rank == 0 and trainer.current_epoch > 10 and hparams.save_path is not None:
            print(f"Saving BERT model at {hparams.save_path}")
            model.bert.save_pretrained(hparams.save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default="rna_ppi_go_mlm")
    parser.add_argument('-p', '--root_path', type=str, default="data/gtex_rna_ppi_multiplex_network.pickle")

    parser.add_argument('-n', '--node_type', type=str, default="GO_term")
    parser.add_argument('-m', '--model', type=str, default="dmis-lab/biobert-base-cased-v1.2")
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('-o', '--save_path', type=str, default="models/bert_mlm")

    parser.add_argument('--use_reverse', type=bool, default=False)
    parser.add_argument('--head_node_type', type=str, default="MessengerRNA")  # Ignore but needed

    parser.add_argument('--mlm_probability', type=float, default=0.20)
    parser.add_argument('-l', '--max_length', type=int, default=150)
    parser.add_argument('--max_epochs', type=int, default=2000)
    parser.add_argument('-b', '--batch_size', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-5)

    parser.add_argument('-g', '--num_gpus', type=int, default=1)
    parser.add_argument('--gpu', type=int, default=None)

    parser.add_argument('--hours', type=float, default=None)

    parser.add_argument('-y', '--config', help="configuration file *.yml", type=str, required=False)
    args = parse_yaml(parser)

    train_mlm(hparams=args)
    print()
