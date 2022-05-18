import argparse
import os.path
from argparse import Namespace

import yaml
from moge.dataset.sequences import MaskedLMDataset
from moge.model.transformers.mlm import BertMLM
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from run.load_data import load_link_dataset
from transformers import BertConfig


def train_mlm(hparams: Namespace):
    network_dataset = load_link_dataset(name=hparams.dataset, hparams=hparams)

    node_type = hparams.node_type

    mlm_dataset = MaskedLMDataset(data=network_dataset.G[node_type]["sequence"],
                                  tokenizer=network_dataset.seq_tokenizer[node_type],
                                  mlm_probability=hparams.mlm_probability,
                                  max_len=hparams.max_length)

    bert_config = BertConfig.from_pretrained(hparams.model)

    bert_config.num_hidden_layers = 2
    bert_config.num_attention_heads = 4
    bert_config.hidden_dropout_prob = 0.1
    bert_config.intermediate_size = 128
    bert_config.hidden_size = 128
    bert_config.num_labels = 128
    bert_config.gradient_checkpointing = True
    bert_config.max_position_embeddings = hparams.max_length

    model = BertMLM(bert_config, dataset=mlm_dataset, hparams=hparams)

    trainer = Trainer(
        gpus=hparams.num_gpus,
        strategy="fsdp" if hparams.num_gpus > 1 else None,
        # auto_lr_find=True,
        # auto_scale_batch_size=True,
        max_epochs=hparams.max_epochs,
        callbacks=[
            EarlyStopping(monitor='loss', patience=5, min_delta=0.01, strict=False),
        ],
        weights_summary='top',
        precision=16
    )

    if hparams.num_gpus == 1:
        trainer.tune(model)

    try:
        trainer.fit(model, val_dataloaders=None)

    except Exception as e:
        print(e)

    finally:
        if trainer.node_rank == 0 and trainer.local_rank == 0:
            model.bert.save_pretrained(hparams.save_path + "_" + node_type)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default="data/gtex_rna_ppi_multiplex_network.pickle")
    parser.add_argument('-n', '--node_type', type=str, default="GO_term")
    parser.add_argument('-m', '--model', type=str, default="dmis-lab/biobert-base-cased-v1.2")
    parser.add_argument('-o', '--save_path', type=str, default="models/bert_mlm")

    parser.add_argument('--num_hidden_layers', type=int, default=2)
    parser.add_argument('--num_attention_heads', type=int, default=4)
    parser.add_argument('--hidden_dropout_prob', type=float, default=0.1)
    parser.add_argument('--intermediate_size', type=int, default=128)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_labels', type=int, default=128)

    parser.add_argument('--use_reverse', type=bool, default=False)
    parser.add_argument('--head_node_type', type=str, default="MessengerRNA")  # Ignore but needed

    parser.add_argument('--mlm_probability', type=float, default=0.15)
    parser.add_argument('-l', '--max_length', type=int, default=150)
    parser.add_argument('--max_epochs', type=int, default=500)
    parser.add_argument('-b', '--batch_size', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-5)

    parser.add_argument('-g', '--num_gpus', type=int, default=1)

    parser.add_argument('-y', '--config', help="configuration file *.yml", type=str, required=False)
    args = parser.parse_args()

    if isinstance(args.config, str) and os.path.exists(args.config):  # yaml priority is higher than args
        opt = yaml.load(open(args.config), Loader=yaml.FullLoader)
        opt.update(vars(args))
        args = opt

        print(args)

    train_mlm(hparams=args)
