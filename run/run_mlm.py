import argparse
from argparse import Namespace

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from transformers import BertConfig

from moge.dataset.sequences import MaskedLMDataset
from moge.model.transformers.mlm import BertMLM
from .load_data import load_link_dataset


def train_mlm(hparams: Namespace):
    network_dataset = load_link_dataset(name=hparams.dataset, hparams=hparams)

    mlm_dataset = MaskedLMDataset(data=network_dataset.G[hparams.ntype]["sequence"],
                                  tokenizer=network_dataset.seq_tokenizer[hparams.ntype],
                                  mlm_probability=hparams.mlm_probability,
                                  max_len=hparams.max_length)

    bert_config = BertConfig.from_pretrained(hparams.model_name)

    bert_config.num_hidden_layers = 2
    bert_config.num_attention_heads = 4
    bert_config.hidden_dropout_prob = 0.1
    bert_config.gradient_checkpointing = True
    bert_config.intermediate_size = 128
    bert_config.hidden_size = 128
    bert_config.num_labels = 128
    bert_config.max_position_embeddings = hparams.max_length

    model = BertMLM(bert_config, dataset=mlm_dataset, hparams=hparams)

    trainer = Trainer(
        gpus=hparams.num_gpus,
        strategy="fsdp" if hparams.num_gpus > 1 else None,
        # auto_lr_find=True,
        # auto_scale_batch_size=True,
        max_epochs=hparams.max_epochs,
        callbacks=[
            EarlyStopping(monitor='loss', patience=5, min_delta=0.01),
        ],
        weights_summary='top',
        precision=16
    )
    trainer.tune(model)

    try:
        trainer.fit(model, val_dataloaders=None)

    except Exception as e:
        print(e)

    finally:
        if trainer.node_rank == 0 and trainer.local_rank == 0:
            model.bert.save_pretrained(hparams.save_path + "_" + hparams.ntype)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default="data/gtex_rna_ppi_multiplex_network.pickle")
    parser.add_argument('--ntype', type=str, default="GO_term")
    parser.add_argument('-m', '--model', type=str, default="dmis-lab/biobert-base-cased-v1.2")
    parser.add_argument('-o', '--save_path', type=str, default="models/bert_mlm")

    parser.add_argument('--mlm_probability', type=float, default=0.15)
    parser.add_argument('--max_length', type=int, default=150)
    parser.add_argument('--max_epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=4)

    parser.add_argument('--num_gpus', type=int, default=1)

    args = parser.parse_args()
