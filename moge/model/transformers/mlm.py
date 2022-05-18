import argparse
from argparse import Namespace

import pytorch_lightning as pl
from torch import Tensor
from transformers import BertForMaskedLM, AdamW, BertConfig

from moge.dataset.sequences import MaskedLMDataset


class BertMLM(pl.LightningModule):
    def __init__(self, config: BertConfig, dataset: MaskedLMDataset, hparams: Namespace):
        super().__init__()
        self.dataset = dataset

        if isinstance(config, str):
            self.bert = BertForMaskedLM.from_pretrained(config)
        else:
            self.bert = BertForMaskedLM(config)

        self.mlm_probability = dataset.mlm_probability
        self.batch_size = hparams.batch_size
        self.lr = hparams.lr

    def forward(self, input_ids: Tensor, labels: Tensor):
        return self.bert(input_ids=input_ids, labels=labels)

    def training_step(self, batch, batch_idx):

        input_ids = batch["input_ids"]
        labels = batch["labels"]

        outputs = self.forward(input_ids=input_ids, labels=labels)
        loss = outputs[0]

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx)

    def train_dataloader(self):
        return self.dataset.train_dataloader(batch_size=self.batch_size)

    def val_dataloader(self):
        return self.dataset.valid_dataloader(batch_size=self.batch_size)

    def test_dataloader(self):
        return self.dataset.test_dataloader(batch_size=self.batch_size)

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr)


def train_mlm(hparams: Namespace):
    network_dataset = load_network_dataset()

    mlm_dataset = MaskedLMDataset(data=network_dataset.G["GO_term"]["sequence"],
                                  tokenizer=network_dataset.seq_tokenizer["GO_term"],
                                  mlm_probability=0.15, max_len=150)
    mlm_dataset

    bert_config = BertConfig.from_pretrained(hparams.model_name)

    bert_config.num_hidden_layers = 2
    bert_config.num_attention_heads = 4
    bert_config.max_position_embeddings = 150
    bert_config.gradient_checkpointing = True
    bert_config.intermediate_size = 64
    bert_config.hidden_size = 64
    bert_config.num_labels = 128

    pass


def load_network_dataset():
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--node_type', type=str, default="GO_term")
    parser.add_argument('--model_name', type=str, default="dmis-lab/biobert-base-cased-v1.2")

    parser.add_argument('--max_len', type=int, default=False)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=4)

    args = parser.parse_args()
