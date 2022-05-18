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
