from argparse import Namespace

import pytorch_lightning as pl
from fairscale.nn import auto_wrap
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

    def configure_sharded_model(self):
        # modules are sharded across processes
        # as soon as they are wrapped with ``wrap`` or ``auto_wrap``.
        # During the forward/backward passes, weights get synced across processes
        # and de-allocated once computation is complete, saving memory.

        # Wraps the layer in a Fully Sharded Wrapper automatically
        self.bert = auto_wrap(self.bert)

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


