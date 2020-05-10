import torch

import pytorch_lightning as pl
from .metrics import Metrics


class LightningModel(pl.LightningModule):
    def __init__(self, model):
        super(LightningModel, self).__init__()

        self._model = model
        self.metrics = Metrics()

    def forward(self, X):
        return self._model(X)

    def training_step(self, batch, batch_nb):
        X, y, weights = batch

        # print("batch_nb", batch_nb)
        # print({k: v.size() if not isinstance(v, list) else (len(v), len(v[0])) for k, v in X.items()})

        Y_hat = self.forward(X)
        loss = self._model.loss(Y_hat, y, weights)

        self.metrics.update(Y_hat, y, training=True)
        progress_bar = self.metrics.compute(training=True)

        return {'loss': loss, 'progress_bar': progress_bar, }

    def validation_step(self, batch, batch_nb):
        X, y, weights = batch
        Y_hat = self._model.forward(X)
        loss = self._model.loss(Y_hat, y, weights)
        self.metrics.update(Y_hat, y, training=False)
        return {"val_loss": loss}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean().item()
        logs = self.metrics.compute(training=True)
        logs.update({"loss": avg_loss})
        self.metrics.reset(training=True)

        return {"loss": avg_loss, "progress_bar": logs, "log": logs, }

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean().item()
        logs = self.metrics.compute(training=False)
        logs.update({"val_loss": avg_loss})

        results = {"progress_bar": logs,
                   "log": logs}
        self.metrics.reset(training=False)
        print(logs)  # print val results every epoch
        return results

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self._model.parameters(),
                                     lr=self._model.hparams.lr,
                                     weight_decay=self._model.hparams.nb_weight_decay
                                     )
        return optimizer

    # def configure_ddp(self, model, device_ids):
    #     model = LightningDistributedDataParallel(
    #         model,
    #         device_ids=device_ids,
    #         find_unused_parameters=True
    #     )
    #     return model

    # def prepare_data(self) -> None:
    #     with open(self.data_path, 'rb') as file:
    #         network = pickle.load(file)
    #     variables = []
    #     targets = ['go_id']
    #     network.process_feature_tranformer(filter_label=targets[0], min_count=100, verbose=False)
    #     classes = network.feature_transformer[targets[0]].classes_
    #     self.n_classes = len(classes)
    #     batch_size = 1000
    #     max_length = 1000
    #     test_frac = 0.05
    #     n_steps = int(400000 / batch_size)
    #     directed = False
    #     seed = random.randint(0, 1000)
    #
    #     self.dataset_train = network.get_train_generator(
    #         SubgraphGenerator, variables=variables, targets=targets,
    #         sampling="bfs", batch_size=batch_size, agg_mode=None,
    #         method="GAT", adj_output="coo",
    #         compression="log", n_steps=n_steps, directed=directed,
    #         maxlen=max_length, padding='post', truncating='post', variable_length=False,
    #         seed=seed, verbose=False)
    #
    #     self.dataset_test = network.get_test_generator(
    #         SubgraphGenerator, variables=variables, targets=targets,
    #         sampling='all', batch_size=batch_size, agg_mode=None,
    #         method="GAT", adj_output="coo",
    #         compression="log", n_steps=1, directed=directed,
    #         maxlen=max_length, padding='post', truncating='post', variable_length=False,
    #         seed=seed, verbose=False)
    #
    #     self.vocab = self.dataset_train.tokenizer.word_index
    #
    # def train_dataloader(self):
    #     return torch.utils.data.DataLoader(
    #         self.dataset_train,
    #         batch_size=None,
    #         num_workers=10
    #     )
    #
    # def val_dataloader(self):
    #     return torch.utils.data.DataLoader(
    #         self.dataset_test,
    #         batch_size=None,
    #         num_workers=2
    #     )
