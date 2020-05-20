import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .metrics import Metrics


class ModelTrainer(pl.LightningModule):
    def __init__(self, model: torch.nn.Module):
        super(ModelTrainer, self).__init__()

        self._model = model
        self.hparams = self._model.hparams
        self.metrics = Metrics(loss_type=self.hparams.loss_type)

    def forward(self, X):
        return self._model.forward(X)

    def training_step(self, batch, batch_nb):
        X, y, weights = batch

        Y_hat = self.forward(X)
        loss = self._model.loss(Y_hat, y, weights)

        self.metrics.update_metrics(Y_hat, y, training=True)

        logs = self.metrics.compute_metrics(training=True)
        logs = _fix_dp_return_type(logs, device=Y_hat.device)

        return {'loss': loss, 'progress_bar': logs, }

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean().item()
        logs = self.metrics.compute_metrics(training=True)

        logs = _fix_dp_return_type(logs, device=outputs[0]["loss"].device)
        logs.update({"loss": avg_loss})
        self.metrics.reset_metrics(training=True)

        return {"log": logs}

    # def training_step_end(self, batch_parts_outputs):
    #     outputs = torch.cat(batch_parts_outputs, dim=1)
    #     return outputs

    def validation_step(self, batch, batch_nb):
        X, y, weights = batch
        Y_hat = self._model.forward(X)
        loss = self._model.loss(Y_hat, y, weights)
        self.metrics.update_metrics(Y_hat, y, training=False)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.cat([x["val_loss"] for x in outputs]).mean().item()

        logs = self.metrics.compute_metrics(training=False)
        logs = _fix_dp_return_type(logs, device=outputs[0]["val_loss"].device)
        self.metrics.reset_metrics(training=False)

        logs.update({"val_loss": avg_loss})
        results = {"progress_bar": logs, "log": logs}

        print_logs(logs)
        return results

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self._model.parameters(),
                                     lr=self.hparams.lr,
                                     weight_decay=self.hparams.weight_decay
                                     )

        scheduler = ReduceLROnPlateau(optimizer, )

        return [optimizer], [scheduler]

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


def _fix_dp_return_type(result, device):
    if isinstance(result, torch.Tensor):
        return result.to(device)
    if isinstance(result, dict):
        return {k: _fix_dp_return_type(v, device) for k, v in result.items()}
    # Must be a number then
    return torch.Tensor([result]).to(device)


def print_logs(logs):
    print({key: f"{item.item():.3f}" if isinstance(item, torch.Tensor) else f"{item:.3f}" for key, item in
           logs.items()})
