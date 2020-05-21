import pytorch_lightning as pl
import torch

from .metrics import Metrics


class ModelTrainer(pl.LightningModule):
    def __init__(self, model: torch.nn.Module, gpus=1, data_path=None):
        super(ModelTrainer, self).__init__()

        self._model = model
        self.hparams = self._model.hparams
        self.metrics = Metrics(loss_type=self.hparams.loss_type)

        self.n_gpus = gpus
        self.data_path = data_path
        if self.data_path is not None:
            self.prepare_data()


    def forward(self, X):
        return self._model.forward(X)

    def training_step(self, batch, batch_nb):
        X, y, weights = batch
        if y.dim() > 2:
            assert y.size(0) == 1
            y = y.squeeze(0)
        if weights.dim() > 2:
            assert weights.size(0) == 1
            weights = weights.squeeze(0)

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
        if y.dim() > 2:
            assert y.size(0) == 1
            y = y.squeeze(0)
        if weights.dim() > 2:
            assert weights.size(0) == 1
            weights = weights.squeeze(0)

        Y_hat = self._model.forward(X)
        loss = self._model.loss(Y_hat, y, weights)

        self.metrics.update_metrics(Y_hat, y, training=False)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.cat([x["val_loss"] for x in outputs]).mean().item()

        logs = self.metrics.compute_metrics(training=False)
        self.metrics.reset_metrics(training=False)
        logs = _fix_dp_return_type(logs, device=outputs[0]["val_loss"].device)
        logs.update({"val_loss": avg_loss})
        print_logs(logs)
        return {"progress_bar": logs, "log": logs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.hparams.lr,
                                     weight_decay=self.hparams.weight_decay)

        # scheduler = ReduceLROnPlateau(optimizer, )
        # return [optimizer], [scheduler]
        return optimizer

    # def prepare_data(self) -> None:
    #     with open(self.data_path, 'rb') as file:
    #         network = pickle.load(file)
    #
    #     variables = []
    #     targets = ['go_id']
    #     network.process_feature_tranformer(filter_label=targets[0], min_count=self.hparams.classes_min_count, verbose=False)
    #     classes = network.feature_transformer[targets[0]].classes_
    #     self.hparams.n_classes = len(classes)
    #     batch_size = 1000
    #     max_length = 1000
    #     n_steps = int(400000 / batch_size)
    #     seed = random.randint(0, 1000)
    #
    #     split_idx = 0
    #     self.generator_train = network.get_train_generator(
    #         MultiplexGenerator, split_idx=split_idx, variables=variables, targets=targets,
    #         traversal=self.hparams.traversal, batch_size=batch_size,
    #         sampling=self.hparams.sampling, n_steps=n_steps,
    #         method="GAT", adj_output="coo",
    #         maxlen=max_length, padding='post', truncating='post',
    #         seed=seed, verbose=False)
    #
    #     self.generator_test = network.get_test_generator(
    #         MultiplexGenerator, split_idx=split_idx, variables=variables, targets=targets,
    #         traversal='all_slices', batch_size=np.ceil(len(network.testing.node_list)*.25).astype(int),
    #         sampling="cycle", n_steps=1,
    #         method="GAT", adj_output="coo",
    #         maxlen=max_length, padding='post', truncating='post',
    #         seed=seed, verbose=False)
    #
    #     self.vocab = self.generator_train.tokenizer.word_index
    #
    # def train_dataloader(self):
    #     if self.gpus == 1 or self.gpus == None:
    #         batch_size = None
    #     else:
    #         batch_size = self.gpus
    #
    #     return torch.utils.data.DataLoader(
    #         self.generator_train,
    #         batch_size=batch_size,
    #         shuffle=False,
    #         num_workers=18,
    #         collate_fn=get_multiplex_collate_fn(node_types=list(self.hparams.encoder.keys()),
    #                                             layers=list(self.hparams.embedder.keys())) if self.gpus > 1 else None
    #     )
    #
    # def val_dataloader(self):
    #     if self.gpus == 1 or self.gpus == None:
    #         batch_size = None
    #     else:
    #         batch_size = self.gpus
    #
    #     return torch.utils.data.DataLoader(
    #         self.generator_test,
    #         batch_size=batch_size,
    #         shuffle=False,
    #         num_workers=4,
    #         collate_fn=get_multiplex_collate_fn(node_types=list(self.hparams.encoder.keys()),
    #                                             layers=list(self.hparams.embedder.keys())) if self.gpus > 1 else None
    #     )


def _fix_dp_return_type(result, device):
    if isinstance(result, torch.Tensor):
        return result.to(device)
    if isinstance(result, dict):
        return {k: _fix_dp_return_type(v, device) for k, v in result.items()}
    # Must be a number then
    return torch.Tensor([result]).to(device)


def print_logs(logs):
    print({key: f"{item.item():.3f}" if isinstance(item, torch.Tensor) \
        else f"{item:.5f}" for key, item in logs.items()})

