import pytorch_lightning as pl
import torch
from ignite.metrics import Precision, Recall

from .metrics import top_k_multiclass, TopKMulticlassAccuracy

class LightningModel(pl.LightningModule):
    def __init__(self, model):
        super(LightningModel, self).__init__()

        self._model = model
        self.init_metrics()

    def forward(self, X):
        return self._model(X)

    def training_step(self, batch, batch_nb):
        X, y, weights = batch

        Y_hat = self.forward(X)
        loss = self._model.loss(Y_hat, y, weights)

        self.update_metrics(Y_hat, y, training=True)
        progress_bar = {
            "precision": self.precision.compute(),
            "recall": self.recall.compute(),
            "top_k": self.top_k_train.compute(),
        }

        return {'loss': loss, 'progress_bar': progress_bar, }

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        logs = {
            "loss": avg_loss,
            "precision": self.precision.compute(),
            "recall": self.recall.compute(),
            "top_k": self.top_k_train.compute(),
        }
        self.reset_metrics(training=True)
        return {"loss": avg_loss, "progress_bar": logs, "log": logs, }

    def validation_step(self, batch, batch_nb):
        X, y, weights = batch
        Y_hat = self._model.forward(X)
        loss = self._model.loss(Y_hat, y, weights)
        self.update_metrics(Y_hat, y, training=False)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        logs = {
            "val_loss": avg_loss,
            "val_precision": self.precision_val.compute(),
            "val_recall": self.recall_val.compute(),
            "val_top_k": self.top_k_val.compute(),
        }

        results = {"progress_bar": logs,
                   "log": logs}
        self.reset_metrics(training=False)
        print(logs)
        return results

    def init_metrics(self):
        self.precision = Precision(average=True, is_multilabel=True)
        self.recall = Recall(average=True, is_multilabel=True)
        self.top_k_train = TopKMulticlassAccuracy(k=25)
        self.precision_val = Precision(average=True, is_multilabel=True)
        self.recall_val = Recall(average=True, is_multilabel=True)
        self.top_k_val = TopKMulticlassAccuracy(k=25)

    def update_metrics(self, y_pred: torch.Tensor, y_true: torch.Tensor, training: bool):
        if training:
            self.precision.update(((y_pred > 0.5).type_as(y_true), y_true))
            self.recall.update(((y_pred > 0.5).type_as(y_true), y_true))
            self.top_k_train.update((y_pred, y_true))
        else:
            self.precision_val.update(((y_pred > 0.5).type_as(y_true), y_true))
            self.recall_val.update(((y_pred > 0.5).type_as(y_true), y_true))
            self.top_k_val.update((y_pred, y_true))


    def reset_metrics(self, training: bool):
        if training:
            self.precision.reset()
            self.recall.reset()
            self.top_k_train.reset()
        else:
            self.precision_val.reset()
            self.recall_val.reset()
            self.top_k_val.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self._model.hparams.lr,
                                     weight_decay=self._model.hparams.nb_weight_decay
                                     )
        return optimizer

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
    #     network.split_stratified(directed=directed, stratify_label=targets[0], stratify_omic=False,
    #                              n_splits=int(1 / test_frac), dropna=True, seed=seed, verbose=False)
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
