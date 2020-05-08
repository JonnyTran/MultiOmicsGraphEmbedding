import torch
from torch.nn import functional as F

import pytorch_lightning as pl
from ignite.metrics import Precision, Recall

from transformers import AlbertConfig

from .metrics import TopKMulticlassAccuracy
from .encoder import ConvLSTM, AlbertEncoder
from .embedder import GAT
from .classifier import Dense


class EncoderEmbedderClassifier(pl.LightningModule):
    def __init__(self, hparams):
        super(EncoderEmbedderClassifier, self).__init__()

        if hparams.encoder == "ConvLSTM":
            self._encoder = ConvLSTM(hparams)
        if hparams.encoder == "Albert":
            config = AlbertConfig(
                vocab_size=hparams.vocab_size,
                embedding_size=hparams.word_embedding_size,
                hidden_size=hparams.encoding_dim,
                num_hidden_layers=hparams.num_hidden_layers,
                num_hidden_groups=hparams.num_hidden_groups,
                hidden_dropout_prob=hparams.hidden_dropout_prob,
                attention_probs_dropout_prob=hparams.attention_probs_dropout_prob,
                num_attention_heads=hparams.num_attention_heads,
                intermediate_size=hparams.intermediate_size,
                type_vocab_size=1,
                max_position_embeddings=hparams.max_length,
            )
            self._encoder = AlbertEncoder(config)
        else:
            raise Exception("hparams.encoder must be one of {'ConvLSTM'}")

        if hparams.embedder == "GAT":
            self._embedder = GAT(hparams)
        else:
            raise Exception("hparams.embedder must be one of {'GAT'}")

        if hparams.classifier == "Dense":
            self._classifier = Dense(hparams)
        else:
            raise Exception("hparams.classifier must be one of {'Dense'}")

        self.hparams = hparams

        self.criterion = torch.nn.BCEWithLogitsLoss()

    def forward(self, X):
        input_seqs, subnetwork = X["input_seqs"], X["subnetwork"]

        encodings = self._encoder(input_seqs)
        embeddings = self._embedder(encodings, subnetwork)
        y_pred = self._classifier(embeddings)
        return y_pred

    def loss(self, Y_hat, Y, weights=None):
        Y = Y.type_as(Y_hat)
        idx = torch.nonzero(weights).view(-1)
        Y = Y[idx, :]
        Y_hat = Y_hat[idx, :]

        # return F.binary_cross_entropy(Y_hat, Y, reduction="mean")

        return self.criterion(Y_hat, Y)

    def get_embeddings(self, X, cuda=True):
        """
        Get embeddings for a set of nodes in `X`.
        :param X: a dict with keys {"input_seqs", "subnetwork"}
        :param cuda (bool): whether to run computations in
        :return (np.array): a numpy array of size (node size, embedding dim)
        """
        if not isinstance(X["input_seqs"], torch.Tensor):
            X = {k: torch.tensor(v).cuda() for k, v in X.items()}

        if cuda:
            X = {k: v.cuda() for k, v in X.items()}
        else:
            X = {k: v.cpu() for k, v in X.items()}

        encodings = self._encoder(X["input_seqs"])
        embeddings = self._embedder(encodings, X["subnetwork"])

        return embeddings.detach().cpu().numpy()

    def predict(self, embeddings, cuda=True):
        if not isinstance(embeddings, torch.Tensor):
            embeddings = torch.tensor(embeddings)

        if cuda:
            embeddings = embeddings.cuda()
        else:
            embeddings = embeddings.cpu()

        y_pred = self._classifier(embeddings)
        return y_pred.detach().cpu().numpy()


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
        progress_bar = self.metric_logs(training=True)

        return {'loss': loss, 'progress_bar': progress_bar, }

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        logs = self.metric_logs(training=True)
        logs.update({"loss": avg_loss})
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
        logs = self.metric_logs(training=False)
        logs.update({"val_loss": avg_loss})

        results = {"progress_bar": logs,
                   "log": logs}
        self.reset_metrics(training=False)
        print(logs)  # print val results every epoch
        return results

    def init_metrics(self):
        self.precision = Precision(average=True, is_multilabel=True)
        self.recall = Recall(average=True, is_multilabel=True)
        self.top_k_train = TopKMulticlassAccuracy(k=107)
        self.precision_val = Precision(average=True, is_multilabel=True)
        self.recall_val = Recall(average=True, is_multilabel=True)
        self.top_k_val = TopKMulticlassAccuracy(k=107)

    def update_metrics(self, y_pred: torch.Tensor, y_true: torch.Tensor, training: bool):
        if training:
            self.precision.update(((y_pred > 0.5).type_as(y_true), y_true))
            self.recall.update(((y_pred > 0.5).type_as(y_true), y_true))
            self.top_k_train.update((y_pred, y_true))
        else:
            self.precision_val.update(((y_pred > 0.5).type_as(y_true), y_true))
            self.recall_val.update(((y_pred > 0.5).type_as(y_true), y_true))
            self.top_k_val.update((y_pred, y_true))

    def metric_logs(self, training: bool):
        if training:
            logs = {
                "precision": self.precision.compute(),
                "recall": self.recall.compute(),
                "top_k": self.top_k_train.compute()}
        else:
            logs = {
                "val_precision": self.precision_val.compute(),
                "val_recall": self.recall_val.compute(),
                "val_top_k": self.top_k_val.compute()}
        return logs

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
