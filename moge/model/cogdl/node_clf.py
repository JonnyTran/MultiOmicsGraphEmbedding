import numpy as np
import torch
from cogdl.models.emb.hin2vec import Hin2vec, RWgraph, Hin2vec_layer
from moge.model.PyG.hgt import HGTModel
from torch.nn import functional as F
from tqdm import tqdm

from moge.dataset.graph import HeteroGraphDataset
from moge.model.classifier import DenseClassification
from moge.model.cogdl.conv import GTN as Gtn, HAN as Han
from moge.model.losses import ClassificationLoss
from moge.model.trainer import NodeClfTrainer
from moge.model.utils import filter_samples


class HGT(HGTModel, NodeClfTrainer):
    def __init__(self, hparams, dataset: HeteroGraphDataset, metrics=["precision"]):
        super(HGT, self).__init__(
            in_dim=dataset.in_features,
            n_hid=hparams.embedding_dim,
            n_layers=hparams.n_layers,
            num_types=len(dataset.node_types),
            num_relations=len(dataset.edge_index_dict),
            n_heads=hparams.attn_heads,
            dropout=hparams.attn_dropout,
            prev_norm=hparams.prev_norm,
            last_norm=hparams.last_norm,
            use_RTE=False, hparams=hparams, dataset=dataset, metrics=metrics)

        self.classifier = DenseClassification(hparams)
        self.criterion = ClassificationLoss(loss_type=hparams.loss_type, n_classes=dataset.n_classes,
                                            class_weight=dataset.class_weight if hasattr(dataset, "class_weight") and \
                                                                                 hparams.use_class_weights else None,
                                            multilabel=dataset.multilabel)

        self.collate_fn = hparams.collate_fn
        self.dataset = dataset
        self.hparams.n_params = self.get_n_params()

    def name(self):
        if hasattr(self, "_name"):
            return self._name
        else:
            return self.__class__.__name__

    def forward(self, X):
        if not self.training:
            self._node_ids = X["global_node_index"]

        return super().forward(node_feature=X["node_inp"],
                               node_type=X["node_type"],
                               edge_time=X["edge_time"],
                               edge_index=X["edge_index"],
                               edge_type=X["edge_type"])

    def training_step(self, batch, batch_nb):
        X, y, weights = batch
        embeddings = self.forward(X)

        nids = (X["node_type"] == int(self.dataset.node_types.index(self.dataset.head_node_type)))
        y_hat = self.classifier.forward(embeddings[nids])

        y_hat, y = filter_samples(Y_hat=y_hat, Y=y, weights=weights)
        self.train_metrics.update_metrics(y_hat, y, weights=None)
        loss = self.criterion(y_hat, y)
        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        X, y, weights = batch
        embeddings = self.forward(X)
        nids = (X["node_type"] == int(self.dataset.node_types.index(self.dataset.head_node_type)))
        y_hat = self.classifier.forward(embeddings[nids])

        y_hat, y = filter_samples(Y_hat=y_hat, Y=y, weights=weights)
        loss = self.criterion(y_hat, y)
        self.valid_metrics.update_metrics(y_hat, y, weights=None)

        return {"val_loss": loss}

    def test_step(self, batch, batch_nb):
        X, y, weights = batch
        embeddings = self.forward(X)

        nids = (X["node_type"] == int(self.dataset.node_types.index(self.dataset.head_node_type)))
        y_hat = self.classifier.forward(embeddings[nids])

        y_hat, y = filter_samples(Y_hat=y_hat, Y=y, weights=weights)
        loss = self.criterion(y_hat, y)
        self.test_metrics.update_metrics(y_hat, y, weights=None)

        return {"test_loss": loss}

    def configure_optimizers(self):
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'alpha_activation', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, eps=1e-06)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                        pct_start=0.05, anneal_strategy='linear',
                                                        final_div_factor=10, max_lr=5e-4,
                                                        epochs=self.hparams.n_epoch,
                                                        steps_per_epoch=int(np.ceil(self.dataset.training_idx.shape[
                                                                                        0] / self.hparams.batch_size)))

        return {"optimizer": optimizer, "scheduler": scheduler, "monitor": "val_loss"}


class GTN(Gtn, NodeClfTrainer):
    def __init__(self, hparams, dataset: HeteroGraphDataset, metrics=["precision"]):
        num_edge = len(dataset.edge_index_dict)
        num_layers = hparams.num_layers
        num_class = dataset.n_classes
        self.multilabel = dataset.multilabel
        self.head_node_type = dataset.head_node_type
        self.collate_fn = hparams.collate_fn
        num_nodes = dataset.num_nodes_dict[dataset.head_node_type]

        if dataset.in_features:
            w_in = dataset.in_features
        else:
            w_in = hparams.embedding_dim

        w_out = hparams.embedding_dim
        num_channels = hparams.num_channels
        super(GTN, self).__init__(num_edge, num_channels, w_in, w_out, num_class, num_nodes, num_layers,
                                  hparams=hparams, dataset=dataset, metrics=metrics)

        if not hasattr(dataset, "x") and not hasattr(dataset, "x_dict"):
            if num_nodes > 10000:
                self.embedding = {self.head_node_type: torch.nn.Embedding(num_embeddings=num_nodes,
                                                                          embedding_dim=hparams.embedding_dim).cpu()}
            else:
                self.embedding = torch.nn.Embedding(num_embeddings=num_nodes, embedding_dim=hparams.embedding_dim)

        self.dataset = dataset
        self.head_node_type = self.dataset.head_node_type
        self.hparams.n_params = self.get_n_params()

    def forward(self, inputs):
        A, X, x_idx = inputs["adj"], inputs["x"], inputs["idx"]

        if not self.training:
            self._node_ids = inputs["global_node_index"]

        if X is None:
            if isinstance(self.embedding, dict):
                X = self.embedding[self.head_node_type].weight[x_idx].to(self.layers[0].device)
            else:
                X = self.embedding.weight[x_idx]

        Ws = []
        for i in range(self.num_layers):
            if i == 0:
                H, W = self.layers[i](A)
            else:
                H = self.normalization(H)
                H, W = self.layers[i](A, H)
            Ws.append(W)
        for i in range(self.num_channels):
            if i == 0:
                edge_index, edge_weight = H[i][0], H[i][1]
                X_ = self.gcn(X, edge_index=edge_index.detach(), edge_weight=edge_weight)
                X_ = F.relu(X_)
            else:
                edge_index, edge_weight = H[i][0], H[i][1]
                X_ = torch.cat((X_, F.relu(self.gcn(X, edge_index=edge_index.detach(), edge_weight=edge_weight))),
                               dim=1)
        X_ = self.embedder(X_)
        X_ = F.relu(X_)
        # X_ = F.dropout(X_, p=0.5)
        if x_idx is not None and X_.size(0) > x_idx.size(0):
            y = self.classifier(X_[x_idx])
        else:
            y = self.classifier(X_)

        return y

    def loss(self, y_hat, y):
        if not self.multilabel:
            loss = self.cross_entropy_loss(y_hat, y)
        else:
            loss = F.binary_cross_entropy_with_logits(y_hat, y.type_as(y_hat))

        return loss

    def training_step(self, batch, batch_nb):
        X, y, weights = batch
        y_hat = self.forward(X)
        y_hat, y = filter_samples(Y_hat=y_hat, Y=y, weights=weights)
        self.train_metrics.update_metrics(y_hat, y, weights=None)
        loss = self.loss(y_hat, y)
        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        X, y, weights = batch

        y_hat = self.forward(X)
        y_hat, y = filter_samples(Y_hat=y_hat, Y=y, weights=weights)
        loss = self.loss(y_hat, y)
        self.valid_metrics.update_metrics(y_hat, y, weights=None)

        return {"val_loss": loss}

    def test_step(self, batch, batch_nb):
        X, y, weights = batch
        y_hat = self.forward(X)
        y_hat, y = filter_samples(Y_hat=y_hat, Y=y, weights=weights)
        loss = self.loss(y_hat, y)
        self.test_metrics.update_metrics(y_hat, y, weights=None)

        return {"test_loss": loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


class HAN(Han, NodeClfTrainer):
    def __init__(self, hparams, dataset: HeteroGraphDataset, metrics=["precision"]):
        num_edge = len(dataset.edge_index_dict)
        num_layers = hparams.num_layers
        num_class = dataset.n_classes
        self.collate_fn = hparams.collate_fn
        self.multilabel = dataset.multilabel
        num_nodes = dataset.num_nodes_dict[dataset.head_node_type]

        if dataset.in_features:
            w_in = dataset.in_features
        else:
            w_in = hparams.embedding_dim

        w_out = hparams.embedding_dim

        super(HAN, self).__init__(num_edge, w_in, w_out, num_class, num_nodes, num_layers,
                                  hparams=hparams, dataset=dataset, metrics=metrics)

        if not hasattr(dataset, "x") and not hasattr(dataset, "x_dict"):
            if num_nodes > 10000:
                self.embedding = {dataset.head_node_type: torch.nn.Embedding(num_embeddings=num_nodes,
                                                                             embedding_dim=hparams.embedding_dim).cpu()}
            else:
                self.embedding = torch.nn.Embedding(num_embeddings=num_nodes, embedding_dim=hparams.embedding_dim)

        self.dataset = dataset
        self.head_node_type = self.dataset.head_node_type
        self.hparams.n_params = self.get_n_params()

    def forward(self, inputs):
        A, X, x_idx = inputs["adj"], inputs["x"], inputs["idx"]

        if not self.training:
            self._node_ids = inputs["global_node_index"]

        if X is None:
            if isinstance(self.embedding, dict):
                X = self.embedding[self.head_node_type].weight[x_idx].to(self.layers[0].device)
            else:
                X = self.embedding.weight[x_idx]

        for i in range(len(self.layers)):
            X = self.layers[i](X, A)

        X = self.embedder(X, A)

        if x_idx is not None and X.size(0) > x_idx.size(0):
            y = self.classifier(X[x_idx])
        else:
            y = self.classifier(X)
        return y

    def loss(self, y_hat, y):
        if not self.multilabel:
            loss = self.cross_entropy_loss(y_hat, y)
        else:
            loss = F.binary_cross_entropy_with_logits(y_hat, y.type_as(y_hat))
        return loss

    def training_step(self, batch, batch_nb):
        X, y, weights = batch
        y_hat = self.forward(X)
        y_hat, y = filter_samples(Y_hat=y_hat, Y=y, weights=weights)
        self.train_metrics.update_metrics(y_hat, y, weights=None)
        loss = self.loss(y_hat, y)
        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        X, y, weights = batch
        y_hat = self.forward(X)
        y_hat, y = filter_samples(Y_hat=y_hat, Y=y, weights=weights)
        self.valid_metrics.update_metrics(y_hat, y, weights=None)
        loss = self.loss(y_hat, y)

        return {"val_loss": loss}

    def test_step(self, batch, batch_nb):
        X, y, weights = batch
        y_hat = self.forward(X)
        y_hat, y = filter_samples(Y_hat=y_hat, Y=y, weights=weights)
        self.test_metrics.update_metrics(y_hat, y, weights=None)
        loss = self.loss(y_hat, y)

        return {"test_loss": loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


class HIN2Vec(Hin2vec):
    def __init__(self, hparams, dataset: HeteroGraphDataset, metrics=None):
        self.train_ratio = hparams.train_ratio
        self.batch_size = hparams.batch_size
        self.sparse = hparams.sparse

        embedding_dim = hparams.embedding_dim
        walk_length = hparams.walk_length
        hop = hparams.context_size
        walks_per_node = hparams.walks_per_node
        num_negative_samples = hparams.num_negative_samples

        # Dataset
        self.dataset = dataset
        num_nodes_dict = None
        metapaths = self.dataset.get_metapaths()
        self.head_node_type = self.dataset.head_node_type
        edge_index_dict = dataset.edge_index_dict
        first_node_type = metapaths[0][0]

        for metapath in reversed(metapaths):
            if metapath[-1] == first_node_type:
                last_metapath = metapath
                break
        metapaths.pop(metapaths.index(last_metapath))
        metapaths.append(last_metapath)
        print("metapaths", metapaths)

        if dataset.use_reverse:
            dataset.add_reverse_edge_index(dataset.edge_index_dict)

        super().__init__(embedding_dim, walk_length, walks_per_node, hparams.batch_size, hop, num_negative_samples,
                         1000, hparams.lr, cpu=True)

    def train(self, G, node_type):
        self.num_node = G.number_of_nodes()
        rw = RWgraph(G, node_type)
        walks = rw._simulate_walks(self.walk_length, self.walk_num)
        pairs, relation = rw.data_preparation(walks, self.hop, self.negative)

        self.num_relation = len(relation)
        model = Hin2vec_layer(self.num_node, self.num_relation, self.hidden_dim, self.cpu)
        self.model = model.to(self.device)

        num_batch = int(len(pairs) / self.batch_size)
        print_num_batch = 100
        print("number of batch", num_batch)

        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        epoch_iter = tqdm(range(self.epoches))
        for epoch in epoch_iter:
            loss_n, pred, label = [], [], []
            for i in range(num_batch):
                batch_pairs = torch.from_numpy(pairs[i * self.batch_size:(i + 1) * self.batch_size])
                batch_pairs = batch_pairs.to(self.device)
                batch_pairs = batch_pairs.T
                x, y, r, l = batch_pairs[0], batch_pairs[1], batch_pairs[2], batch_pairs[3]
                opt.zero_grad()
                logits, loss = self.model.forward(x, y, r, l)

                loss_n.append(loss.item())
                label.append(l)
                pred.extend(logits)
                if i % print_num_batch == 0 and i != 0:
                    label = torch.cat(label).to(self.device)
                    pred = torch.stack(pred, dim=0)
                    pred = pred.max(1)[1]
                    acc = pred.eq(label).sum().item() / len(label)
                    epoch_iter.set_description(
                        f"Epoch: {i:03d}, Loss: {sum(loss_n) / print_num_batch:.5f}, Acc: {acc:.5f}"
                    )
                    loss_n, pred, label = [], [], []

                loss.backward()
                opt.step()

        embedding = self.model.get_emb()
        return embedding.cpu().detach().numpy()

    def get_n_params(self):
        size = 0
        for name, param in dict(self.named_parameters()).items():
            nn = 1
            for s in list(param.size()):
                nn = nn * s
            size += nn
        return size

    def train_dataloader(self):
        return self.dataset.train_dataloader(collate_fn=self.sample, batch_size=self.hparams.batch_size)

    def val_dataloader(self):
        return self.dataset.valid_dataloader(collate_fn=self.sample, batch_size=self.hparams.batch_size)

    def valtrain_dataloader(self):
        return self.dataset.valtrain_dataloader(collate_fn=self.sample,
                                                batch_size=self.hparams.batch_size)

    def test_dataloader(self):
        return self.dataset.test_dataloader(collate_fn=self.sample, batch_size=self.hparams.batch_size)

    def configure_optimizers(self):
        if self.sparse:
            return torch.optim.SparseAdam(self.parameters(), lr=self.hparams.lr)
        else:
            return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
