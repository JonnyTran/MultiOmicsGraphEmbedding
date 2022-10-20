import logging
import math
import traceback
from argparse import Namespace
from typing import Dict, Iterable, Union, Tuple, Any, List

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch_sparse.sample
import tqdm
from fairscale.nn import auto_wrap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.multiclass import OneVsRestClassifier
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch_geometric.nn import MetaPath2Vec as Metapath2vec

from moge.dataset.PyG.hetero_generator import HeteroNodeClfDataset
from moge.dataset.graph import HeteroGraphDataset
from moge.model.PyG.conv import HGT
from moge.model.PyG.latte import LATTE
from moge.model.PyG.latte_flat import LATTE as LATTE_Flat
from moge.model.PyG.link_pred import LATTEFlatLinkPred
from moge.model.PyG.relations import RelationAttention
from moge.model.classifier import DenseClassification, LabelGraphNodeClassifier, LabelNodeClassifer
from moge.model.encoder import HeteroSequenceEncoder, HeteroNodeFeatureEncoder
from moge.model.losses import ClassificationLoss
from moge.model.metrics import Metrics
from moge.model.trainer import NodeClfTrainer, print_pred_class_counts
from moge.model.utils import filter_samples_weights, stack_tensor_dicts, activation, concat_dict_batch, to_device


class LATTENodeClf(NodeClfTrainer):
    def __init__(self, hparams, dataset: HeteroGraphDataset, metrics=["accuracy"],
                 collate_fn="neighbor_sampler") -> None:
        if not isinstance(hparams, Namespace) and isinstance(hparams, dict):
            hparams = Namespace(**hparams)
        super().__init__(hparams=hparams, dataset=dataset, metrics=metrics)
        self.head_node_type = dataset.head_node_type
        self.node_types = list(dataset.num_nodes_dict.keys())
        self.dataset = dataset
        self.multilabel = dataset.multilabel
        self.y_types = list(dataset.y_dict.keys())
        self._name = f"LATTE-{hparams.t_order}"
        self.collate_fn = collate_fn

        # Node attr input
        if hasattr(dataset, 'seq_tokenizer'):
            self.seq_encoder = HeteroSequenceEncoder(hparams, dataset)

        if not hasattr(self, "seq_encoder") or len(self.seq_encoder.seq_encoders.keys()) < len(self.node_types):
            self.encoder = HeteroNodeFeatureEncoder(hparams, dataset)

        self.embedder = LATTE(n_layers=hparams.n_layers,
                              t_order=min(hparams.t_order, hparams.n_layers),
                              embedding_dim=hparams.embedding_dim,
                              num_nodes_dict=dataset.num_nodes_dict,
                              metapaths=dataset.get_metapaths(k_hop=True if "khop" in collate_fn else None),
                              activation=hparams.activation,
                              attn_heads=hparams.attn_heads,
                              attn_activation=hparams.attn_activation,
                              attn_dropout=hparams.attn_dropout,
                              layer_pooling=hparams.layer_pooling if 'layer_pooling' in hparams else 'last',
                              edge_sampling=hparams.edge_sampling if hasattr(hparams, "edge_sampling") else False,
                              hparams=hparams)

        if hparams.layer_pooling == "concat":
            hparams.embedding_dim = hparams.embedding_dim * hparams.n_layers
            logging.info("embedding_dim {}".format(hparams.embedding_dim))
        elif hparams.layer_pooling == "order_concat":
            hparams.embedding_dim = hparams.embedding_dim * self.embedder.layers[-1].t_order
        else:
            assert "concat" not in hparams.layer_pooling, "Layer pooling cannot be `concat` or `rel_concat` when output of network is a GNN"

        if "cls_graph" in hparams and hparams.cls_graph:
            self.classifier = LabelGraphNodeClassifier(dataset, hparams)
        else:
            self.classifier = DenseClassification(hparams)

        self.criterion = ClassificationLoss(
            loss_type=hparams.loss_type, n_classes=dataset.n_classes,
            class_weight=dataset.class_weight if hasattr(dataset, "class_weight") and \
                                                 'use_class_weights' in hparams and hparams.use_class_weights else None,
            pos_weight=dataset.pos_weight if hasattr(dataset, "pos_weight") and
                                             'use_pos_weights' in hparams and hparams.use_pos_weights else None,
            multilabel=dataset.multilabel,
            reduction="mean" if "reduction" not in hparams else hparams.reduction)

        self.hparams.n_params = self.get_n_params()
        self.lr = self.hparams.lr

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')

        if hasattr(self, "feature_projection"):
            for ntype in self.feature_projection:
                nn.init.xavier_normal_(self.feature_projection[ntype].weight, gain=gain)

    def forward(self, inputs: Dict[str, Union[Tensor, Dict[str, Tensor]]], return_embeddings=False, **kwargs):
        if not self.training:
            self._node_ids = inputs["global_node_index"]

        h_out = {}
        if 'sequences' in inputs and hasattr(self, "seq_encoder"):
            h_out.update(self.seq_encoder.forward(inputs['sequences'],
                                                  split_batch_size=math.sqrt(self.hparams.batch_size // 4)))

        input_nodes = inputs["global_node_index"][0]
        if len(h_out) < len(input_nodes.keys()):
            embs = self.encoder.forward(inputs["x_dict"], global_node_index=input_nodes)
            h_out.update({ntype: emb for ntype, emb in embs.items() if ntype not in h_out})

        h_out, edge_pred_dict = self.embedder.forward(h_out,
                                                      inputs["edge_index"],
                                                      inputs["sizes"],
                                                      inputs["global_node_index"], **kwargs)

        if isinstance(self.head_node_type, str):
            logits = self.classifier(h_out[self.head_node_type]) \
                if hasattr(self, "classifier") else h_out[self.head_node_type]

        elif isinstance(self.head_node_type, Iterable):
            if hasattr(self, "classifier"):
                logits = {ntype: self.classifier(emb) for ntype, emb in h_out.items()}
            else:
                logits = h_out

        if return_embeddings:
            return h_out, logits
        else:
            return logits

    def on_validation_epoch_start(self) -> None:
        for l in range(self.embedder.n_layers):
            self.embedder.layers[l].reset()
        super().on_validation_epoch_start()

    def on_test_epoch_start(self) -> None:
        for l in range(self.embedder.n_layers):
            self.embedder.layers[l].reset()
        super().on_test_epoch_start()

    def on_predict_epoch_start(self) -> None:
        for l in range(self.embedder.n_layers):
            self.embedder.layers[l].reset()
        super().on_predict_epoch_start()

    def training_step(self, batch, batch_nb):
        X, y_true, weights = batch
        scores = self.forward(X)

        scores, y_true, weights = stack_tensor_dicts(scores, y_true, weights)
        scores, y_true, weights = filter_samples_weights(y_pred=scores, y_true=y_true, weights=weights)
        if y_true.size(0) == 0: return torch.tensor(0.0, requires_grad=True)

        loss = self.criterion.forward(scores, y_true, weights=weights)
        self.update_node_clf_metrics(self.train_metrics, scores, y_true, weights)

        if batch_nb % 100 == 0 and isinstance(self.train_metrics, Metrics):
            logs = self.train_metrics.compute_metrics()
        else:
            logs = {}

        self.log("loss", loss, logger=True, on_step=True)
        self.log_dict(logs, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_nb):
        X, y_true, weights = batch
        y_pred = self.forward(X, save_betas=False)

        y_pred, y_true, weights = stack_tensor_dicts(y_pred, y_true, weights)
        y_pred, y_true, weights = filter_samples_weights(y_pred=y_pred, y_true=y_true, weights=weights)
        if y_true.size(0) == 0: return torch.tensor(0.0, requires_grad=True)

        val_loss = self.criterion.forward(y_pred, y_true, weights=weights)
        self.update_node_clf_metrics(self.valid_metrics, y_pred, y_true, weights)

        self.log("val_loss", val_loss)

        return val_loss

    def test_step(self, batch, batch_nb):
        X, y_true, weights = batch
        y_pred = self.forward(X, save_betas=True)

        y_pred, y_true, weights = stack_tensor_dicts(y_pred, y_true, weights)
        y_pred, y_true, weights = filter_samples_weights(y_pred=y_pred, y_true=y_true, weights=weights)
        if y_true.size(0) == 0: return torch.tensor(0.0, requires_grad=True)

        test_loss = self.criterion(y_pred, y_true, weights=weights)

        if batch_nb == 0:
            print_pred_class_counts(y_pred, y_true, multilabel=self.dataset.multilabel)

        self.update_node_clf_metrics(self.test_metrics, y_pred, y_true, weights)
        self.log("test_loss", test_loss)

        return test_loss

    def predict_step(self, batch, batch_idx: int, dataloader_idx=None):
        X, y_true, weights = batch
        y_pred = self.forward(X, save_betas=True)

        predict_loss = self.criterion(y_pred, y_true)
        self.test_metrics.update_metrics(y_pred, y_true)

        self.log("predict_loss", predict_loss)

        return predict_loss

    @torch.no_grad()
    def predict(self, dataloader: DataLoader, node_names: pd.Index = None, save_betas=False, **kwargs) \
            -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, np.ndarray]]:
        """

        Args:
            dataloader ():
            node_names (): an pd.Index or np.ndarray to map node indices to node names for `head_node_type`.
            **kwargs ():

        Returns:
            targets, scores, embeddings, ntype_nids
        """
        if isinstance(self, RelationAttention):
            self.reset()

        y_trues = []
        y_preds = []
        embs = []
        nids = []

        for batch in tqdm.tqdm(dataloader, desc='Predict dataloader'):
            X, y_true, weights = to_device(batch, device=self.device)
            h_dict, logits = self.forward(X, save_betas=save_betas, return_embeddings=True)
            y_pred = activation(logits, loss_type=self.hparams['loss_type'])

            y_pred, y_true, weights = concat_dict_batch(X['batch_size'], y_pred, y_true, weights)
            y_pred, y_true, weights = stack_tensor_dicts(y_pred, y_true, weights)
            mask = filter_samples_weights(y_pred=y_pred, y_true=y_true, weights=weights, return_index=True)

            y_true = y_true[mask]
            y_pred = y_pred[mask]
            emb: Tensor = h_dict[self.head_node_type][mask]

            global_node_index = X["global_node_index"][-1] \
                if isinstance(X["global_node_index"], (list, tuple)) else X["global_node_index"]
            nid = global_node_index[self.head_node_type][mask]

            # Convert all to CPU device
            y_trues.append(y_true.cpu().numpy())
            y_preds.append(y_pred.cpu().numpy())
            embs.append(emb.cpu().numpy())
            nids.append(nid.cpu().numpy())

        targets = np.concatenate(y_trues, axis=0)
        scores = np.concatenate(y_preds, axis=0)
        embeddings = np.concatenate(embs, axis=0)
        node_ids = np.concatenate(nids, axis=0)

        if node_names is not None:
            index = node_names[node_ids]
        else:
            index = pd.Index(node_ids, name="nid")

        targets = pd.DataFrame(targets, index=index, columns=self.dataset.classes)
        scores = pd.DataFrame(scores, index=index, columns=self.dataset.classes)
        embeddings = pd.DataFrame(embeddings, index=index)
        ntype_nids = {self.head_node_type: node_ids}

        return targets, scores, embeddings, ntype_nids

    def on_test_end(self):
        try:
            if self.wandb_experiment is not None:
                dataloader = self.test_dataloader()
                y_true, scores, embeddings, global_node_index = self.predict(dataloader, save_betas=False)

                if hasattr(self.dataset, "nodes_namespace"):
                    y_true_dict = self.dataset.split_array_by_namespace(y_true, axis=1)
                    y_pred_dict = self.dataset.split_array_by_namespace(scores, axis=1)

                    for namespace in y_true_dict.keys():
                        if self.head_node_type in self.dataset.nodes_namespace:
                            # nids = global_node_index[self.head_node_type]
                            nids = y_true_dict[namespace].index
                            split_samples = self.dataset.nodes_namespace[self.head_node_type].iloc[nids]
                            title = f"{namespace}_PR_Curve_{split_samples.name}"
                        else:
                            split_samples = None
                            title = f"{namespace}_PR_Curve"

                        self.plot_pr_curve(targets=y_true_dict[namespace], scores=y_pred_dict[namespace],
                                           split_samples=split_samples, title=title)

                self.plot_embeddings_tsne(global_node_index=global_node_index,
                                          embeddings={self.head_node_type: embeddings},
                                          targets=y_true, y_pred=scores)
                self.cleanup_artifacts()

        except Exception as e:
            traceback.print_exc()

        finally:
            super().on_test_end()


class LATTEFlatNodeClf(LATTENodeClf):
    dataset: HeteroNodeClfDataset

    def __init__(self, hparams, dataset: HeteroNodeClfDataset, metrics=["accuracy"], collate_fn=None) -> None:
        if not isinstance(hparams, Namespace) and isinstance(hparams, dict):
            hparams = Namespace(**hparams)
        super(LATTENodeClf, self).__init__(hparams=hparams, dataset=dataset, metrics=metrics)
        self.head_node_type = dataset.head_node_type
        self.node_types = dataset.node_types
        self.dataset = dataset
        self.multilabel = dataset.multilabel
        self._name = f"LATTE-{hparams.n_layers}-{hparams.t_order}"
        self.collate_fn = collate_fn

        # Node attr input
        if hasattr(dataset, 'seq_tokenizer'):
            self.seq_encoder = HeteroSequenceEncoder(hparams, dataset)

        if not hasattr(self, "seq_encoder") or len(self.seq_encoder.seq_encoders.keys()) < len(self.node_types):
            self.encoder = HeteroNodeFeatureEncoder(hparams, dataset)

        if dataset.pred_ntypes is not None:
            hparams.pred_ntypes = dataset.pred_ntypes

        self.embedder = LATTE_Flat(n_layers=hparams.n_layers,
                                   t_order=hparams.t_order,
                                   embedding_dim=hparams.embedding_dim,
                                   num_nodes_dict=dataset.num_nodes_dict,
                                   metapaths=dataset.get_metapaths(),
                                   layer_pooling=hparams.layer_pooling,
                                   activation=hparams.activation,
                                   attn_heads=hparams.attn_heads,
                                   attn_activation=hparams.attn_activation,
                                   attn_dropout=hparams.attn_dropout,
                                   edge_sampling=hparams.edge_sampling if hasattr(hparams, "edge_sampling") else False,
                                   hparams=hparams)

        # Output layer
        if self.embedder.layer_pooling == 'concat':
            hparams.embedding_dim = hparams.embedding_dim * hparams.n_layers
        if "cls_graph" in hparams and hparams.cls_graph is not None:
            self.classifier = LabelGraphNodeClassifier(dataset, hparams)
        elif dataset.pred_ntypes is not None and dataset.class_indices:
            self.classifier = LabelNodeClassifer(dataset, hparams)
        else:
            self.classifier = DenseClassification(hparams)

        self.criterion = ClassificationLoss(
            loss_type=hparams.loss_type, n_classes=dataset.n_classes,
            class_weight=dataset.class_weight if hasattr(dataset, "class_weight") and \
                                                 'use_class_weights' in hparams and hparams.use_class_weights else None,
            pos_weight=dataset.pos_weight if hasattr(dataset, "pos_weight") and
                                             'use_pos_weights' in hparams and hparams.use_pos_weights else None,
            multilabel=dataset.multilabel,
            reduction=hparams.reduction if "reduction" in hparams else "mean")

    def configure_sharded_model(self):
        # modules are sharded across processes
        # as soon as they are wrapped with ``wrap`` or ``auto_wrap``.
        # During the forward/backward passes, weights get synced across processes
        # and de-allocated once computation is complete, saving memory.

        # Wraps the layer in a Fully Sharded Wrapper automatically
        if hasattr(self, "seq_encoder"):
            self.seq_encoder = auto_wrap(self.seq_encoder)
        if hasattr(self, "encoder"):
            self.encoder = auto_wrap(self.encoder)

    def forward(self, inputs: Dict[str, Union[Tensor, Dict[Union[str, Tuple[str]], Union[Tensor, int]]]],
                return_embeddings=False, **kwargs) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        if not self.training:
            self._node_ids = inputs["global_node_index"]

        h_out = {}
        if 'sequences' in inputs and hasattr(self, "seq_encoder"):
            h_out.update(self.seq_encoder.forward(inputs['sequences'],
                                                  split_batch_size=math.sqrt(self.hparams.batch_size // 4)))

        if len(h_out) < len(inputs["global_node_index"].keys()):
            embs = self.encoder.forward(inputs["x_dict"], global_node_index=inputs["global_node_index"])
            h_out.update({ntype: emb for ntype, emb in embs.items() if ntype not in h_out})

        h_out = self.embedder.forward(h_out,
                                      edge_index_dict=inputs["edge_index_dict"],
                                      global_node_index=inputs["global_node_index"],
                                      batch_sizes=inputs["batch_size"],
                                      **kwargs)

        if hasattr(self, "classifier"):
            head_ntype_embeddings = h_out[self.head_node_type]
            if "batch_size" in inputs and self.head_node_type in inputs["batch_size"]:
                head_ntype_embeddings = head_ntype_embeddings[:inputs["batch_size"][self.head_node_type]]

            logits = self.classifier.forward(head_ntype_embeddings, h_dict=h_out)
        else:
            logits = h_out[self.head_node_type]

        if return_embeddings:
            return h_out, logits
        else:
            return logits

    def training_step(self, batch, batch_nb):
        X, y_true, weights = batch
        y_pred = self.forward(X)

        y_pred, y_true, weights = concat_dict_batch(X['batch_size'], y_pred, y_true, weights)
        y_pred, y_true, weights = filter_samples_weights(y_pred=y_pred, y_true=y_true, weights=weights)
        if y_true.size(0) == 0:
            return torch.tensor(0.0, requires_grad=False)

        loss = self.criterion.forward(y_pred, y_true, weights=weights)

        self.update_node_clf_metrics(self.train_metrics, y_pred, y_true, weights)

        self.log("loss", loss, prog_bar=True, logger=True, on_step=True)

        return loss

    def validation_step(self, batch, batch_nb):
        X, y_true, weights = batch
        y_pred = self.forward(X)

        y_pred, y_true, weights = concat_dict_batch(X['batch_size'], y_pred, y_true, weights)
        y_pred, y_true, weights = filter_samples_weights(y_pred=y_pred, y_true=y_true, weights=weights)
        if y_true.size(0) == 0:
            return torch.tensor(0.0, requires_grad=False)

        val_loss = self.criterion.forward(y_pred, y_true, weights=weights)

        self.update_node_clf_metrics(self.valid_metrics, y_pred, y_true, weights)

        self.log("val_loss", val_loss, )

        return val_loss

    def test_step(self, batch, batch_nb):
        X, y_true, weights = batch
        y_pred = self.forward(X, save_betas=batch_nb == 0)

        y_pred, y_true, weights = concat_dict_batch(X['batch_size'], y_pred, y_true, weights)
        y_pred, y_true, weights = filter_samples_weights(y_pred=y_pred, y_true=y_true, weights=weights)
        if y_true.size(0) == 0:
            return torch.tensor(0.0, requires_grad=False)

        test_loss = self.criterion(y_pred, y_true, weights=weights)

        if batch_nb == 0:
            print_pred_class_counts(y_pred, y_true, multilabel=self.dataset.multilabel)

        self.update_node_clf_metrics(self.test_metrics, y_pred, y_true, weights)

        self.log("test_loss", test_loss)

        return test_loss


class MetaPath2Vec(Metapath2vec, pl.LightningModule):
    def __init__(self, hparams, dataset: HeteroGraphDataset, metrics=None):
        # Hparams
        self.batch_size = hparams.batch_size
        self.sparse = hparams.sparse

        embedding_dim = hparams.embedding_dim
        walk_length = hparams.walk_length
        context_size = hparams.context_size
        walks_per_node = hparams.walks_per_node
        num_negative_samples = hparams.num_negative_samples

        # Dataset
        self.dataset = dataset

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

        num_nodes_dict = dataset.get_num_nodes_dict(dataset.edge_index_dict)

        super(MetaPath2Vec, self).__init__(edge_index_dict=edge_index_dict, embedding_dim=embedding_dim,
                                           metapath=metapaths, walk_length=walk_length, context_size=context_size,
                                           walks_per_node=walks_per_node,
                                           num_negative_samples=num_negative_samples, num_nodes_dict=num_nodes_dict,
                                           sparse=hparams.sparse)

        hparams.name = self.name()
        hparams.n_params = self.get_n_params()
        hparams.inductive = dataset.inductive
        self._set_hparams(hparams)

    def get_n_params(self):
        size = 0
        for name, param in dict(self.named_parameters()).items():
            nn = 1
            for s in list(param.size()):
                nn = nn * s
            size += nn
        return size

    def name(self):
        if hasattr(self, "_name"):
            return self._name
        else:
            return self.__class__.__name__

    def forward(self, node_type, batch=None):
        return super().forward(node_type, batch=batch)

    def training_step(self, batch, batch_nb):

        pos_rw, neg_rw = batch
        loss = self.loss(pos_rw, neg_rw)
        self.log("loss", loss, logger=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_nb):
        pos_rw, neg_rw = batch
        val_loss = self.loss(pos_rw, neg_rw)
        self.log("val_loss", val_loss, on_step=True)
        return val_loss

    def test_step(self, batch, batch_nb):
        pos_rw, neg_rw = batch
        test_loss = self.loss(pos_rw, neg_rw)
        self.log("test_loss", test_loss, on_step=True)
        return test_loss

    def training_epoch_end(self, outputs):
        if self.current_epoch % 10 == 0:
            results = self.classification_results(training=True)
        else:
            results = {}

        logs = results
        self.log_dict(logs, prog_bar=True)
        return None

    def validation_epoch_end(self, outputs):
        logs = {}
        if self.current_epoch % 5 == 0:
            logs.update({"val_" + k: v for k, v in self.classification_results(training=False).items()})
        self.log_dict(logs, prog_bar=True)
        return None

    def test_epoch_end(self, outputs):
        logs = {}
        logs.update({"test_" + k: v for k, v in self.classification_results(training=False, testing=True).items()})
        self.log_dict(logs, prog_bar=True)
        return None

    def sample(self, src: torch_sparse.SparseTensor, num_neighbors: int,
               subset=None) -> torch.Tensor:

        rowptr, col, _ = src.csr()
        rowcount = src.storage.rowcount()

        if subset is not None:
            rowcount = rowcount[subset]
            rowptr = rowptr[subset]

        rand = torch.rand((rowcount.size(0), num_neighbors), device=col.device)
        rand.mul_(rowcount.to(rand.dtype).view(-1, 1))
        rand = rand.to(torch.long)
        rand.add_(rowptr.view(-1, 1))

        rand = torch.clamp(rand, min=0, max=min(col.size(0) - 1, rand.squeeze(-1).max()))

        return col[rand]

    def pos_sample(self, batch):
        batch = batch.repeat(self.walks_per_node)

        rws = [batch]
        for i in range(self.walk_length):
            keys = self.metapath[i % len(self.metapath)]
            adj = self.adj_dict[keys]
            batch = self.sample(adj, num_neighbors=1, subset=batch).squeeze()
            rws.append(batch)

        rw = torch.stack(rws, dim=-1)
        rw.add_(self.offset.view(1, -1))

        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.context_size])
        return torch.cat(walks, dim=0)

    def neg_sample(self, batch):
        batch = batch.repeat(self.walks_per_node * self.num_negative_samples)

        rws = [batch]
        for i in range(self.walk_length):
            keys = self.metapath[i % len(self.metapath)]
            batch = torch.randint(0, self.num_nodes_dict[keys[-1]],
                                  (batch.size(0),), dtype=torch.long)
            rws.append(batch)

        rw = torch.stack(rws, dim=-1)
        rw.add_(self.offset.view(1, -1))

        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.context_size])
        return torch.cat(walks, dim=0)

    def collate_fn(self, batch):
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch)
        return self.pos_sample(batch), self.neg_sample(batch)

    def train_dataloader(self, ):
        loader = torch.utils.data.DataLoader(range(self.dataset.num_nodes_dict[self.dataset.head_node_type]),
                                             batch_size=self.hparams.batch_size,
                                             shuffle=True, num_workers=0,
                                             collate_fn=self.collate_fn,
                                             )
        return loader

    def val_dataloader(self, ):
        loader = torch.utils.data.DataLoader(self.dataset.validation_idx,
                                             batch_size=self.hparams.batch_size,
                                             shuffle=False, num_workers=0,
                                             collate_fn=self.collate_fn, )
        return loader

    def test_dataloader(self, ):
        loader = torch.utils.data.DataLoader(self.dataset.testing_idx,
                                             batch_size=self.hparams.batch_size,
                                             shuffle=False, num_workers=0,
                                             collate_fn=self.collate_fn, )
        return loader

    def classification_results(self, training=True, testing=False):
        if training:
            z = self.forward(self.head_node_type,
                             batch=self.dataset.training_idx)
            y = self.dataset.y_dict[self.head_node_type][self.dataset.training_idx]

            perm = torch.randperm(z.size(0))
            train_perm = perm[:int(z.size(0) * self.dataset.train_ratio)]
            test_perm = perm[int(z.size(0) * self.dataset.train_ratio):]

            if y.dim() > 1 and y.size(1) > 1:
                multilabel = True
                clf = OneVsRestClassifier(LogisticRegression(solver="lbfgs", multi_class="auto", max_iter=150))
            else:
                multilabel = False
                clf = LogisticRegression(solver="lbfgs", multi_class="auto", max_iter=150)

            clf.fit(z[train_perm].detach().cpu().numpy(),
                    y[train_perm].detach().cpu().numpy())

            y_pred = clf.predict(z[test_perm].detach().cpu().numpy())
            y_test = y[test_perm].detach().cpu().numpy()
        else:
            z_train = self.forward(self.head_node_type,
                                   batch=self.dataset.training_idx)
            y_train = self.dataset.y_dict[self.head_node_type][self.dataset.training_idx]

            z = self.forward(self.head_node_type,
                             batch=self.dataset.validation_idx if not testing else self.dataset.testing_idx)
            y = self.dataset.y_dict[self.head_node_type][
                self.dataset.validation_idx if not testing else self.dataset.testing_idx]

            if y_train.dim() > 1 and y_train.size(1) > 1:
                multilabel = True
                clf = OneVsRestClassifier(LogisticRegression(solver="lbfgs", multi_class="auto", max_iter=150))
            else:
                multilabel = False
                clf = LogisticRegression(solver="lbfgs", multi_class="auto", max_iter=150)

            clf.fit(z_train.detach().cpu().numpy(), y_train.detach().cpu().numpy())
            y_pred = clf.predict(z.detach().cpu().numpy())
            y_test = y.detach().cpu().numpy()

        result = dict(zip(["precision", "recall", "f1"],
                          precision_recall_fscore_support(y_test, y_pred, average="micro")))
        result["acc" if not multilabel else "accuracy"] = result["precision"]
        return result

    def configure_optimizers(self):
        if self.sparse:
            optimizer = torch.optim.SparseAdam(self.parameters(), lr=self.hparams.lr)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

        return {"optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "val_loss"}


class LATTEFlatNodeClfLink(LATTEFlatLinkPred):
    dataset: HeteroNodeClfDataset
    def __init__(self, hparams: Namespace, dataset: HeteroNodeClfDataset,
                 metrics: Dict[str, List[str]] = ["obgn-mag"], collate_fn=None) -> None:
        dataset.pred_metapaths = [("_N", "_E", "_N")]
        dataset.go_ntype = "GO_term"
        super().__init__(hparams, dataset, metrics, collate_fn)

    def forward(self, inputs: Dict[str, Any], **kwargs):
        embeddings = super().forward(inputs, edges_true=None, return_embeddings=True, **kwargs)

        logits = embeddings[self.head_node_type] @ embeddings[self.dataset.go_ntype].T
        return logits

    def training_step(self, batch, batch_nb):
        X, y_true, weights = batch
        y_pred = self.forward(X)

        # y_pred, y_true, weights = process_tensor_dicts(y_pred, y_true, weights)
        y_pred, y_true, weights = concat_dict_batch(X['batch_size'], y_pred, y_true, weights)
        y_pred, y_true, weights = filter_samples_weights(y_pred=y_pred, y_true=y_true, weights=weights)
        if y_true.size(0) == 0: return torch.tensor(0.0, requires_grad=False)

        loss = self.criterion.forward(y_pred, y_true, neg_edges=weights)

        self.update_node_clf_metrics(self.train_metrics, y_pred, y_true, weights)

        self.log("loss", loss, logger=True, on_step=True)

        return loss

    def validation_step(self, batch, batch_nb):
        X, y_true, weights = batch
        y_pred = self.forward(X)

        y_pred, y_true, weights = concat_dict_batch(X['batch_size'], y_pred, y_true, weights)
        y_pred, y_true, weights = filter_samples_weights(y_pred=y_pred, y_true=y_true, weights=weights)
        if y_true.size(0) == 0:
            return torch.tensor(0.0, requires_grad=False)

        val_loss = self.criterion.forward(y_pred, y_true, neg_edges=weights)
        self.update_node_clf_metrics(self.valid_metrics, y_pred, y_true, weights)

        self.log("val_loss", val_loss, )

        return val_loss

    def test_step(self, batch, batch_nb):
        X, y_true, weights = batch
        y_pred = self.forward(X, save_betas=False)

        y_pred, y_true, weights = concat_dict_batch(X['batch_size'], y_pred, y_true, weights)
        y_pred, y_true, weights = filter_samples_weights(y_pred=y_pred, y_true=y_true, weights=weights)
        if y_true.size(0) == 0: return torch.tensor(0.0, requires_grad=False)

        test_loss = self.criterion(y_pred, y_true, weights=weights)

        if batch_nb == 0:
            print_pred_class_counts(y_pred, y_true, multilabel=self.dataset.multilabel)

        self.update_node_clf_metrics(self.test_metrics, y_pred, y_true, weights)

        self.log("test_loss", test_loss)

        return test_loss

    def update_node_clf_metrics(self, metrics: Union[Metrics, Dict[str, Metrics]],
                                y_pred: Tensor, y_true: Tensor, weights: Tensor):
        if isinstance(metrics, dict):
            y_pred_dict = self.dataset.split_array_by_namespace(y_pred)
            y_true_dict = self.dataset.split_array_by_namespace(y_true)

            for namespace in y_true_dict.keys():
                go_type = "BPO" if namespace == 'biological_process' else \
                    "CCO" if namespace == 'cellular_component' else \
                        "MFO" if namespace == 'molecular_function' else namespace
                metrics[go_type].update_metrics(y_pred_dict[namespace], y_true_dict[namespace], weights=weights)

        else:
            metrics.update_metrics(y_pred, y_true, weights=weights)

    def on_validation_end(self) -> None:
        super().on_validation_end()
        if self.current_epoch % 50 == 1:
            self.plot_sankey_flow(layer=-1)

    def on_test_end(self):
        try:
            if self.wandb_experiment is not None:
                X, y, weights = self.dataset.full_batch()
                embs = self.cpu().forward(X, return_embs=True, save_betas=True)

                self.plot_embeddings_tsne(X, embs, weights=weights)
                self.plot_sankey_flow(layer=-1)
                self.cleanup_artifacts()

        except Exception as e:
            traceback.print_exc()

        finally:
            super().on_test_end()


class HGTNodeClf(LATTEFlatNodeClf):
    def __init__(self, hparams, dataset: HeteroNodeClfDataset, metrics=["accuracy"], collate_fn=None) -> None:
        super().__init__(hparams, dataset, metrics)
        self.head_node_type = dataset.head_node_type
        self.dataset = dataset
        self.multilabel = dataset.multilabel
        self._name = f"HGT-{hparams.n_layers}"
        self.collate_fn = collate_fn
        # Node attr input
        if hasattr(dataset, 'seq_tokenizer'):
            self.seq_encoder = HeteroSequenceEncoder(hparams, dataset)

        if not hasattr(self, "seq_encoder") or len(self.seq_encoder.seq_encoders.keys()) < len(dataset.node_types):
            self.encoder = HeteroNodeFeatureEncoder(hparams, dataset)

        self.embedder = HGT(embedding_dim=hparams.embedding_dim, num_layers=hparams.n_layers,
                            num_heads=hparams.attn_heads,
                            node_types=dataset.G.node_types, metadata=dataset.G.metadata())

        # Output layer
        if "cls_graph" in hparams and hparams.cls_graph is not None:
            self.classifier = LabelGraphNodeClassifier(dataset, hparams)

        elif hparams.nb_cls_dense_size >= 0:
            if hparams.layer_pooling == "concat":
                hparams.embedding_dim = hparams.embedding_dim * hparams.t_order
                logging.info("embedding_dim {}".format(hparams.embedding_dim))

            self.classifier = DenseClassification(hparams)
        else:
            assert hparams.layer_pooling != "concat", "Layer pooling cannot be concat when output of network is a GNN"

        self.criterion = ClassificationLoss(loss_type=hparams.loss_type, n_classes=dataset.n_classes,
                                            class_weight=dataset.class_weight if hasattr(dataset, "class_weight") and \
                                                                                 hparams.use_class_weights else None,
                                            multilabel=dataset.multilabel,
                                            reduction="mean" if "reduction" not in hparams else hparams.reduction)

        self.hparams.n_params = self.get_n_params()
        self.lr = self.hparams.lr
