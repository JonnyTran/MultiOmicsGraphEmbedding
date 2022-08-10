import itertools
import logging
import os
from collections import defaultdict
from typing import Union, Iterable, Dict, Tuple, List, Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wandb
from pandas import DataFrame, Series
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import WandbLogger
from sklearn.cluster import KMeans
from torch import Tensor
from torch.utils.data.distributed import DistributedSampler

from moge.criterion.clustering import clustering_metrics
from moge.dataset import DGLNodeSampler, HeteroNeighborGenerator
from moge.dataset.graph import HeteroGraphDataset
from moge.dataset.utils import edge_index_to_adjs
from moge.model.metrics import Metrics
from moge.model.utils import tensor_sizes, preprocess_input
from moge.visualization.attention import plot_sankey_flow


class ClusteringEvaluator(LightningModule):
    def register_hooks(self):
        # Register hooks for embedding layer and classifier layer
        for name, layer in self.named_children():
            layer.__name__ = name
            print(name)
            layer.register_forward_hook(self.save_embedding)
            layer.register_forward_hook(self.save_pred)

        # def save_node_ids(module, inputs):
        #     # if module.training: return
        #     logging.info(f"save_node_ids @ {module.__name__} {tensor_sizes(inputs)}")
        #
        # # Register a hook to get node_ids input
        # for layer in itertools.islice(self.modules(), 1):
        #     print(layer.name())
        #     layer.register_forward_pre_hook(save_node_ids)

    def save_embedding(self, module, _, outputs):
        if self.training:
            return

        if module.__name__ == "embedder":
            logging.info(f"save_embedding @ {module.__name__}")

            if isinstance(outputs, (list, tuple)):
                self._embeddings = outputs[0]
            else:
                self._embeddings = outputs

    def save_pred(self, module, _, outputs):
        if self.training:
            return

        if module.__name__ in ["classifier"]:
            logging.info(
                f"save_pred @ {module.__name__}, output {tensor_sizes(outputs)}")

            if isinstance(outputs, (list, tuple)):
                self._y_pred = outputs[0]
            else:
                self._y_pred = outputs

    def trainvalidtest_dataloader(self):
        return self.dataset.trainvalidtest_dataloader(collate_fn=self.collate_fn, )

    def clustering_metrics(self, n_runs=10, compare_node_types=True):
        loader = self.trainvalidtest_dataloader()
        X_all, y_all, _ = next(iter(loader))
        self.cpu().forward(preprocess_input(X_all, device="cpu"))

        if not isinstance(self._embeddings, dict):
            self._embeddings = {list(self._node_ids.keys())[0]: self._embeddings}

        embeddings_all, types_all, y_true = self.get_embeddings_labels(self._embeddings, self._node_ids)

        # Record metrics for each run in a list of dict's
        res = [{}, ] * n_runs
        for i in range(n_runs):
            y_pred = self.predict_cluster(n_clusters=len(y_true.unique()), seed=i)

            if compare_node_types and len(self.dataset.node_types) > 1:
                res[i].update(clustering_metrics(y_true=types_all,
                                                 # Match y_pred to type_all's index
                                                 y_pred=types_all.index.map(lambda idx: y_pred.get(idx, "")),
                                                 metrics=["homogeneity_ntype", "completeness_ntype", "nmi_ntype"]))

            if y_pred.shape[0] != y_true.shape[0]:
                y_pred = y_pred.loc[y_true.index]
            res[i].update(clustering_metrics(y_true,
                                             y_pred,
                                             metrics=["homogeneity", "completeness", "nmi"]))

        res_df = pd.DataFrame(res)
        metrics = res_df.mean(0).to_dict()
        return metrics

    def get_embeddings_labels(self, embeddings: Dict[str, Tensor], global_node_index: Dict[str, Tensor],
                              cache: bool = True) \
            -> Tuple[DataFrame, Series, Series]:
        if cache and hasattr(self, "_embeddings") and hasattr(self, "_node_types") and hasattr(self, "_labels"):
            return self._embeddings, self._node_types, self._labels

        # Building a dataframe of embeddings, indexed by "{node_type}{node_id}"
        emb_df_list = []
        for node_type in self.dataset.node_types:
            nid = global_node_index[node_type].cpu().numpy().astype(str)
            n_type_id = np.core.defchararray.add(node_type[0], nid)

            if isinstance(embeddings[node_type], Tensor):
                df = pd.DataFrame(embeddings[node_type].detach().cpu().numpy(), index=n_type_id)
            else:
                df = pd.DataFrame(embeddings[node_type], index=n_type_id)
            emb_df_list.append(df)

        embeddings = pd.concat(emb_df_list, axis=0)
        node_types = embeddings.index.to_series().str.slice(0, 1)
        target_ntype = self.dataset.head_node_type

        # Build vector of labels for all node types
        if hasattr(self.dataset, "y_dict") and len(self.dataset.y_dict) > 0:
            labels = pd.Series(
                self.dataset.y_dict[target_ntype][global_node_index[target_ntype]].squeeze(-1).numpy(),
                index=emb_df_list[0].index,
                dtype=str)
        else:
            labels = None

        # Save results
        self._embeddings, self._node_types, self._labels = embeddings, node_types, labels

        return embeddings, node_types, labels

    def predict_cluster(self, n_clusters=8, n_jobs=-2, save_kmeans=False, seed=None):
        kmeans = KMeans(n_clusters, n_jobs=n_jobs, random_state=seed)
        logging.info(f"Kmeans with k={n_clusters}")
        y_pred = kmeans.fit_predict(self.embeddings)
        if save_kmeans:
            self.kmeans = kmeans

        y_pred = pd.Series(y_pred, index=self.embeddings.index, dtype=str)
        return y_pred


class NodeEmbeddingEvaluator(LightningModule):
    dataset: HeteroGraphDataset

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.attn_plot_name = "sankey_flow"
        self.embedding_plot_name = "node_emb_umap_plot"
        self.score_avg_table_name = "score_avgs"
        self.beta_degree_corr_table_name = "beta_degree_correlation"

    def plot_embeddings_tsne(self, global_node_index: Dict[str, Tensor], embeddings: Dict[str, Tensor],
                             targets: Any = None, y_pred: Any = None, weights: Dict[str, Tensor] = None,
                             columns=["node", "ntype", "pos1", "pos2", "loss"], n_samples: int = 1000) -> DataFrame:
        node_losses = self.get_node_loss(targets, y_pred, global_node_index=global_node_index)
        df = self.dataset.get_node_metadata(global_node_index, embeddings, weights=weights, losses=node_losses)

        # Log_table
        df_filter = df.reset_index().filter(columns, axis="columns")
        if n_samples:
            df_filter = df_filter.sample(n_samples)

        table = wandb.Table(data=df_filter)
        wandb.log({self.embedding_plot_name: table})
        logging.info("Logging node_emb_umap_plot")

        return df

    def get_node_loss(self, targets: Union[Tensor, Dict[Any, Any]], y_pred: Union[Tensor, Dict[Any, Any]],
                      global_node_index: Dict[str, Tensor], ) -> DataFrame:
        """
        Compute the loss for each nodes given targets and predicted values for either node_clf or link_pred tasks.
        Args:
            targets ():
            y_pred ():
            global_node_index ():
        """
        raise NotImplementedError

    def plot_sankey_flow(self, layer: int = -1, width=500, height=300):
        if self.wandb_experiment is None or not hasattr(self.embedder, "layers"):
            return

        run_id = self.wandb_experiment.id

        node_types = list(self.embedder.layers[layer]._betas.keys())
        table = wandb.Table(columns=[f"Layer{layer + 1 if layer >= 0 else len(self.embedder.layers)}_{ntype}" \
                                     for ntype in node_types])

        # Log plotly HTMLs as a wandb.Table
        plotly_htmls = []
        for ntype in node_types:
            nodes, links = self.embedder.get_sankey_flow(layer=layer, node_type=ntype, self_loop=True)
            fig = plot_sankey_flow(nodes, links, width=width, height=height)

            path_to_plotly_html = f"./wandb_fig_run_{run_id}_{ntype}.html"
            fig.write_html(path_to_plotly_html, auto_play=False, include_plotlyjs=True, full_html=True,
                           config=dict(displayModeBar=False))
            plotly_htmls.append(wandb.Html(path_to_plotly_html))

        # Add Plotly figure as HTML file into Table
        table.add_data(*plotly_htmls)

        # Log Table
        wandb.log({self.attn_plot_name: table})
        logging.info("Logging sankey_flow")
        os.system(f"rm -f ./wandb_fig_run_{run_id}*.html")

    def log_score_averages(self, edge_scores_dict: Dict[str, Dict[Tuple[str, str, str], Tensor]]) -> DataFrame:
        if isinstance(edge_scores_dict, dict) and isinstance(list(edge_scores_dict.keys())[0], str):
            score_avgs = pd.DataFrame({key: {metapath: f"{values.mean().item():.4f} ± {values.std().item():.2f}" \
                                             for metapath, values in edge_index_dict.items()} \
                                       for key, edge_index_dict in edge_scores_dict.items()})

            score_avgs.index.names = ("head", "relation", "tail")

        elif isinstance(edge_scores_dict, dict) and isinstance(list(edge_scores_dict.keys())[0], tuple):
            score_avgs = pd.DataFrame.from_dict({m: f"{values.mean().item():.4f} ± {values.std().item():.2f}"
                                                 for m, values in edge_scores_dict.items()}, orient="index")
            score_avgs.columns = ["score"]
            score_avgs.index = pd.MultiIndex.from_tuples(score_avgs.index, names=("head", "relation", "tail"))

        try:
            table = wandb.Table(dataframe=score_avgs.reset_index())

            wandb.log({self.score_avg_table_name: table})
            logging.info("Logging log_score_averages")
        except:
            pass
        finally:
            return score_avgs

    def log_beta_degree_correlation(self, global_node_index: Dict[str, Tensor],
                                    batch_size: Dict[str, int] = None) -> DataFrame:
        nodes_index = {ntype: nids.numpy() \
                       for ntype, nids in global_node_index.items()}
        if batch_size:
            nodes_index = {ntype: nids[: batch_size[ntype]] \
                           for ntype, nids in nodes_index.items() if ntype in batch_size}

        nodes_index = pd.MultiIndex.from_tuples(
            ((ntype, nid) for ntype, nids in nodes_index.items() for nid in nids),
            names=["ntype", "nid"])

        nodes_degree = self.dataset.get_node_degrees().loc[nodes_index].fillna(0)

        layers_corr = {}
        for layer, betas in enumerate(self.betas):
            nodes_betas = pd.concat(betas, names=["ntype", "nid"]).loc[nodes_index].fillna(0)
            nodes_corr = nodes_betas.corrwith(nodes_degree, axis=1, drop=True) \
                .groupby("ntype") \
                .mean()

            layers_corr[layer + 1] = nodes_corr

        layers_corr = pd.concat(layers_corr, names=["Layer", "ntype"]).unstack().fillna(0.0)
        layers_corr.columns.name = None

        try:
            wandb.log({self.beta_degree_corr_table_name: layers_corr.reset_index()})
            logging.info("Logging beta_degree_correlation")
        except:
            pass

        return layers_corr

    @property
    def wandb_experiment(self):
        if isinstance(self.logger, WandbLogger):
            return self.logger.experiment
        else:
            return None

    def cleanup_artifacts(self):
        experiment = self.wandb_experiment
        if experiment is None:
            return

        api = wandb.Api(overrides={"project": experiment.project, "entity": experiment.entity})

        artifact_type, artifact_name = "run_table", f"run-{experiment.id}-{self.attn_plot_name}"
        for version in api.artifact_versions(artifact_type, artifact_name):
            # Clean up all versions that don't have an alias such as 'latest'.
            # NOTE: You can put whatever deletion logic you want here.
            if len(version.aliases) == 0:
                version.delete()

        # artifact_type, artifact_name = "run_table", f"run-{experiment.id}-{self.embedding_plot_name}"
        # for version in api.artifact_versions(artifact_type, artifact_name):
        #     # Clean up all versions that don't have an alias such as 'latest'.
        #     # NOTE: You can put whatever deletion logic you want here.
        #     if len(version.aliases) == 0:
        #         version.delete()


class NodeClfTrainer(ClusteringEvaluator, NodeEmbeddingEvaluator):
    def __init__(self, hparams, dataset, metrics: Union[List[str], Dict[str, List[str]]], *args, **kwargs):
        super().__init__(*args, **kwargs)

        if isinstance(metrics, dict):
            self.train_metrics = {
                subtype: Metrics(prefix="" + subtype + "_", loss_type=hparams.loss_type, n_classes=dataset.n_classes,
                                 multilabel=dataset.multilabel, metrics=keywords) \
                for subtype, keywords in metrics.items()}
            self.valid_metrics = {
                subtype: Metrics(prefix="val_" + subtype + "_", loss_type=hparams.loss_type,
                                 n_classes=dataset.n_classes,
                                 multilabel=dataset.multilabel, metrics=keywords) \
                for subtype, keywords in metrics.items()}
            self.test_metrics = {
                subtype: Metrics(prefix="test_" + subtype + "_", loss_type=hparams.loss_type,
                                 n_classes=dataset.n_classes,
                                 multilabel=dataset.multilabel, metrics=keywords) \
                for subtype, keywords in metrics.items()}
        else:
            self.train_metrics = Metrics(prefix="", loss_type=hparams.loss_type, n_classes=dataset.n_classes,
                                         multilabel=dataset.multilabel, metrics=metrics)
            self.valid_metrics = Metrics(prefix="val_", loss_type=hparams.loss_type, n_classes=dataset.n_classes,
                                         multilabel=dataset.multilabel, metrics=metrics)
            self.test_metrics = Metrics(prefix="test_", loss_type=hparams.loss_type, n_classes=dataset.n_classes,
                                        multilabel=dataset.multilabel, metrics=metrics)
        hparams.name = self.name()
        hparams.inductive = dataset.inductive
        self._set_hparams(hparams)

    def name(self):
        if hasattr(self, "_name"):
            return self._name
        else:
            return self.__class__.__name__.replace("_", "-")

    def training_epoch_end(self, outputs):
        metrics_dict = {}
        if isinstance(self.train_metrics, Metrics):
            metrics_dict = self.train_metrics.compute_metrics()
            self.train_metrics.reset_metrics()
        elif isinstance(self.train_metrics, dict):
            metrics_dict = {k: v for subtype, metrics in self.train_metrics.items() \
                            for k, v in metrics.compute_metrics().items()}

            for subtype, metrics in self.train_metrics.items():
                metrics.reset_metrics()

        self.log_dict(metrics_dict, prog_bar=True)

        return None

    def validation_epoch_end(self, outputs):
        metrics_dict = {}
        if isinstance(self.valid_metrics, Metrics):
            metrics_dict = self.valid_metrics.compute_metrics()

            self.valid_metrics.reset_metrics()

        elif isinstance(self.valid_metrics, dict):
            metrics_dict = {k: v for subtype, metrics in self.valid_metrics.items() \
                            for k, v in metrics.compute_metrics().items()}

            for subtype, metrics in self.valid_metrics.items():
                metrics.reset_metrics()

        self.log_dict(metrics_dict, prog_bar=True)

        return None

    def test_epoch_end(self, outputs):
        metrics_dict = {}
        if isinstance(self.test_metrics, Metrics):
            metrics_dict = self.test_metrics.compute_metrics()
            self.test_metrics.reset_metrics()

        elif isinstance(self.test_metrics, dict):
            metrics_dict = {k: v for subtype, metrics in self.test_metrics.items() \
                            for k, v in metrics.compute_metrics().items()}

            for subtype, metrics in self.test_metrics.items():
                metrics.reset_metrics()

        self.log_dict(metrics_dict, prog_bar=True)
        return None

    def get_node_loss(self, targets: Tensor, y_pred: Tensor, global_node_index: Dict[str, Tensor] = None):
        losses = F.binary_cross_entropy(y_pred.detach(),
                                        target=targets.float(),
                                        reduce=False).mean(dim=1).numpy()

        losses = {self.head_node_type: losses}

        return losses

    def train_dataloader(self, **kwargs):
        if hasattr(self.hparams, "num_gpus") and self.hparams.num_gpus > 1:
            train_sampler = DistributedSampler(self.dataset.training_idx, num_replicas=self.hparams.num_gpus,
                                               rank=self.local_rank)
        else:
            train_sampler = None

        dataset = self.dataset.train_dataloader(collate_fn=self.collate_fn,
                                                batch_size=self.hparams.batch_size, batch_sampler=train_sampler,
                                                **kwargs)
        return dataset

    def val_dataloader(self, **kwargs):
        if hasattr(self.hparams, "num_gpus") and self.hparams.num_gpus > 1:
            train_sampler = DistributedSampler(self.dataset.validation_idx, num_replicas=self.hparams.num_gpus,
                                               rank=self.local_rank)
        else:
            train_sampler = None

        dataset = self.dataset.valid_dataloader(collate_fn=self.collate_fn,
                                                batch_size=self.hparams.batch_size, batch_sampler=train_sampler,
                                                **kwargs)

        return dataset


    def test_dataloader(self, **kwargs):
        if hasattr(self.hparams, "num_gpus") and self.hparams.num_gpus > 1:
            train_sampler = DistributedSampler(self.dataset.testing_idx, num_replicas=self.hparams.num_gpus,
                                               rank=self.local_rank)
        else:
            train_sampler = None

        dataset = self.dataset.test_dataloader(collate_fn=self.collate_fn,
                                               batch_size=self.hparams.batch_size, batch_sampler=train_sampler,
                                               **kwargs)
        return dataset

    def get_n_params(self):
        # size = 0
        # for name, param in dict(self.named_parameters()).items():
        #     nn = 1
        #     for s in list(param.size()):
        #         nn = nn * s
        #     size += nn
        # return size
        model_parameters = filter(lambda tup: tup[1].requires_grad and "embedding" not in tup[0],
                                  self.named_parameters())

        params = sum([np.prod(p.size()) for name, p in model_parameters])
        return params

    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if self.trainer.max_steps:
            return self.trainer.max_steps

        limit_batches = self.trainer.limit_train_batches
        batches = len(self.train_dataloader())
        batches = min(batches, limit_batches) if isinstance(limit_batches, int) else int(limit_batches * batches)

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_accum = self.trainer.accumulate_grad_batches * num_devices
        return (batches // effective_accum) * self.trainer.max_epochs

    def set_fanouts(self, dataset: Union[DGLNodeSampler, HeteroNeighborGenerator], fanouts: Iterable):
        dataset.neighbor_sizes = fanouts

        if isinstance(dataset, DGLNodeSampler):
            dataset.neighbor_sampler.fanouts = fanouts
            dataset.neighbor_sampler.num_layers = len(fanouts)
        elif isinstance(dataset, HeteroNeighborGenerator):
            dataset.graph_sampler.graph_sampler.sizes = fanouts

        print(f"Changed graph neighbor sampling sizes to {fanouts}, because method have {len(fanouts)} layers.")


class LinkPredTrainer(NodeClfTrainer):
    def __init__(self, hparams, dataset, metrics: Union[List[str], Dict[str, List[str]]], *args, **kwargs):
        super().__init__(hparams, dataset, metrics, *args, **kwargs)

    def get_node_loss(self,
                      edge_pred: Union[
                          Dict[Tuple[str, str, str], Tensor], Dict[str, Dict[Tuple[str, str, str], Tensor]]],
                      edge_true: Union[
                          Dict[Tuple[str, str, str], Tensor], Dict[str, Dict[Tuple[str, str, str], Tensor]]],
                      global_node_index: Optional[Dict[str, Tensor]] = None) -> Dict[str, Tensor]:
        edge_index_loss = self.get_edge_index_loss(edge_pred, edge_true, global_node_index)

        adj_losses = edge_index_to_adjs(edge_index_loss, nodes=self.dataset.nodes)

        # Aggregate loss from adj_losses to all nodes of each ntype
        ntypes = {ntype for metapath in adj_losses.keys() for ntype in metapath[0::2] if ntype in global_node_index}

        ntype_losses = {ntype: torch.zeros(len(self.dataset.nodes[ntype])) for ntype in ntypes}
        ntype_counts = defaultdict(lambda: 0)
        for metapath, adj in adj_losses.items():
            head_type, tail_type = metapath[0], metapath[-1]
            ntype_losses[head_type] += adj.sum(1)
            ntype_losses[tail_type] += adj.sum(0)
            ntype_counts[head_type] += 1
            ntype_counts[tail_type] += 1

        ntype_losses = {ntype: loss[global_node_index[ntype]] / ntype_counts[ntype] \
                        for ntype, loss in ntype_losses.items()}

        return ntype_losses

    def train_dataloader(self, **kwargs):
        return self.dataset.train_dataloader(collate_fn=self.collate_fn,
                                             batch_size=self.hparams.batch_size, **kwargs)

    def val_dataloader(self, **kwargs):
        if self.dataset.name() in ["ogbl-biokg", "ogbl-wikikg"]:
            batch_size = self.test_batch_size
        else:
            batch_size = self.hparams.batch_size
        return self.dataset.valid_dataloader(collate_fn=self.collate_fn,
                                             batch_size=batch_size, **kwargs)

    def test_dataloader(self, **kwargs):
        if self.dataset.name() in ["ogbl-biokg", "ogbl-wikikg"]:
            batch_size = self.test_batch_size
        else:
            batch_size = self.hparams.batch_size
        return self.dataset.test_dataloader(collate_fn=self.collate_fn,
                                            batch_size=batch_size, **kwargs)


class GraphClfTrainer(LightningModule):
    def __init__(self, hparams, dataset, metrics, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.train_metrics = Metrics(prefix="", loss_type=hparams.loss_type, n_classes=dataset.n_classes,
                                     multilabel=dataset.multilabel, metrics=metrics)
        self.valid_metrics = Metrics(prefix="val_", loss_type=hparams.loss_type, n_classes=dataset.n_classes,
                                     multilabel=dataset.multilabel, metrics=metrics)
        self.test_metrics = Metrics(prefix="test_", loss_type=hparams.loss_type, n_classes=dataset.n_classes,
                                    multilabel=dataset.multilabel, metrics=metrics)
        hparams.name = self.name()
        hparams.inductive = dataset.inductive
        self.hparams = hparams

    def name(self):
        if hasattr(self, "_name"):
            return self._name
        else:
            return self.__class__.__name__

    def get_n_params(self):
        size = 0
        for name, param in dict(self.named_parameters()).items():
            nn = 1
            for s in list(param.size()):
                nn = nn * s
            size += nn
        return size

    def training_epoch_end(self, outputs):
        logs = self.train_metrics.compute_metrics()
        self.train_metrics.reset_metrics()
        self.log_dict(logs, logger=True, prog_bar=True)
        return None

    def validation_epoch_end(self, outputs):
        logs = self.valid_metrics.compute_metrics()
        # print({k: np.around(v.item(), decimals=3) for k, v in logs.items()})

        self.valid_metrics.reset_metrics()
        self.log_dict(logs, logger=True, prog_bar=True)
        return None

    def test_epoch_end(self, outputs):
        if hasattr(self, "test_metrics"):
            logs = self.test_metrics.compute_metrics()
            self.test_metrics.reset_metrics()
        else:
            logs = {}

        self.log_dict(logs, logger=True, prog_bar=True)
        return None

    def train_dataloader(self):
        return self.dataset.train_dataloader(collate_fn=self.collate_fn,
                                             batch_size=self.hparams.batch_size)

    def val_dataloader(self):
        return self.dataset.valid_dataloader(collate_fn=self.collate_fn,
                                             batch_size=self.hparams.batch_size)

    def valtrain_dataloader(self):
        return self.dataset.valtrain_dataloader(collate_fn=self.collate_fn,
                                                batch_size=self.hparams.batch_size)

    def test_dataloader(self):
        return self.dataset.test_dataloader(collate_fn=self.collate_fn,
                                            batch_size=self.hparams.batch_size)


def print_pred_class_counts(y_pred, y_true, multilabel, n_top_class=8):
    if (y_pred < 0.0).any():
        y_pred = torch.sigmoid(y_pred.clone()) if multilabel else torch.softmax(y_pred.clone(), dim=-1)

    if multilabel:
        y_pred_dict = pd.Series(y_pred.sum(1).detach().cpu().type(torch.int).numpy()).value_counts().to_dict()
        y_true_dict = pd.Series(y_true.sum(1).detach().cpu().type(torch.int).numpy()).value_counts().to_dict()
        print(f"y_pred {len(y_pred_dict)} classes",
              {str(k): v for k, v in itertools.islice(y_pred_dict.items(), n_top_class)})
        print(f"y_true {len(y_true_dict)} classes",
              {str(k): v for k, v in itertools.islice(y_true_dict.items(), n_top_class)})
    else:
        y_pred_dict = pd.Series(y_pred.argmax(1).detach().cpu().type(torch.int).numpy()).value_counts().to_dict()
        y_true_dict = pd.Series(y_true.detach().cpu().type(torch.int).numpy()).value_counts().to_dict()
        print(f"y_pred {len(y_pred_dict)} classes",
              {str(k): v for k, v in itertools.islice(y_pred_dict.items(), n_top_class)})
        print(f"y_true {len(y_true_dict)} classes",
              {str(k): v for k, v in itertools.islice(y_true_dict.items(), n_top_class)})
