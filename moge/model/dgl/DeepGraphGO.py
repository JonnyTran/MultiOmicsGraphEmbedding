import os.path
import warnings
from argparse import Namespace
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import List, Dict, Tuple

import dgl
import dgl.function as fn
import joblib
import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from Bio.Blast import NCBIXML
from Bio.Blast.Applications import NcbipsiblastCommandline
from dask import dataframe as dd
from dgl.heterograph import DGLBlock
from dgl.udf import NodeBatch
from logzero import logger
from ruamel.yaml import YAML
from scipy import sparse as ssp
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import MultiLabelBinarizer
from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from moge.model.metrics import Metrics, add_common_metrics
from moge.model.trainer import NodeEmbeddingEvaluator
from moge.model.utils import to_device


def get_pid_list(pid_list_file):
    try:
        with open(pid_list_file) as fp:
            return [line.split()[0] for line in fp]
    except TypeError:
        return pid_list_file


def get_go_list(pid_go_file, pid_list):
    if pid_go_file is not None:
        pid_go = defaultdict(list)
        with open(pid_go_file) as fp:
            for line in fp:
                line_list = line.split()
                pid_go[(line_list)[0]].append(line_list[1])
        return [pid_go[pid_] for pid_ in pid_list]
    else:
        return None


def get_pid_go(pid_go_file):
    if pid_go_file is not None:
        pid_go = defaultdict(list)
        with open(pid_go_file) as fp:
            for line in fp:
                line_list = line.split('\t')
                pid_go[(line_list)[0]].append(line_list[1])
        return dict(pid_go)
    else:
        return None


def get_pid_go_sc(pid_go_sc_file):
    pid_go_sc = defaultdict(dict)
    with open(pid_go_sc_file) as fp:
        for line in fp:
            line_list = line.split('\t')
            pid_go_sc[line_list[0]][line_list[1]] = float((line_list)[2])
    return dict(pid_go_sc)


def get_data(fasta_file, pid_go_file=None, feature_type=None, subset_pid=None, **kwargs):
    pid_list, seqs = [], []
    for seq in SeqIO.parse(fasta_file, 'fasta'):
        if subset_pid and seq.id not in subset_pid:
            continue

        pid_list.append(seq.id)
        seqs.append(str(seq.seq))

    if feature_type is not None:
        feature_path = Path(kwargs[feature_type])
        if feature_path.suffix == '.npy':
            seqs = np.load(feature_path)
        elif feature_path.suffix == '.npz':
            seqs = ssp.load_npz(feature_path)
        else:
            raise ValueError(F'Only support suffix of .npy for np.ndarray or .npz for scipy.csr_matrix as feature.')

    go_labels = get_go_list(pid_go_file, pid_list)

    return pid_list, seqs, go_labels


def get_ppi_idx(pid_list: List[str], data_y: ssp.csr_matrix, net_pid_map: Dict[str, int]):
    pid_list_ = tuple(zip(*[(i, pid, net_pid_map[pid])
                            for i, pid in enumerate(pid_list) if pid in net_pid_map]))
    assert pid_list_
    pid_list_ = (np.asarray(pid_list_[0]), pid_list_[1], np.asarray(pid_list_[2]))
    return pid_list_[0], pid_list_[1], pid_list_[2], data_y[pid_list_[0]] if data_y is not None else data_y


def psiblast(blastdb, pid_list, fasta_path, output_path: Path, evalue=1e-3, num_iterations=3,
             num_threads=40, bits=True, query_self=False, **kwargs):
    output_path = output_path.with_suffix('.xml')
    if not output_path.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cline = NcbipsiblastCommandline(query=fasta_path, db=blastdb, evalue=evalue, outfmt=5, out=output_path,
                                        num_iterations=num_iterations, num_threads=num_threads, **kwargs)
        logger.info(cline)
        cline()
    else:
        logger.info(F'Using exists blast output file {output_path}')
    with open(output_path) as fp:
        psiblast_sim = defaultdict(dict)
        for pid, rec in zip(tqdm(pid_list, desc='Parsing PsiBlast results'), NCBIXML.parse(fp)):
            query_pid, sim = rec.query, []
            assert pid == query_pid
            for alignment in rec.alignments:
                alignment_pid = alignment.hit_def.split()[0]
                if alignment_pid != query_pid or query_self:
                    psiblast_sim[query_pid][alignment_pid] = max(hsp.bits if bits else hsp.identities / rec.query_length
                                                                 for hsp in alignment.hsps)
    return psiblast_sim


def get_homo_ppi_idx(pid_list, fasta_file, data_y, net_pid_map: Dict[str, int], net_blastdb: Path,
                     blast_output_path: Path):
    blast_sim = psiblast(net_blastdb, pid_list=pid_list, fasta_path=fasta_file, output_path=blast_output_path,
                         num_iterations=1)
    pid_list_ = []
    for i, pid in enumerate(pid_list):
        blast_sim[pid][None] = float('-inf')
        pid_ = pid if pid in net_pid_map else max(blast_sim[pid].items(), key=lambda x: x[1])[0]
        if pid_ is not None:
            pid_list_.append((i, pid, net_pid_map[pid_]))
    pid_list_ = tuple(zip(*pid_list_))
    pid_list_ = (np.asarray(pid_list_[0]), pid_list_[1], np.asarray(pid_list_[2]))
    return pid_list_[0], pid_list_[1], pid_list_[2], data_y[pid_list_[0]] if data_y is not None else data_y


def get_mlb(mlb_path: Path, labels=None, **kwargs) -> MultiLabelBinarizer:
    if isinstance(mlb_path, Path) and mlb_path.exists() or os.path.exists(mlb_path):
        return joblib.load(mlb_path)
    mlb = MultiLabelBinarizer(sparse_output=True, **kwargs)
    mlb.fit(labels)
    joblib.dump(mlb, mlb_path)
    return mlb


def fmax(targets: ssp.csr_matrix, scores: np.ndarray) -> Tuple[float, float]:
    if isinstance(targets, pd.DataFrame):
        targets = targets.values
    if isinstance(scores, pd.DataFrame):
        scores = scores.values
    if not isinstance(targets, ssp.csr_matrix):
        targets = ssp.csr_matrix(targets)

    fmax_ = 0.0, 0.0
    for thresh in (c / 100 for c in range(101)):
        cut_sc = ssp.csr_matrix((scores >= thresh).astype(np.int32))
        correct = cut_sc.multiply(targets).sum(axis=1)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            p, r = correct / cut_sc.sum(axis=1), correct / targets.sum(axis=1)
            p, r = np.average(p[np.invert(np.isnan(p))]), np.average(r)
        if np.isnan(p):
            continue

        try:
            fmax_ = max(fmax_, (2 * p * r / (p + r) if p + r > 0.0 else 0.0, thresh))
        except ZeroDivisionError:
            pass
    return fmax_


def pair_aupr(targets: np.ndarray, scores: np.ndarray, top=200):
    if isinstance(targets, pd.DataFrame):
        targets = targets.values
    if isinstance(scores, pd.DataFrame):
        scores = scores.values
    if isinstance(targets, ssp.csr_matrix):
        targets = targets.toarray()

    rows = np.arange(scores.shape[0])[:, None]
    top_k_cols = scores.argpartition(scores.shape[1] - top)[:, :-top]
    scores[rows, top_k_cols] = 0  # -1e100

    # scores = np.nan_to_num(scores, neginf=0)
    return average_precision_score(targets.flatten(), scores.flatten())


def output_res(res_path: Path, pid_list, go_list, sc_mat):
    res_path.parent.mkdir(parents=True, exist_ok=True)
    with open(res_path, 'w') as fp:
        for pid_, sc_ in zip(pid_list, sc_mat):
            for go_, s_ in zip(go_list, sc_):
                if s_ > 0.0:
                    print(pid_, go_, s_, sep='\t', file=fp)


class NodeUpdate(nn.Module):
    def __init__(self, in_dim, out_dim, dropout: float):
        super(NodeUpdate, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, node: NodeBatch, ):
        outputs = self.dropout(F.relu(self.linear(node.data['ppi_out'])))

        if 'res' in node.data:
            outputs = outputs + node.data['res']
        return {'h': outputs}

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)


class GcnNet(nn.Module):
    """
    """

    def __init__(self, hidden_size, num_gcn=0, dropout=0.5, residual=True, **kwargs):
        super().__init__()

        self.layers: List[NodeUpdate] = nn.ModuleList([
            NodeUpdate(hidden_size, hidden_size, dropout) for _ in range(num_gcn)])

        self.residual = residual
        self.num_gcn = num_gcn
        self.reset_parameters()

    def reset_parameters(self):
        for update in self.layers:
            update.reset_parameters()

    def forward(self, blocks: List[DGLBlock], h: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            blocks[i].srcdata['h'] = h

            if self.residual:
                blocks[i].update_all(fn.u_mul_e('h', 'self', out='m_res'),
                                     fn.sum(msg='m_res', out='res'))
            blocks[i].update_all(fn.u_mul_e('h', 'ppi', out='ppi_m_out'),
                                 fn.sum(msg='ppi_m_out', out='ppi_out'), layer)
            h = blocks[i].dstdata['h']

        return h


class DeepGraphGO(NodeEmbeddingEvaluator):
    def __init__(self, hparams: Namespace, model_path: Path, dgl_graph: dgl.DGLGraph, node_feats: ssp.csr_matrix,
                 metrics: List[str] = ["aupr", "fmax"]):
        if not isinstance(hparams, Namespace) and isinstance(hparams, dict):
            hparams = Namespace(**hparams)
        super().__init__()

        # Protein - species mapping
        if hasattr(hparams, 'protein_data') and isinstance(hparams.protein_data, str):
            df = dd.read_table(hparams.protein_data,
                               names=['pid', "go_id", 'namespace', 'species_id'],
                               usecols=['pid', 'species_id'])
            groupby = df.groupby('pid')
            proteins = groupby['species_id'].first().to_frame()
            self.node_namespace = proteins.compute()
        elif hasattr(hparams, 'protein_data') and isinstance(hparams.protein_data, pd.DataFrame):
            self.node_namespace = hparams.protein_data

        if hasattr(hparams, 'nodes'):
            self.nodes = hparams.nodes

        if metrics:
            self.metric_prefix = {'bp': "BPO", 'cc': 'CCO', 'mf': 'MFO'}[hparams.namespace]
            self.train_metrics = Metrics(prefix=f"{self.metric_prefix}_", metrics=metrics, loss_type='BCE_WITH_LOGITS',
                                         n_classes=hparams.n_classes, multilabel=True)
            self.valid_metrics = Metrics(prefix=f"val_{self.metric_prefix}_", metrics=metrics,
                                         loss_type='BCE_WITH_LOGITS', n_classes=hparams.n_classes, multilabel=True)
            self.test_metrics = Metrics(prefix=f"test_{self.metric_prefix}_", metrics=metrics,
                                        loss_type='BCE_WITH_LOGITS', n_classes=hparams.n_classes, multilabel=True)

        self.input = nn.EmbeddingBag(hparams.input_size, hparams.hidden_size, mode='sum', include_last_offset=True)
        self.input_bias = nn.Parameter(torch.zeros(hparams.hidden_size))
        self.dropout = nn.Dropout(hparams.dropout)

        self.model = GcnNet(**hparams.__dict__)

        self.n_classes = hparams.n_classes
        self.classifier = nn.Linear(hparams.hidden_size, self.n_classes)
        logger.info(
            F'GCN: labels_num={hparams.n_classes}, input size={hparams.input_size}, hidden_size={hparams.hidden_size}, '
            F'num_gcn={hparams.num_gcn}, dropout={hparams.dropout}, residual={hparams.residual}')

        model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model_path = model_path
        self.criterion = nn.BCEWithLogitsLoss()
        self.dgl_graph, self.node_feats, self.batch_size = dgl_graph, node_feats, hparams.batch_size

        self.training_idx, self.validation_idx, self.testing_idx = hparams.training_idx, hparams.validation_idx, hparams.testing_idx

        self._set_hparams(hparams)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.input.weight)
        nn.init.xavier_uniform_(self.classifier.weight)

    def forward(self, blocks: List[DGLBlock], return_embeddings=False):
        node_ids = blocks[0].srcdata["_ID"]
        batch_x = self.node_feats[node_ids.cpu().numpy()]

        input = torch.from_numpy(batch_x.indices).to(self.device).long()
        offsets = torch.from_numpy(batch_x.indptr).to(self.device).long()
        per_sample_weights = torch.from_numpy(batch_x.data).to(self.device).float()

        h = self.input(input, offsets, per_sample_weights) + self.input_bias
        h = self.dropout(F.relu(h))

        h = self.model.forward(blocks, h)
        logits = self.classifier(h)

        if return_embeddings:
            return h, logits

        return logits

    def training_step(self, batch: Tuple[Tensor, Tensor, List[DGLBlock]], batch_nb):
        input_nodes, seeds, blocks = batch
        targets = blocks[-1].dstdata["label"]

        logits = self.forward(blocks)
        loss = self.criterion.forward(logits, targets)
        self.log("loss", loss, logger=True, prog_bar=True, on_step=False, on_epoch=True)

        # self.train_metrics.update_metrics(torch.sigmoid(logits), targets)
        # scores = torch.sigmoid(logits).detach().cpu().numpy()
        # y_true = y_true.detach().cpu().numpy()
        # (fmax_, t_), aupr_ = fmax(y_true, scores), pair_aupr(y_true, scores)
        # self.log_dict({"fmax": fmax_, "aupr": aupr_}, logger=True, prog_bar=True,
        #               on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: Tuple[Tensor, Tensor, List[DGLBlock]], batch_nb):
        input_nodes, seeds, blocks = batch
        targets = blocks[-1].dstdata["label"]

        logits = self.forward(blocks)
        loss = self.criterion(logits, targets)
        self.log("val_loss", loss, logger=True, on_step=False, on_epoch=True)

        self.valid_metrics.update_metrics(torch.sigmoid(logits), targets)
        return loss

    def test_step(self, batch: Tuple[Tensor, Tensor, List[DGLBlock]], batch_nb):
        input_nodes, seeds, blocks = batch
        targets = blocks[-1].dstdata["label"]

        logits = self.forward(blocks)
        loss = self.criterion(logits, targets)
        self.log("test_loss", loss, logger=True, on_step=False, on_epoch=True)

        self.test_metrics.update_metrics(torch.sigmoid(logits), targets)
        return loss

    @torch.no_grad()
    def predict_step(self, data_x):
        self.model.eval()
        input_nodes, seeds, blocks = data_x

        return torch.sigmoid(self.forward(blocks)).cpu().numpy()

    def training_epoch_end(self, outputs):
        if hasattr(self, 'train_metrics'):
            metrics_dict = self.train_metrics.compute_metrics()
            self.train_metrics.reset_metrics()
            metrics_dict = add_common_metrics(metrics_dict, prefix='', metrics_suffixes=['aupr', 'fmax'])
            self.log_dict(metrics_dict, prog_bar=True, on_step=False, on_epoch=True)
        super().training_epoch_end(outputs)

    def validation_epoch_end(self, outputs):
        if hasattr(self, 'valid_metrics'):
            metrics_dict = self.valid_metrics.compute_metrics()
            self.valid_metrics.reset_metrics()
            metrics_dict = add_common_metrics(metrics_dict, prefix='val_', metrics_suffixes=['aupr', 'fmax'])
            self.log_dict(metrics_dict, prog_bar=True, on_step=False, on_epoch=True)
        super().validation_epoch_end(outputs)

    def test_epoch_end(self, outputs):
        if hasattr(self, 'test_metrics'):
            metrics_dict = self.test_metrics.compute_metrics()
            self.test_metrics.reset_metrics()
            metrics_dict = add_common_metrics(metrics_dict, prefix='test_', metrics_suffixes=['aupr', 'fmax'])
            self.log_dict(metrics_dict, prog_bar=True, on_step=False, on_epoch=True)
        super().test_epoch_end(outputs)

    def on_test_end(self) -> None:
        super().on_test_end()
        self.eval()

        targets = []
        scores = []
        nids = []
        for batch in self.test_dataloader():
            input_nodes, seeds, blocks = batch
            logits = self.forward(to_device(blocks, self.device))
            score = torch.sigmoid(logits)

            targets.append(blocks[-1].dstdata["label"])
            scores.append(score)
            nids.append(blocks[-1].dstdata["_ID"])

        targets = torch.cat(targets, dim=0).detach().cpu().numpy()
        scores = torch.cat(scores, dim=0).detach().cpu().numpy()
        nids = torch.cat(nids, dim=0).detach().cpu().numpy()

        if self.wandb_experiment is not None:
            (fmax_, t_), aupr_ = fmax(targets, scores), pair_aupr(targets, scores)
            final_metrics = {f"test_{self.metric_prefix}_fmax": fmax_, f"test_{self.metric_prefix}_aupr": aupr_}
            final_metrics = add_common_metrics(final_metrics, prefix='test_', metrics_suffixes=['aupr', 'fmax'])
            print('final_metrics', final_metrics)
            self.wandb_experiment.log(final_metrics | {'epoch': self.current_epoch})

            # Plot PR curve
            if hasattr(self, 'node_namespace'):
                mask = nids < self.node_namespace.size
                self.plot_pr_curve(targets[mask], scores[mask],
                                   split_samples=self.node_namespace.iloc[nids[mask]]['species_id'])

    def train_dataloader(self, batch_size=None, num_workers=0, **kwargs):
        neighbor_sampler = dgl.dataloading.MultiLayerFullNeighborSampler(num_layers=self.model.num_gcn)

        collator = dgl.dataloading.NodeCollator(self.dgl_graph, nids=self.training_idx, graph_sampler=neighbor_sampler)
        dataloader = DataLoader(collator.dataset, collate_fn=collator.collate,
                                batch_size=batch_size if batch_size else self.batch_size, shuffle=True, drop_last=False,
                                num_workers=num_workers, **kwargs)
        return dataloader

    def val_dataloader(self, batch_size=None, num_workers=0, **kwargs):
        neighbor_sampler = dgl.dataloading.MultiLayerFullNeighborSampler(num_layers=self.model.num_gcn)

        collator = dgl.dataloading.NodeCollator(self.dgl_graph, nids=self.validation_idx,
                                                graph_sampler=neighbor_sampler)
        dataloader = DataLoader(collator.dataset, collate_fn=collator.collate,
                                batch_size=batch_size if batch_size else self.batch_size, shuffle=False,
                                drop_last=False,
                                num_workers=num_workers, **kwargs)
        return dataloader

    def test_dataloader(self, batch_size=None, num_workers=0, **kwargs):
        neighbor_sampler = dgl.dataloading.MultiLayerFullNeighborSampler(num_layers=self.model.num_gcn)

        collator = dgl.dataloading.NodeCollator(self.dgl_graph, nids=self.testing_idx, graph_sampler=neighbor_sampler)
        dataloader = DataLoader(collator.dataset, collate_fn=collator.collate,
                                batch_size=batch_size if batch_size else self.batch_size, shuffle=False,
                                drop_last=False,
                                num_workers=num_workers, **kwargs)
        return dataloader

    def configure_optimizers(self):
        weight_decay = self.hparams.weight_decay if 'weight_decay' in self.hparams else 1e-2
        lr_annealing = self.hparams.lr_annealing if "lr_annealing" in self.hparams else None
        lr = self.hparams.lr if 'lr' in self.hparams else 1e-3

        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

        return {"optimizer": optimizer}

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)

    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_path))


def main(data_cnf, model_cnf, mode, model_id):
    if not isinstance(data_cnf, dict):
        data_cnf = YAML(typ='safe').load(Path(data_cnf))
    if not isinstance(model_cnf, dict):
        model_cnf = YAML(typ='safe').load(Path(model_cnf))

    model_id = F'-Model-{model_id}' if model_id is not None else ''
    data_name, model_name = data_cnf['name'], model_cnf['name']
    run_name = F'{model_name}{model_id}-{data_name}'
    dgl_graph, node_feats, net_pid_list, train_idx, valid_idx, test_idx = load_dgl_graph(data_cnf, model_cnf,
                                                                                         model_id=model_id)
    net_pid_map = {pid: i for i, pid in enumerate(net_pid_list)}

    if mode is None or mode == 'train':
        train_pid_list, train_seqs, train_go_labels = get_data(**data_cnf['train'])
        valid_pid_list, valid_seqs, valid_go_labels = get_data(**data_cnf['valid'])
        mlb = get_mlb(data_cnf['mlb'], train_go_labels)
        num_labels = len(mlb.classes_)

        train_y, valid_y = mlb.transform(train_go_labels).astype(np.float32), mlb.transform(valid_go_labels).astype(
            np.float32)
        *_, train_idx, train_y = get_ppi_idx(train_pid_list, train_y, net_pid_map)
        *_, valid_ppi, valid_y = get_homo_ppi_idx(pid_list=valid_pid_list, fasta_file=data_cnf['valid']['fasta_file'],
                                                  data_y=valid_y, net_pid_map=net_pid_map,
                                                  net_blastdb=data_cnf['network']['blastdb'],
                                                  blast_output_path=data_cnf[
                                                                        'results'] / F'{data_name}-valid-ppi-blast-out')
        logger.info(F'Number of Labels: {num_labels}')
        logger.info(F'Size of Training Set: {len(train_idx)}')
        logger.info(F'Size of Validation Set: {len(valid_idx)}')

        model = DeepGraphGO(labels_num=num_labels, dgl_graph=dgl_graph, node_feats=node_feats,
                            input_size=node_feats.shape[1], **model_cnf['model'])
        model.train((train_idx, train_y), (valid_idx, valid_y), **model_cnf['train'])

    if mode is None or mode == 'eval':
        mlb = get_mlb(data_cnf['mlb'])
        num_labels = len(mlb.classes_)

        if model is None:
            model = DeepGraphGO(labels_num=num_labels, dgl_graph=dgl_graph, node_feats=node_feats,
                                input_size=node_feats.shape[1], **model_cnf['model'])

        test_name = data_cnf['test'].pop('name')
        test_pid_list, _, test_go_labels = get_data(**data_cnf['test'])
        test_res_idx_, test_pid_list_, test_ppi, _ = get_homo_ppi_idx(
            pid_list=test_pid_list, fasta_file=data_cnf['test']['fasta_file'], data_y=None, net_pid_map=net_pid_map,
            net_blastdb=data_cnf['network']['blastdb'],
            blast_output_path=data_cnf['results'] / F'{data_name}-{test_name}-ppi-blast-out')
        scores = np.zeros((len(test_pid_list), len(mlb.classes_)))
        scores[test_res_idx_] = model.predict(test_ppi, **model_cnf['test'])
        res_path = data_cnf['results'] / F'{run_name}-{test_name}'
        output_res(res_path.with_suffix('.txt'), test_pid_list, mlb.classes_, scores)
        np.save(res_path, scores)


def load_dgl_graph(data_cnf, model_cnf, model_id=None, subset_pid: List[str] = None) \
        -> Tuple[dgl.DGLGraph, ssp.csr_matrix, Dict[str, int], Tensor, Tensor, Tensor]:
    # Set paths
    data_name, model_name = data_cnf['name'], model_cnf['name']
    run_name = F'{model_name}{model_id}-{data_name}'

    model_cnf['model']['model_path'] = Path(data_cnf['model_path']) / F'{run_name}'
    data_cnf['mlb'] = Path(data_cnf['mlb'])
    data_cnf['results'] = Path(data_cnf['results'])

    # Get all protein node id lists
    net_pid_list = get_pid_list(data_cnf['network']['pid_list'])
    net_pid_map = {pid: i for i, pid in enumerate(net_pid_list)}

    # Load DGL Graph
    dgl_graph: dgl.DGLGraph = dgl.data.utils.load_graphs(data_cnf['network']['dgl'])[0][0]
    self_loop = torch.zeros_like(dgl_graph.edata['ppi'])
    nr_ = np.arange(dgl_graph.number_of_nodes())
    self_loop[dgl_graph.edge_ids(nr_, nr_)] = 1.0

    dgl_graph.edata['self'] = self_loop
    dgl_graph.edata['ppi'] = dgl_graph.edata['ppi'].float()
    dgl_graph.edata['self'] = dgl_graph.edata['self'].float()

    # Node features
    node_feats = ssp.load_npz(data_cnf['network']['feature'])

    if subset_pid is not None:
        subset_pid = [node for node in subset_pid if node in net_pid_map]
        node_ids = torch.tensor([net_pid_map[node] for node in subset_pid], dtype=torch.int64)
        dgl_graph = dgl.node_subgraph(dgl_graph, nodes=node_ids)
        node_feats = node_feats[node_ids.numpy()]
        net_pid_map = {pid: i for i, pid in enumerate(subset_pid)}
        net_pid_list = subset_pid

    assert dgl_graph.num_nodes() == len(net_pid_list), f"{dgl_graph.num_nodes()} != {len(net_pid_list)}"
    logger.info(F'{dgl_graph}')

    # Get train/valid/test split of node lists
    train_pid_list, _, train_go_labels = get_data(**data_cnf['train'], subset_pid=subset_pid)
    valid_pid_list, _, valid_go_labels = get_data(**data_cnf['valid'], subset_pid=subset_pid)
    test_pid_list, _, test_go_labels = get_data(**data_cnf['test'], subset_pid=subset_pid)

    # Binarize multilabel GO annotations
    mlb = get_mlb(data_cnf['mlb'], train_go_labels)
    num_labels = len(mlb.classes_)
    train_y, valid_y, test_y = mlb.transform(train_go_labels).astype(np.float32), \
                               mlb.transform(valid_go_labels).astype(np.float32), \
                               mlb.transform(test_go_labels).astype(np.float32)

    # Get train/valid/test index split and labels index_list
    *_, train_idx, train_y = get_ppi_idx(train_pid_list, train_y, net_pid_map)
    *_, valid_idx, valid_y = get_ppi_idx(valid_pid_list, valid_y, net_pid_map)
    *_, test_idx, test_y = get_ppi_idx(test_pid_list, test_y, net_pid_map)

    # Assign labels to dgl_graph
    dgl_graph.ndata["label"] = torch.zeros(dgl_graph.num_nodes(), train_y.shape[1])
    dgl_graph.ndata["label"][train_idx] = torch.tensor(train_y.todense())
    dgl_graph.ndata["label"][valid_idx] = torch.tensor(valid_y.todense())
    dgl_graph.ndata["label"][test_idx] = torch.tensor(test_y.todense())

    return dgl_graph, node_feats, net_pid_list, train_idx, valid_idx, test_idx


def load_protein_dataset(path: str, namespaces: List[str] = ['mf', 'bp', 'cc']) -> pd.DataFrame:
    def get_pid_list(pid_list_file):
        with open(pid_list_file) as fp:
            return [line.split()[0] for line in fp]

    if not any(name in ['mf', 'bp', 'cc'] for name in namespaces):
        rename_dict = {'molecular_function': 'mf', 'biological_process': 'bp', 'cellular_component': 'cc',
                       'mf': 'mf', 'bp': 'bp', 'cc': 'cc'}
        namespaces = [rename_dict[name] for name in namespaces]
        logger.info(f'DGG namespaces: {namespaces}')

    net_pid_list = get_pid_list(os.path.join(path, 'ppi_pid_list.txt'))
    # dgl_graph: dgl.DGLGraph = dgl.data.utils.load_graphs(f'{path}/ppi_dgl_top_100')[0][0]
    # self_loop = torch.zeros_like(dgl_graph.edata['ppi'])
    # nr_ = np.arange(dgl_graph.number_of_nodes())
    # self_loop[dgl_graph.edge_ids(nr_, nr_)] = 1.0

    # dgl_graph.edata['self'] = self_loop
    # dgl_graph.edata['ppi'] = dgl_graph.edata['ppi'].float()
    # dgl_graph.edata['self'] = dgl_graph.edata['self'].float()

    all_pid = set(net_pid_list)
    for namespace in namespaces:
        all_pid = all_pid.union(get_pid_list(os.path.join(path, f'{namespace}_train_pid_list.txt')))
        all_pid = all_pid.union(get_pid_list(os.path.join(path, f'{namespace}_valid_pid_list.txt')))
        all_pid = all_pid.union(get_pid_list(os.path.join(path, f'{namespace}_test_pid_list.txt')))

    uniprot_go_id = pd.DataFrame({"train_mask": False, "valid_mask": False, "test_mask": False, "sequence": None},
                                 index=pd.Index(all_pid, name="protein_id"))
    uniprot_go_id["go_id"] = [[], ] * uniprot_go_id.index.size

    for namespace in namespaces:
        data_cnf = {'mlb': f'{path}/{namespace}_go.mlb',
                    'model_path': 'models',
                    'name': namespace,
                    'network': {'blastdb': f'{path}/ppi_blastdb',
                                'dgl': f'{path}/ppi_dgl_top_100',
                                'feature': f'{path}/ppi_interpro.npz',
                                'pid_list': f'{path}/ppi_pid_list.txt',
                                'weight_mat': f'{path}/ppi_mat.npz'},
                    'results': 'results',
                    'test': {'fasta_file': f'{path}/{namespace}_test.fasta',
                             'name': 'test',
                             'pid_go_file': f'{path}/{namespace}_test_go.txt',
                             'pid_list_file': f'{path}/{namespace}_test_pid_list.txt'},
                    'train': {'fasta_file': f'{path}/{namespace}_train.fasta',
                              'name': 'train',
                              'pid_go_file': f'{path}/{namespace}_train_go.txt',
                              'pid_list_file': f'{path}/{namespace}_train_pid_list.txt'},
                    'valid': {'fasta_file': f'{path}/{namespace}_valid.fasta',
                              'name': 'valid',
                              'pid_go_file': f'{path}/{namespace}_valid_go.txt',
                              'pid_list_file': f'{path}/{namespace}_valid_pid_list.txt'}}

        train_pid_list, train_seqs, train_go_labels = get_data(**data_cnf['train'], subset_pid=None)
        valid_pid_list, valid_seqs, valid_go_labels = get_data(**data_cnf['valid'], subset_pid=None)
        test_pid_list, test_seqs, test_go_labels = get_data(**data_cnf['test'], subset_pid=None)

        uniprot_go_id.loc[train_pid_list, "go_id"] = uniprot_go_id.loc[train_pid_list, "go_id"] + \
                                                     pd.Series(train_go_labels, index=train_pid_list)
        uniprot_go_id.loc[valid_pid_list, "go_id"] = uniprot_go_id.loc[valid_pid_list, "go_id"] + \
                                                     pd.Series(valid_go_labels, index=valid_pid_list)
        uniprot_go_id.loc[test_pid_list, "go_id"] = uniprot_go_id.loc[test_pid_list, "go_id"] + \
                                                    pd.Series(test_go_labels, index=test_pid_list)

        # Set train/valid/test protein ids from False to True
        uniprot_go_id.loc[train_pid_list, "train_mask"] = True
        uniprot_go_id.loc[valid_pid_list, "valid_mask"] = True
        uniprot_go_id.loc[test_pid_list, "test_mask"] = True

        uniprot_go_id.loc[train_pid_list, "sequence"] = train_seqs
        uniprot_go_id.loc[valid_pid_list, "sequence"] = valid_seqs
        uniprot_go_id.loc[test_pid_list, "sequence"] = test_seqs

    # If node not in either train/valid/test, then set mask to True on all train/valid/test
    mask_cols = ['train_mask', 'valid_mask', 'test_mask']
    uniprot_go_id.loc[~uniprot_go_id[mask_cols].any(axis=1), mask_cols] = uniprot_go_id.loc[
        ~uniprot_go_id[mask_cols].any(axis=1), mask_cols].replace({'train_mask': {False: True}})
    # uniprot_go_id['go_id'] = uniprot_go_id['go_id'].map(
    #     lambda li: None if isinstance(li, list) and len(li) == 0 else li)
    uniprot_go_id['go_id'] = uniprot_go_id['go_id'].map(lambda d: d if isinstance(d, Iterable) else [])

    return uniprot_go_id
