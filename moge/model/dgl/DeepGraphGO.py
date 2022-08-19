import os.path
import warnings
from argparse import Namespace
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Tuple

import dgl
import dgl.function as fn
import joblib
import numpy as np
import scipy.sparse as ssp
import torch
import torch.nn.functional as F
from Bio import SeqIO
from Bio.Blast import NCBIXML
from Bio.Blast.Applications import NcbipsiblastCommandline
from dgl.heterograph import DGLBlock
from dgl.udf import NodeBatch
from logzero import logger
from moge.model.metrics import Metrics
from pytorch_lightning import LightningModule
from ruamel.yaml import YAML
from sklearn.metrics import average_precision_score as aupr
from sklearn.preprocessing import MultiLabelBinarizer
from torch import nn, Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm


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


def get_data(fasta_file, pid_go_file=None, feature_type=None, **kwargs):
    pid_list, seqs = [], []
    for seq in SeqIO.parse(fasta_file, 'fasta'):
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


def fmax(targets: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
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


def pair_aupr(targets: ssp.csr_matrix, scores: np.ndarray, top=200):
    scores[np.arange(scores.shape[0])[:, None],
           scores.argpartition(scores.shape[1] - top)[:, :-top]] = -1e100
    return aupr(targets.toarray().flatten(), scores.flatten())


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

    def __init__(self, n_classes, input_size, hidden_size, num_gcn=0, dropout=0.5, residual=True, **kwargs):
        super().__init__()
        logger.info(F'GCN: labels_num={n_classes}, input size={input_size}, hidden_size={hidden_size}, '
                    F'num_gcn={num_gcn}, dropout={dropout}, residual={residual}')
        self.embedding = nn.EmbeddingBag(input_size, hidden_size, mode='sum', include_last_offset=True)
        self.input_bias = nn.Parameter(torch.zeros(hidden_size))

        self.dropout = nn.Dropout(dropout)

        self.layers: List[NodeUpdate] = nn.ModuleList([
            NodeUpdate(hidden_size, hidden_size, dropout) for _ in range(num_gcn)])

        self.n_classes = n_classes
        self.output = nn.Linear(hidden_size, self.n_classes)
        self.residual = residual
        self.num_gcn = num_gcn
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        for update in self.layers:
            update.reset_parameters()
        nn.init.xavier_uniform_(self.output.weight)

    def forward(self, blocks: List[DGLBlock], input, offets, per_sample_weights):
        h = self.embedding.forward(input, offets, per_sample_weights) + self.input_bias
        h = self.dropout(F.relu(h))

        for i, layer in enumerate(self.layers):
            blocks[i].srcdata['h'] = h
            if self.residual:
                blocks[i].update_all(fn.u_mul_e('h', 'self', out='m_res'),
                                     fn.sum(msg='m_res', out='res'))
            blocks[i].update_all(fn.u_mul_e('h', 'ppi', out='ppi_m_out'),
                                 fn.sum(msg='ppi_m_out', out='ppi_out'), layer)
            h = blocks[i].dstdata['h']

        h = self.output(h)
        return h


class DeepGraphGO(LightningModule):
    def __init__(self, hparams: Namespace, model_path: Path, dgl_graph: dgl.DGLGraph, node_feats: ssp.csr_matrix,
                 metrics: List[str]):
        if not isinstance(hparams, Namespace) and isinstance(hparams, dict):
            hparams = Namespace(**hparams)
        super().__init__()

        self.train_metrics = Metrics(prefix="", loss_type='BCE_WITH_LOGITS', n_classes=hparams.n_classes,
                                     multilabel=True, metrics=metrics)
        self.valid_metrics = Metrics(prefix="val_", loss_type='BCE_WITH_LOGITS', n_classes=hparams.n_classes,
                                     multilabel=True, metrics=metrics)
        self.test_metrics = Metrics(prefix="test_", loss_type='BCE_WITH_LOGITS', n_classes=hparams.n_classes,
                                    multilabel=True, metrics=metrics)

        self.model = GcnNet(**hparams.__dict__)

        model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model_path = model_path
        self.criterion = nn.BCEWithLogitsLoss()
        self.dgl_graph, self.node_feats, self.batch_size = dgl_graph, node_feats, hparams.batch_size

        self.training_idx, self.validation_idx, self.testing_idx = hparams.training_idx, hparams.validation_idx, hparams.testing_idx

        self._set_hparams(hparams)

    def forward(self, blocks: List[DGLBlock]):
        node_ids = blocks[0].srcdata["_ID"]
        batch_x = self.node_feats[node_ids.cpu().numpy()]

        input = torch.from_numpy(batch_x.indices).to(self.device).long()
        offsets = torch.from_numpy(batch_x.indptr).to(self.device).long()
        per_sample_weights = torch.from_numpy(batch_x.data).to(self.device).float()

        logits = self.model.forward(blocks, input, offsets, per_sample_weights)

        return logits

    def training_step(self, batch, batch_nb):
        input_nodes, seeds, blocks = batch
        y_true = blocks[-1].dstdata["label"]

        logits = self.forward(blocks)
        loss = self.criterion(logits, y_true)
        self.log("loss", loss, logger=True, prog_bar=True, on_step=False, on_epoch=True)

        self.train_metrics.update_metrics(torch.sigmoid(logits), y_true)
        scores = torch.sigmoid(logits).detach().cpu().numpy()
        y_true = y_true.detach().cpu().numpy()
        (fmax_, t_), aupr_ = fmax(y_true, scores), aupr(y_true.flatten(), scores.flatten())
        self.log_dict({"fmax": fmax_, "aupr": aupr_}, logger=True, prog_bar=True,
                      on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_nb):
        input_nodes, seeds, blocks = batch
        y_true = blocks[-1].dstdata["label"]

        logits = self.forward(blocks)
        loss = self.criterion(logits, y_true)
        self.log("val_loss", loss, logger=True, on_step=False, on_epoch=True)

        self.valid_metrics.update_metrics(torch.sigmoid(logits), y_true)
        scores = torch.sigmoid(logits).detach().cpu().numpy()
        y_true = y_true.detach().cpu().numpy()
        (fmax_, t_), aupr_ = fmax(y_true, scores), aupr(y_true.flatten(), scores.flatten())
        self.log_dict({"val_fmax": fmax_, "val_aupr": aupr_}, logger=True, prog_bar=True,
                      on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_nb):
        input_nodes, seeds, blocks = batch
        y_true = blocks[-1].dstdata["label"]

        logits = self.forward(blocks)
        loss = self.criterion(logits, y_true)
        self.log("test_loss", loss, logger=True, on_step=False, on_epoch=True)

        self.test_metrics.update_metrics(torch.sigmoid(logits), y_true)
        # scores = torch.sigmoid(logits).detach().cpu().numpy()
        # y_true = y_true.detach().cpu().numpy()
        # (fmax_, t_), aupr_ = fmax(y_true, scores), aupr(y_true.flatten(), scores.flatten())
        # self.log_dict({"test_fmax": fmax_, "test_aupr": aupr_}, logger=True, prog_bar=True,
        #               on_step=False, on_epoch=True)
        return loss

    # def train(self, train_data:Tuple[np.ndarray, ssp.csr_matrix], valid_data:Tuple[np.ndarray, ssp.csr_matrix],
    #           loss_params=(), opt_params=(), epochs_num=10, batch_size=40, **kwargs):
    #     self.get_optimizer(**dict(opt_params))
    #     self.batch_size = batch_size
    #
    #     (train_ppi, train_y), (valid_ppi, valid_y) = train_data, valid_data
    #     ppi_train_idx = np.full(self.node_feats.shape[0], -1, dtype=np.int)
    #     ppi_train_idx[train_ppi] = np.arange(train_ppi.shape[0])
    #     best_fmax = 0.0
    #     for epoch_idx in range(epochs_num):
    #         train_loss = 0.0
    #         for nf in tqdm(dgl.contrib.sampling.sampler.NeighborSampler(self.dgl_graph, batch_size,
    #                                                                     self.dgl_graph.number_of_nodes(),
    #                                                                     num_hops=self.model.num_gcn,
    #                                                                     seed_nodes=train_ppi,
    #                                                                     prefetch=True, shuffle=True),
    #                        desc=F'Epoch {epoch_idx}', leave=False, dynamic_ncols=True,
    #                        total=(len(train_ppi) + batch_size - 1) // batch_size):
    #             batch_y = train_y[ppi_train_idx[nf.layer_parent_nid(-1).numpy()]].toarray()
    #
    #             train_loss += self.train_step(train_x=nf, train_y=torch.from_numpy(batch_y), update=True)
    #         best_fmax = self.valid_step(valid_loader=valid_ppi, targets=valid_y, epoch_idx=epoch_idx,
    #                                     train_loss=train_loss / len(train_ppi), best_fmax=best_fmax)

    @torch.no_grad()
    def predict_step(self, data_x):
        self.model.eval()
        input_nodes, seeds, blocks = data_x

        return torch.sigmoid(self.forward(blocks)).cpu().numpy()

    def training_epoch_end(self, outputs):
        metrics_dict = self.train_metrics.compute_metrics()
        self.train_metrics.reset_metrics()
        self.log_dict(metrics_dict, prog_bar=True)

    def validation_epoch_end(self, outputs):
        metrics_dict = self.valid_metrics.compute_metrics()
        self.valid_metrics.reset_metrics()
        self.log_dict(metrics_dict, prog_bar=True)

    def test_epoch_end(self, outputs):
        metrics_dict = self.test_metrics.compute_metrics()
        self.test_metrics.reset_metrics()
        self.log_dict(metrics_dict, prog_bar=True)

    def train_dataloader(self):
        neighbor_sampler = dgl.dataloading.MultiLayerFullNeighborSampler(num_layers=self.model.num_gcn)

        collator = dgl.dataloading.NodeCollator(self.dgl_graph, nids=self.training_idx,
                                                graph_sampler=neighbor_sampler)
        dataloader = DataLoader(collator.dataset, collate_fn=collator.collate,
                                batch_size=self.batch_size, shuffle=True, drop_last=False, )
        return dataloader

    def val_dataloader(self):
        neighbor_sampler = dgl.dataloading.MultiLayerFullNeighborSampler(num_layers=self.model.num_gcn)

        collator = dgl.dataloading.NodeCollator(self.dgl_graph, nids=self.validation_idx,
                                                graph_sampler=neighbor_sampler)
        dataloader = DataLoader(collator.dataset, collate_fn=collator.collate,
                                batch_size=self.batch_size, shuffle=True, drop_last=False, )
        return dataloader

    def test_dataloader(self):
        neighbor_sampler = dgl.dataloading.MultiLayerFullNeighborSampler(num_layers=self.model.num_gcn)

        collator = dgl.dataloading.NodeCollator(self.dgl_graph, nids=self.testing_idx,
                                                graph_sampler=neighbor_sampler)
        dataloader = DataLoader(collator.dataset, collate_fn=collator.collate,
                                batch_size=self.batch_size, shuffle=True, drop_last=False, )
        return dataloader

    def configure_optimizers(self):
        weight_decay = self.hparams.weight_decay if 'weight_decay' in self.hparams else 0.0
        lr_annealing = self.hparams.lr_annealing if "lr_annealing" in self.hparams else None

        optimizer = torch.optim.AdamW(self.model.parameters(),
                                      # lr=self.hparams.lr
                                      )

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
    dgl_graph, node_feats, net_pid_map, train_idx, valid_idx, test_idx = load_dgl_graph(data_cnf, model_cnf,
                                                                                        model_id=model_id)

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


def load_dgl_graph(data_cnf, model_cnf, model_id=None) \
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
    dgl_graph = dgl.data.utils.load_graphs(data_cnf['network']['dgl'])[0][0]
    self_loop = torch.zeros_like(dgl_graph.edata['ppi'])
    nr_ = np.arange(dgl_graph.number_of_nodes())
    self_loop[dgl_graph.edge_ids(nr_, nr_)] = 1.0

    dgl_graph.edata['self'] = self_loop
    dgl_graph.edata['ppi'] = dgl_graph.edata['ppi'].float()
    dgl_graph.edata['self'] = dgl_graph.edata['self'].float()

    logger.info(F'{dgl_graph}')
    # Node features
    node_feats = ssp.load_npz(data_cnf['network']['feature'])

    # Get train/valid/test split of node lists
    train_pid_list, _, train_go_labels = get_data(**data_cnf['train'])
    valid_pid_list, _, valid_go_labels = get_data(**data_cnf['valid'])
    test_pid_list, _, test_go_labels = get_data(**data_cnf['test'])

    # Binarize multilabel GO annotations
    mlb = get_mlb(data_cnf['mlb'], train_go_labels)
    num_labels = len(mlb.classes_)
    train_y, valid_y, test_y = mlb.transform(train_go_labels).astype(np.float32), mlb.transform(valid_go_labels).astype(
        np.float32), mlb.transform(test_go_labels).astype(np.float32)

    # Get train/valid/test index split and labels index_list
    *_, train_idx, train_y = get_ppi_idx(train_pid_list, train_y, net_pid_map)
    *_, valid_idx, valid_y = get_ppi_idx(valid_pid_list, valid_y, net_pid_map)
    *_, test_idx, test_y = get_ppi_idx(test_pid_list, test_y, net_pid_map)

    # Assign labels to dgl_graph
    dgl_graph.ndata["label"] = torch.zeros(dgl_graph.num_nodes(), train_y.shape[1])
    dgl_graph.ndata["label"][train_idx] = torch.tensor(train_y.todense())
    dgl_graph.ndata["label"][valid_idx] = torch.tensor(valid_y.todense())
    dgl_graph.ndata["label"][test_idx] = torch.tensor(test_y.todense())

    return dgl_graph, node_feats, net_pid_map, train_idx, valid_idx, test_idx
