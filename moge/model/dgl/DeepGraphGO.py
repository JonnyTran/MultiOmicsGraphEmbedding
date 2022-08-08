import warnings
from collections import defaultdict
from pathlib import Path

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
from logzero import logger
from ruamel_yaml import YAML
from sklearn.metrics import average_precision_score as aupr
from sklearn.preprocessing import MultiLabelBinarizer
from torch import nn
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
    pid_list, data_x = [], []
    for seq in SeqIO.parse(fasta_file, 'fasta'):
        pid_list.append(seq.id)
        data_x.append(str(seq.seq))
    if feature_type is not None:
        feature_path = Path(kwargs[feature_type])
        if feature_path.suffix == '.npy':
            data_x = np.load(feature_path)
        elif feature_path.suffix == '.npz':
            data_x = ssp.load_npz(feature_path)
        else:
            raise ValueError(F'Only support suffix of .npy for np.ndarray or .npz for scipy.csr_matrix as feature.')
    return pid_list, data_x, get_go_list(pid_go_file, pid_list)


def get_ppi_idx(pid_list, data_y, net_pid_map):
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


def blast(*args, **kwargs):
    return psiblast(*args, **kwargs, num_iterations=1)


def get_homo_ppi_idx(pid_list, fasta_file, data_y, net_pid_map, net_blastdb, blast_output_path):
    blast_sim = blast(net_blastdb, pid_list, fasta_file, blast_output_path)
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
    if mlb_path.exists():
        return joblib.load(mlb_path)
    mlb = MultiLabelBinarizer(sparse_output=True, **kwargs)
    mlb.fit(labels)
    joblib.dump(mlb, mlb_path)
    return mlb


def fmax(targets: ssp.csr_matrix, scores: np.ndarray):
    fmax_ = 0.0, 0.0
    for cut in (c / 100 for c in range(101)):
        cut_sc = ssp.csr_matrix((scores >= cut).astype(np.int32))
        correct = cut_sc.multiply(targets).sum(axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            p, r = correct / cut_sc.sum(axis=1), correct / targets.sum(axis=1)
            p, r = np.average(p[np.invert(np.isnan(p))]), np.average(r)
        if np.isnan(p):
            continue
        try:
            fmax_ = max(fmax_, (2 * p * r / (p + r) if p + r > 0.0 else 0.0, cut))
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
    """

    """

    def __init__(self, in_f, out_f, dropout: float):
        super(NodeUpdate, self).__init__()
        self.ppi_linear = nn.Linear(in_f, out_f)
        self.dropout = nn.Dropout(dropout)

    def forward(self, node):
        outputs = self.dropout(F.relu(self.ppi_linear(node.data['ppi_out'])))
        if 'res' in node.data:
            outputs = outputs + node.data['res']
        return {'h': outputs}

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.ppi_linear.weight)


class GcnNet(nn.Module):
    """
    """

    def __init__(self, *, labels_num, input_size, hidden_size, num_gcn=0, dropout=0.5, residual=True,
                 **kwargs):
        super(GcnNet, self).__init__()
        logger.info(F'GCN: labels_num={labels_num}, input size={input_size}, hidden_size={hidden_size}, '
                    F'num_gcn={num_gcn}, dropout={dropout}, residual={residual}')
        self.labels_num = labels_num
        self.input = nn.EmbeddingBag(input_size, hidden_size, mode='sum', include_last_offset=True)
        self.input_bias = nn.Parameter(torch.zeros(hidden_size))
        self.dropout = nn.Dropout(dropout)
        self.update = nn.ModuleList(NodeUpdate(hidden_size, hidden_size, dropout) for _ in range(num_gcn))
        self.output = nn.Linear(hidden_size, self.labels_num)
        self.residual = residual
        self.num_gcn = num_gcn
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.input.weight)
        for update in self.update:
            update.reset_parameters()
        nn.init.xavier_uniform_(self.output.weight)

    def forward(self, nf: dgl.NodeFlow, inputs):
        nf.copy_from_parent()
        outputs = self.dropout(F.relu(self.input(*inputs) + self.input_bias))
        nf.layers[0].data['h'] = outputs
        for i, update in enumerate(self.update):
            if self.residual:
                nf.block_compute(i,
                                 fn.u_mul_e('h', 'self', out='m_res'),
                                 fn.sum(msg='m_res', out='res'))
            nf.block_compute(i,
                             fn.u_mul_e('h', 'ppi', out='ppi_m_out'),
                             fn.sum(msg='ppi_m_out', out='ppi_out'), update)
        return self.output(nf.layers[-1].data['h'])


class DeepGraphGO(object):
    """

    """

    def __init__(self, *, model_path: Path, dgl_graph, network_x, **kwargs):
        self.model = self.network = GcnNet(**kwargs)
        self.dp_network = nn.DataParallel(self.network.cuda())
        model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model_path = model_path
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.optimizer = None
        self.dgl_graph, self.network_x, self.batch_size = dgl_graph, network_x, None

    def get_scores(self, nf: dgl.NodeFlow):
        batch_x = self.network_x[nf.layer_parent_nid(0).numpy()]
        scores = self.network.forward(nf, (torch.from_numpy(batch_x.indices).cuda().long(),
                                           torch.from_numpy(batch_x.indptr).cuda().long(),
                                           torch.from_numpy(batch_x.data).cuda().float()))
        return scores

    def get_optimizer(self, **kwargs):
        self.optimizer = torch.optim.AdamW(self.model.parameters(), **kwargs)

    def train_step(self, train_x, train_y, update, **kwargs):
        self.model.train()
        scores = self.get_scores(train_x)
        loss = self.loss_fn(scores, train_y.cuda())
        loss.backward()
        if update:
            self.optimizer.step(closure=None)
            self.optimizer.zero_grad()
        return loss.item()

    def train(self, train_data, valid_data, loss_params=(), opt_params=(), epochs_num=10, batch_size=40, **kwargs):
        self.get_optimizer(**dict(opt_params))
        self.batch_size = batch_size

        (train_ppi, train_y), (valid_ppi, valid_y) = train_data, valid_data
        ppi_train_idx = np.full(self.network_x.shape[0], -1, dtype=np.int)
        ppi_train_idx[train_ppi] = np.arange(train_ppi.shape[0])
        best_fmax = 0.0
        for epoch_idx in range(epochs_num):
            train_loss = 0.0
            for nf in tqdm(dgl.contrib.sampling.sampler.NeighborSampler(self.dgl_graph, batch_size,
                                                                        self.dgl_graph.number_of_nodes(),
                                                                        num_hops=self.model.num_gcn,
                                                                        seed_nodes=train_ppi,
                                                                        prefetch=True, shuffle=True),
                           desc=F'Epoch {epoch_idx}', leave=False, dynamic_ncols=True,
                           total=(len(train_ppi) + batch_size - 1) // batch_size):
                batch_y = train_y[ppi_train_idx[nf.layer_parent_nid(-1).numpy()]].toarray()
                train_loss += self.train_step(train_x=nf, train_y=torch.from_numpy(batch_y), update=True)
            best_fmax = self.valid(valid_loader=valid_ppi, targets=valid_y, epoch_idx=epoch_idx,
                                   train_loss=train_loss / len(train_ppi), best_fmax=best_fmax)

    def valid(self, valid_loader, targets, epoch_idx, train_loss, best_fmax):
        scores = self.predict(valid_loader, valid=True)
        (fmax_, t_), aupr_ = fmax(targets, scores), aupr(targets.toarray().flatten(), scores.flatten())
        logger.info(F'Epoch {epoch_idx}: Loss: {train_loss:.5f} '
                    F'Fmax: {fmax_:.3f} {t_:.2f} AUPR: {aupr_:.3f}')
        if fmax_ > best_fmax:
            best_fmax = fmax_
            self.save_model()
        return best_fmax

    @torch.no_grad()
    def predict_step(self, data_x):
        self.model.eval()
        return torch.sigmoid(self.get_scores(data_x)).cpu().numpy()

    def predict(self, test_ppi, batch_size=None, valid=False, **kwargs):
        if batch_size is None:
            batch_size = self.batch_size
        if not valid:
            self.load_model()
        unique_test_ppi = np.unique(test_ppi)
        mapping = {x: i for i, x in enumerate(unique_test_ppi)}
        test_ppi = np.asarray([mapping[x] for x in test_ppi])
        scores = np.vstack([self.predict_step(nf)
                            for nf in dgl.contrib.sampling.sampler.NeighborSampler(self.dgl_graph, batch_size,
                                                                                   self.dgl_graph.number_of_nodes(),
                                                                                   num_hops=self.model.num_gcn,
                                                                                   seed_nodes=unique_test_ppi,
                                                                                   prefetch=True)])
        return scores[test_ppi]

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)

    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_path))


def main(data_cnf, model_cnf, mode, model_id):
    model_id = F'-Model-{model_id}' if model_id is not None else ''
    yaml = YAML(typ='safe')
    data_cnf, model_cnf = yaml.load(Path(data_cnf)), yaml.load(Path(model_cnf))
    data_name, model_name = data_cnf['name'], model_cnf['name']
    run_name = F'{model_name}{model_id}-{data_name}'
    model, model_cnf['model']['model_path'] = None, Path(data_cnf['model_path']) / F'{run_name}'
    data_cnf['mlb'] = Path(data_cnf['mlb'])
    data_cnf['results'] = Path(data_cnf['results'])
    logger.info(F'Model: {model_name}, Path: {model_cnf["model"]["model_path"]}, Dataset: {data_name}')

    net_pid_list = get_pid_list(data_cnf['network']['pid_list'])
    net_pid_map = {pid: i for i, pid in enumerate(net_pid_list)}
    net_blastdb = data_cnf['network']['blastdb']
    dgl_graph = dgl.data.utils.load_graphs(data_cnf['network']['dgl'])[0][0]
    self_loop = torch.zeros_like(dgl_graph.edata['ppi'])

    nr_ = np.arange(dgl_graph.number_of_nodes())
    self_loop[dgl_graph.edge_ids(nr_, nr_)] = 1.0
    dgl_graph.edata['self'] = self_loop
    dgl_graph.edata['ppi'] = dgl_graph.edata['ppi'].float().cuda()
    dgl_graph.edata['self'] = dgl_graph.edata['self'].float().cuda()
    logger.info(F'{dgl_graph}')
    network_x = ssp.load_npz(data_cnf['network']['feature'])

    if mode is None or mode == 'train':
        train_pid_list, _, train_go = get_data(**data_cnf['train'])
        valid_pid_list, _, valid_go = get_data(**data_cnf['valid'])
        mlb = get_mlb(data_cnf['mlb'], train_go)
        labels_num = len(mlb.classes_)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            train_y, valid_y = mlb.transform(train_go).astype(np.float32), mlb.transform(valid_go).astype(np.float32)
        *_, train_ppi, train_y = get_ppi_idx(train_pid_list, train_y, net_pid_map)
        *_, valid_ppi, valid_y = get_homo_ppi_idx(valid_pid_list, data_cnf['valid']['fasta_file'],
                                                  valid_y, net_pid_map, net_blastdb,
                                                  data_cnf['results'] / F'{data_name}-valid-ppi-blast-out')
        logger.info(F'Number of Labels: {labels_num}')
        logger.info(F'Size of Training Set: {len(train_ppi)}')
        logger.info(F'Size of Validation Set: {len(valid_ppi)}')

        model = Model(labels_num=labels_num, dgl_graph=dgl_graph, network_x=network_x,
                      input_size=network_x.shape[1], **model_cnf['model'])
        model.train((train_ppi, train_y), (valid_ppi, valid_y), **model_cnf['train'])

    if mode is None or mode == 'eval':
        mlb = get_mlb(data_cnf['mlb'])
        labels_num = len(mlb.classes_)
        if model is None:
            model = Model(labels_num=labels_num, dgl_graph=dgl_graph, network_x=network_x,
                          input_size=network_x.shape[1], **model_cnf['model'])
        test_cnf = data_cnf['test']
        test_name = test_cnf.pop('name')
        test_pid_list, _, test_go = get_data(**test_cnf)
        test_res_idx_, test_pid_list_, test_ppi, _ = get_homo_ppi_idx(test_pid_list, test_cnf['fasta_file'],
                                                                      None, net_pid_map, net_blastdb,
                                                                      data_cnf['results'] / F'{data_name}-{test_name}'
                                                                                            F'-ppi-blast-out')
        scores = np.zeros((len(test_pid_list), len(mlb.classes_)))
        scores[test_res_idx_] = model.predict(test_ppi, **model_cnf['test'])
        res_path = data_cnf['results'] / F'{run_name}-{test_name}'
        output_res(res_path.with_suffix('.txt'), test_pid_list, mlb.classes_, scores)
        np.save(res_path, scores)
