import math
from argparse import Namespace
from collections import Counter, deque
from typing import Dict, List, Tuple

import pandas as pd
import torch as th
from torch import nn, Tensor, LongTensor
from torch.nn import functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch_sparse import SparseTensor

from moge.dataset.PyG.hetero_generator import HeteroNodeClfDataset
from moge.model.trainer import NodeClfTrainer
from moge.model.utils import concat_dict_batch, filter_samples_weights, to_device


class DeepGOZero(NodeClfTrainer):
    def __init__(self, hparams: Namespace,
                 dataset: HeteroNodeClfDataset,
                 metrics: List[str] = ["aupr", "fmax"], *args, **kwargs):
        if not isinstance(hparams, Namespace) and isinstance(hparams, dict):
            hparams = Namespace(**hparams)
        super().__init__(hparams, dataset, metrics, *args, **kwargs)
        self.dataset = dataset
        self.head_node_type = dataset.head_node_type
        self.load_data(hparams, dataset)

        self.nb_gos = n_classes = dataset.n_classes
        input_length = dataset.node_attr_size
        hidden_dim = getattr(hparams, 'embedding_dim', 1024)
        embed_dim = getattr(hparams, 'embedding_dim', 1024)
        margin = 0.1
        net = []
        net.append(MLPBlock(input_length, hidden_dim))
        net.append(Residual(MLPBlock(hidden_dim, hidden_dim)))
        self.net = nn.Sequential(*net)

        # ELEmbeddings
        self.embed_dim = embed_dim
        self.hasFuncIndex = th.LongTensor([self.n_rels])
        self.go_embed = nn.Embedding(n_classes + self.nb_zero_gos, embed_dim)
        self.go_norm = nn.BatchNorm1d(embed_dim)
        k = math.sqrt(1 / embed_dim)
        nn.init.uniform_(self.go_embed.weight, -k, k)
        self.go_rad = nn.Embedding(n_classes + self.nb_zero_gos, 1)
        nn.init.uniform_(self.go_rad.weight, -k, k)

        self.rel_embed = nn.Embedding(self.n_rels + 1, embed_dim)
        nn.init.uniform_(self.rel_embed.weight, -k, k)
        self.all_gos = th.arange(self.nb_gos)
        self.margin = margin

    def load_data(self, hparams, dataset, go_file='../deepgozero/data/go.norm'):
        terms_dict = {c: i for i, c in enumerate(dataset.classes)}
        go_file = getattr(hparams, 'go_file', go_file)

        nf1, nf2, nf3, nf4, relations, zero_classes = load_normal_forms(go_file, terms_dict)
        self.n_rels = len(relations)
        self.normal_forms = nf1, nf2, nf3, nf4
        self.nb_zero_gos = len(zero_classes)

    def forward(self, features: Tensor, **kwargs) -> Tensor:
        if isinstance(features, SparseTensor):
            features = features.to_dense()
        self.all_gos = self.all_gos.to(self.device)
        self.hasFuncIndex = self.hasFuncIndex.to(self.device)

        x = self.net(features)
        go_embed = self.go_embed(self.all_gos)
        hasFunc = self.rel_embed(self.hasFuncIndex)
        hasFuncGO = go_embed + hasFunc
        go_rad = th.abs(self.go_rad(self.all_gos).view(1, -1))
        x = th.matmul(x, hasFuncGO.T) + go_rad
        logits = x
        return logits

    def training_step(self, batch, batch_nb):
        X, y_true, weights = batch
        batch_features = X['x_dict'][self.head_node_type]

        logits = self.forward(batch_features)

        logits, y_true, weights = concat_dict_batch(X['batch_size'], logits, y_true, weights)
        logits, y_true, weights = filter_samples_weights(y_pred=logits, y_true=y_true, weights=weights)

        loss = F.binary_cross_entropy_with_logits(logits, y_true)
        loss = loss + self.el_loss(self.normal_forms)
        self.log("loss", loss, logger=True, on_step=True)
        self.update_node_clf_metrics(self.train_metrics, logits, y_true, weights)
        return loss

    def validation_step(self, batch, batch_nb):
        X, y_true, weights = batch
        batch_features = X['x_dict'][self.head_node_type]

        logits = self.forward(batch_features)

        logits, y_true, weights = concat_dict_batch(X['batch_size'], logits, y_true, weights)
        logits, y_true, weights = filter_samples_weights(y_pred=logits, y_true=y_true, weights=weights)

        loss = F.binary_cross_entropy_with_logits(logits, y_true)
        self.log("val_loss", loss, on_step=True)
        self.update_node_clf_metrics(self.valid_metrics, logits, y_true, weights)
        return loss

    def test_step(self, batch, batch_nb):
        X, y_true, weights = batch
        batch_features = X['x_dict'][self.head_node_type]

        logits = self.forward(batch_features)

        logits, y_true, weights = concat_dict_batch(X['batch_size'], logits, y_true, weights)
        logits, y_true, weights = filter_samples_weights(y_pred=logits, y_true=y_true, weights=weights)

        loss = F.binary_cross_entropy_with_logits(logits, y_true)
        self.log("test_loss", loss, on_step=True)
        self.update_node_clf_metrics(self.test_metrics, logits, y_true, weights)
        return loss

    def predict_zero(self, features, data):
        x = self.net(features)
        go_embed = self.go_embed(data)
        hasFunc = self.rel_embed(self.hasFuncIndex)
        hasFuncGO = go_embed + hasFunc
        go_rad = th.abs(self.go_rad(data).view(1, -1))
        x = th.matmul(x, hasFuncGO.T) + go_rad
        logits = th.sigmoid(x)
        return logits

    def el_loss(self, go_normal_forms: Tuple[LongTensor]):
        nf1, nf2, nf3, nf4 = to_device(go_normal_forms, device=self.device)
        nf1_loss = self.nf1_loss(nf1)
        nf2_loss = self.nf2_loss(nf2)
        nf3_loss = self.nf3_loss(nf3)
        nf4_loss = self.nf4_loss(nf4)
        return nf1_loss + nf3_loss + nf4_loss + nf2_loss

    def class_dist(self, data):
        c = self.go_norm(self.go_embed(data[:, 0]))
        d = self.go_norm(self.go_embed(data[:, 1]))
        rc = th.abs(self.go_rad(data[:, 0]))
        rd = th.abs(self.go_rad(data[:, 1]))
        dist = th.linalg.norm(c - d, dim=1, keepdim=True) + rc - rd
        return dist

    def nf1_loss(self, data):
        pos_dist = self.class_dist(data)
        loss = th.mean(th.relu(pos_dist - self.margin))
        return loss

    def nf2_loss(self, data):
        c = self.go_norm(self.go_embed(data[:, 0]))
        d = self.go_norm(self.go_embed(data[:, 1]))
        e = self.go_norm(self.go_embed(data[:, 2]))
        rc = th.abs(self.go_rad(data[:, 0]))
        rd = th.abs(self.go_rad(data[:, 1]))
        re = th.abs(self.go_rad(data[:, 2]))

        sr = rc + rd
        dst = th.linalg.norm(c - d, dim=1, keepdim=True)
        dst2 = th.linalg.norm(e - c, dim=1, keepdim=True)
        dst3 = th.linalg.norm(e - d, dim=1, keepdim=True)
        loss = th.mean(th.relu(dst - sr - self.margin)
                       + th.relu(dst2 - rc - self.margin)
                       + th.relu(dst3 - rd - self.margin))

        return loss

    def nf3_loss(self, data):
        # R some C subClassOf D
        n = data.shape[0]
        # rS = self.rel_space(data[:, 0])
        # rS = rS.reshape(-1, self.embed_dim, self.embed_dim)
        rE = self.rel_embed(data[:, 0])
        c = self.go_norm(self.go_embed(data[:, 1]))
        d = self.go_norm(self.go_embed(data[:, 2]))
        # c = th.matmul(c, rS).reshape(n, -1)
        # d = th.matmul(d, rS).reshape(n, -1)
        rc = th.abs(self.go_rad(data[:, 1]))
        rd = th.abs(self.go_rad(data[:, 2]))

        rSomeC = c + rE
        euc = th.linalg.norm(rSomeC - d, dim=1, keepdim=True)
        loss = th.mean(th.relu(euc + rc - rd - self.margin))
        return loss

    def nf4_loss(self, data):
        # C subClassOf R some D
        n = data.shape[0]
        c = self.go_norm(self.go_embed(data[:, 0]))
        rE = self.rel_embed(data[:, 1])
        d = self.go_norm(self.go_embed(data[:, 2]))

        rc = th.abs(self.go_rad(data[:, 1]))
        rd = th.abs(self.go_rad(data[:, 2]))
        sr = rc + rd
        # c should intersect with d + r
        rSomeD = d + rE
        dst = th.linalg.norm(c - rSomeD, dim=1, keepdim=True)
        loss = th.mean(th.relu(dst - sr - self.margin))
        return loss

    def configure_optimizers(self):
        optimizer = th.optim.Adam(self.parameters(), lr=5e-4)
        scheduler = MultiStepLR(optimizer, milestones=[5, 20], gamma=0.1)

        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return x + self.fn(x)


class MLPBlock(nn.Module):

    def __init__(self, in_features, out_features, bias=True, layer_norm=True, dropout=0.1, activation=nn.ReLU):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        self.activation = activation()
        self.layer_norm = nn.BatchNorm1d(out_features) if layer_norm else None
        self.dropout = nn.Dropout(dropout) if dropout else None

    def forward(self, x):
        x = self.activation(self.linear(x))
        if self.layer_norm:
            x = self.layer_norm(x)
        if self.dropout:
            x = self.dropout(x)
        return x


def load_data(data_root, ont, terms_file):
    terms_df = pd.read_pickle(terms_file)
    terms = terms_df['gos'].values.flatten()
    terms_dict = {v: i for i, v in enumerate(terms)}
    print('Terms', len(terms))

    ipr_df = pd.read_pickle(f'{data_root}/{ont}/interpros.pkl')
    iprs = ipr_df['interpros'].values
    iprs_dict = {v: k for k, v in enumerate(iprs)}

    train_df = pd.read_pickle(f'{data_root}/{ont}/train_data.pkl')
    valid_df = pd.read_pickle(f'{data_root}/{ont}/valid_data.pkl')
    test_df = pd.read_pickle(f'{data_root}/{ont}/test_data.pkl')

    train_data = get_data(train_df, iprs_dict, terms_dict)
    valid_data = get_data(valid_df, iprs_dict, terms_dict)
    test_data = get_data(test_df, iprs_dict, terms_dict)

    return iprs_dict, terms_dict, train_data, valid_data, test_data, test_df


def get_data(df, iprs_dict, terms_dict):
    data = th.zeros((len(df), len(iprs_dict)), dtype=th.float32)
    labels = th.zeros((len(df), len(terms_dict)), dtype=th.float32)
    for i, row in enumerate(df.itertuples()):
        for ipr in row.interpros:
            if ipr in iprs_dict:
                data[i, iprs_dict[ipr]] = 1
        for go_id in row.prop_annotations:  # prop_annotations for full model
            if go_id in terms_dict:
                g_id = terms_dict[go_id]
                labels[i, g_id] = 1
    return data, labels


def load_normal_forms(go_file='data/go.norm', terms_dict: Dict[str, int] = None):
    nf1 = []
    nf2 = []
    nf3 = []
    nf4 = []
    relations = {}
    zclasses = {}

    def get_index(go_id):
        if go_id in terms_dict:
            index = terms_dict[go_id]
        elif go_id in zclasses:
            index = zclasses[go_id]
        else:
            zclasses[go_id] = len(terms_dict) + len(zclasses)
            index = zclasses[go_id]
        return index

    def get_rel_index(rel_id):
        if rel_id not in relations:
            relations[rel_id] = len(relations)
        return relations[rel_id]

    with open(go_file) as f:
        for line in f:
            line = line.strip().replace('_', ':')
            if line.find('SubClassOf') == -1:
                continue
            left, right = line.split(' SubClassOf ')
            # C SubClassOf D
            if len(left) == 10 and len(right) == 10:
                go1, go2 = left, right
                nf1.append((get_index(go1), get_index(go2)))
            elif left.find('and') != -1:  # C and D SubClassOf E
                go1, go2 = left.split(' and ')
                go3 = right
                nf2.append((get_index(go1), get_index(go2), get_index(go3)))
            elif left.find('some') != -1:  # R some C SubClassOf D
                rel, go1 = left.split(' some ')
                go2 = right
                nf3.append((get_rel_index(rel), get_index(go1), get_index(go2)))
            elif right.find('some') != -1:  # C SubClassOf R some D
                go1 = left
                rel, go2 = right.split(' some ')
                nf4.append((get_index(go1), get_rel_index(rel), get_index(go2)))

    nf1 = th.LongTensor(nf1)
    nf2 = th.LongTensor(nf2)
    nf3 = th.LongTensor(nf3)
    nf4 = th.LongTensor(nf4)
    return nf1, nf2, nf3, nf4, relations, zclasses


class Ontology(object):

    def __init__(self, filename='data/go.obo', with_rels=False):
        self.ont = self.load(filename, with_rels)
        self.ic = None
        self.ic_norm = 0.0

    def has_term(self, term_id):
        return term_id in self.ont

    def get_term(self, term_id):
        if self.has_term(term_id):
            return self.ont[term_id]
        return None

    def calculate_ic(self, annots):
        cnt = Counter()
        for x in annots:
            cnt.update(x)
        self.ic = {}
        for go_id, n in cnt.items():
            parents = self.get_parents(go_id)
            if len(parents) == 0:
                min_n = n
            else:
                min_n = min([cnt[x] for x in parents])

            self.ic[go_id] = math.log(min_n / n, 2)
            self.ic_norm = max(self.ic_norm, self.ic[go_id])

    def get_ic(self, go_id):
        if self.ic is None:
            raise Exception('Not yet calculated')
        if go_id not in self.ic:
            return 0.0
        return self.ic[go_id]

    def get_norm_ic(self, go_id):
        return self.get_ic(go_id) / self.ic_norm

    def load(self, filename, with_rels):
        ont = dict()
        obj = None
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line == '[Term]':
                    if obj is not None:
                        ont[obj['id']] = obj
                    obj = dict()
                    obj['is_a'] = list()
                    obj['part_of'] = list()
                    obj['regulates'] = list()
                    obj['alt_ids'] = list()
                    obj['is_obsolete'] = False
                    continue
                elif line == '[Typedef]':
                    if obj is not None:
                        ont[obj['id']] = obj
                    obj = None
                else:
                    if obj is None:
                        continue
                    l = line.split(": ")
                    if l[0] == 'id':
                        obj['id'] = l[1]
                    elif l[0] == 'alt_id':
                        obj['alt_ids'].append(l[1])
                    elif l[0] == 'namespace':
                        obj['namespace'] = l[1]
                    elif l[0] == 'is_a':
                        obj['is_a'].append(l[1].split(' ! ')[0])
                    elif with_rels and l[0] == 'relationship':
                        it = l[1].split()
                        # add all types of relationships
                        obj['is_a'].append(it[1])
                    elif l[0] == 'name':
                        obj['name'] = l[1]
                    elif l[0] == 'is_obsolete' and l[1] == 'true':
                        obj['is_obsolete'] = True
            if obj is not None:
                ont[obj['id']] = obj
        for term_id in list(ont.keys()):
            for t_id in ont[term_id]['alt_ids']:
                ont[t_id] = ont[term_id]
            if ont[term_id]['is_obsolete']:
                del ont[term_id]
        for term_id, val in ont.items():
            if 'children' not in val:
                val['children'] = set()
            for p_id in val['is_a']:
                if p_id in ont:
                    if 'children' not in ont[p_id]:
                        ont[p_id]['children'] = set()
                    ont[p_id]['children'].add(term_id)

        return ont

    def get_anchestors(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        q = deque()
        q.append(term_id)
        while (len(q) > 0):
            t_id = q.popleft()
            if t_id not in term_set:
                term_set.add(t_id)
                for parent_id in self.ont[t_id]['is_a']:
                    if parent_id in self.ont:
                        q.append(parent_id)
        return term_set

    def get_prop_terms(self, terms):
        prop_terms = set()

        for term_id in terms:
            prop_terms |= self.get_anchestors(term_id)
        return prop_terms

    def get_parents(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        for parent_id in self.ont[term_id]['is_a']:
            if parent_id in self.ont:
                term_set.add(parent_id)
        return term_set

    def get_namespace_terms(self, namespace):
        terms = set()
        for go_id, obj in self.ont.items():
            if obj['namespace'] == namespace:
                terms.add(go_id)
        return terms

    def get_namespace(self, term_id):
        return self.ont[term_id]['namespace']

    def get_term_set(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        q = deque()
        q.append(term_id)
        while len(q) > 0:
            t_id = q.popleft()
            if t_id not in term_set:
                term_set.add(t_id)
                for ch_id in self.ont[t_id]['children']:
                    q.append(ch_id)
        return term_set
