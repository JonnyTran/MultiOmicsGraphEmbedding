import dgl
import dgl.function as fn
import numpy as np
import torch
import torch.nn.functional as F
from logzero import logger
from torch import nn
from tqdm import tqdm


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


class Model(object):
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
        scores = self.network(nf, (torch.from_numpy(batch_x.indices).cuda().long(),
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
                train_loss += self.train_step(nf, torch.from_numpy(batch_y), True)
            best_fmax = self.valid(valid_ppi, valid_y, epoch_idx, train_loss / len(train_ppi), best_fmax)

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
