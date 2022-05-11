from argparse import Namespace
from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertConfig, BertForSequenceClassification

from moge.dataset import HeteroNodeClfDataset
from moge.dataset.graph import HeteroGraphDataset
from moge.model.utils import tensor_sizes


class HeteroNodeEncoder(nn.Module):
    def __init__(self, hparams: Namespace, dataset: HeteroGraphDataset) -> None:
        super().__init__()
        self.embeddings = self.initialize_embeddings(embedding_dim=hparams.embedding_dim,
                                                     num_nodes_dict=dataset.num_nodes_dict,
                                                     in_channels_dict=dataset.node_attr_shape,
                                                     pretrain_embeddings=hparams.node_emb_init if "node_emb_init" in hparams else None,
                                                     freeze=hparams.freeze_embeddings if "freeze_embeddings" in hparams else True)
        print("model.encoder.embeddings: ", tensor_sizes(self.embeddings))

        # node types that needs a projection to align to the embedding_dim
        proj_node_types = [ntype for ntype in dataset.node_types \
                           if (ntype in dataset.node_attr_shape and
                               dataset.node_attr_shape[ntype] != hparams.embedding_dim) \
                           or (self.embeddings and ntype in self.embeddings and
                               self.embeddings[ntype].weight.size(1) != hparams.embedding_dim)]

        self.feature_projection = nn.ModuleDict({
            ntype: nn.Linear(in_features=dataset.node_attr_shape[ntype],
                             out_features=hparams.embedding_dim) \
            for ntype in proj_node_types})
        print("model.encoder.feature_projection: ", self.feature_projection)

        if hparams.batchnorm:
            self.batchnorm: Dict[str, nn.BatchNorm1d] = nn.ModuleDict({
                ntype: nn.BatchNorm1d(input_size) \
                for ntype, input_size in dataset.node_attr_shape.items() \
                })

        if hasattr(hparams, "dropout") and hparams.dropout:
            self.dropout = hparams.dropout
        else:
            self.dropout = None

    def initialize_embeddings(self, embedding_dim, num_nodes_dict, in_channels_dict,
                              pretrain_embeddings: Dict[str, Tensor],
                              freeze=True):
        # If some node type are not attributed, instantiate nn.Embedding for them
        if isinstance(in_channels_dict, dict):
            non_attr_node_types = (num_nodes_dict.keys() - in_channels_dict.keys())
        else:
            non_attr_node_types = []

        if non_attr_node_types:
            module_dict = {}

            for ntype in non_attr_node_types:
                if pretrain_embeddings is None or ntype not in pretrain_embeddings:
                    print("Initialized trainable embeddings: ", ntype)
                    module_dict[ntype] = nn.Embedding(num_embeddings=num_nodes_dict[ntype],
                                                      embedding_dim=embedding_dim,
                                                      scale_grad_by_freq=False,
                                                      sparse=False)
                else:
                    print(f"Pretrained embeddings freeze={freeze}", ntype)
                    max_norm = pretrain_embeddings[ntype].norm(dim=1).mean()
                    module_dict[ntype] = nn.Embedding.from_pretrained(pretrain_embeddings[ntype],
                                                                      freeze=freeze,
                                                                      scale_grad_by_freq=False,
                                                                      max_norm=max_norm)

            embeddings = nn.ModuleDict(module_dict)
        else:
            embeddings = None

        return embeddings

    def forward(self, node_feats: Dict[str, Tensor], global_node_idx: Dict[str, Tensor]) -> Dict[str, Tensor]:
        h_dict = {k: v for k, v in node_feats.items()}

        for ntype in global_node_idx:
            if global_node_idx[ntype].numel() == 0: continue

            if ntype not in h_dict:
                h_dict[ntype] = self.embeddings[ntype](global_node_idx[ntype]).to(global_node_idx[ntype].device)

            # project to embedding_dim if node features are not same same dimension
            if ntype in self.feature_projection:
                if hasattr(self, "batchnorm"):
                    h_dict[ntype] = self.batchnorm[ntype].forward(h_dict[ntype])

                h_dict[ntype] = self.feature_projection[ntype](h_dict[ntype])
                h_dict[ntype] = F.relu(h_dict[ntype])
                if hasattr(self, "dropout") and self.dropout:
                    h_dict[ntype] = F.dropout(h_dict[ntype], p=self.dropout, training=self.training)

        return h_dict


class HeteroSequenceEncoder(nn.Module):
    def __init__(self, hparams: Namespace, dataset: HeteroNodeClfDataset) -> None:
        super().__init__()
        seq_encoders = {}

        for ntype, tokenizer in dataset.seq_tokenizer.items():
            max_position_embeddings = dataset.seq_tokenizer.max_length[ntype]
            if max_position_embeddings is None:
                max_position_embeddings = 512

            if hasattr(hparams, "bert_config") and ntype in hparams.bert_config:
                bert_config = hparams.bert_config[ntype]
            else:
                bert_config = BertConfig(vocab_size=tokenizer.vocab_size, hidden_size=128,
                                         max_position_embeddings=max_position_embeddings,
                                         num_hidden_layers=2, num_attention_heads=8, intermediate_size=128,
                                         hidden_dropout_prob=0.1,
                                         pad_token_id=tokenizer.vocab["[PAD]"],
                                         num_labels=hparams.embedding_dim,
                                         position_embedding_type=None,  # "relative_key",
                                         use_cache=False,
                                         classifier_dropout=0.1)

            seq_encoders[ntype] = BertForSequenceClassification(bert_config)
            print("BertForSequenceClassification", ntype)

        self.seq_encoders: Dict[str, BertForSequenceClassification] = nn.ModuleDict(seq_encoders)

    def forward(self, sequences: Dict[str, Dict[str, Tensor]], batch_size=None) -> Dict[str, Tensor]:
        h_out = {}
        for ntype, encoding in sequences.items():
            batch_output = []

            if batch_size:
                for input_ids, attention_mask, token_type_ids in zip(torch.split(encoding["input_ids"], batch_size),
                                                                     torch.split(encoding["attention_mask"],
                                                                                 batch_size),
                                                                     torch.split(encoding["token_type_ids"],
                                                                                 batch_size)):
                    out = self.seq_encoders[ntype].forward(input_ids=input_ids,
                                                           attention_mask=attention_mask,
                                                           token_type_ids=token_type_ids)
                    batch_output.append(out.logits)

                h_out[ntype] = torch.cat(batch_output, dim=0)

            else:
                out = self.seq_encoders[ntype].forward(input_ids=encoding["input_ids"],
                                                       attention_mask=encoding["attention_mask"],
                                                       token_type_ids=encoding["token_type_ids"])
                h_out[ntype] = out.logits

        return h_out


class LSTMSequenceEncoder(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int = 32, kernel_size: int = 13):
        super().__init__()
        self.vocab_size, self.embedding_dim, self.hidden_dim = vocab_size, embedding_dim, hidden_dim

        self.embedding = nn.Embedding(vocab_size, hidden_dim)

        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=self.kernel_size)
        self.maxpool = nn.MaxPool1d(kernel_size=self.kernel_size // 2)

        self.rnn = nn.LSTM(hidden_dim, embedding_dim, num_layers=1, batch_first=True, dropout=0.3, bidirectional=True)

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(embedding_dim * 2, embedding_dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.embedding.weight)
        nn.init.xavier_normal_(self.conv.weight)
        nn.init.xavier_normal_(self.fc.weight)

    def forward(self, seqs: Tensor, lengths: Tensor):
        embs = self.embedding.forward(seqs)
        embs = self.conv.forward(embs.transpose(2, 1))
        embs = self.maxpool.forward(embs).transpose(2, 1)

        lengths = ((lengths - self.kernel_size) / (self.kernel_size // 2)).type_as(lengths)
        lengths = torch.maximum(lengths, torch.tensor(1))

        packed_input = pack_padded_sequence(embs, lengths.cpu(), batch_first=True, enforce_sorted=False)  # unpad
        packed_output, _ = self.rnn.forward(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        out_forward = output[range(len(output)), lengths - 1, :self.embedding_dim]
        out_reverse = output[:, 0, self.embedding_dim:]
        out_reduced = torch.cat((out_forward, out_reverse), 1)

        text_fea = self.dropout(out_reduced)
        text_fea = self.fc(text_fea)

        return text_fea
