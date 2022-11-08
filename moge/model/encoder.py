import logging
from argparse import Namespace
from collections import OrderedDict
from typing import Dict, Union, List

import torch
from esm.model.esm2 import ESM2
from torch import nn, Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch_sparse import SparseTensor
from transformers import BertConfig, BertForSequenceClassification

from moge.dataset.PyG.hetero_generator import HeteroNodeClfDataset
from moge.dataset.graph import HeteroGraphDataset
from moge.model.utils import tensor_sizes

logging.getLogger("transformers").setLevel(logging.ERROR)


class HeteroNodeFeatureEncoder(nn.Module):
    def __init__(self, hparams: Namespace, dataset: HeteroGraphDataset, subset_ntypes: List[str] = None) -> None:
        super().__init__()
        self.embeddings = self.init_embeddings(embedding_dim=hparams.embedding_dim,
                                               num_nodes_dict=dataset.num_nodes_dict,
                                               in_channels_dict=dataset.node_attr_shape,
                                               pretrain_embeddings=hparams.node_emb_init if "node_emb_init" in hparams else None,
                                               freeze=hparams.freeze_embeddings if "freeze_embeddings" in hparams else True,
                                               subset_ntypes=subset_ntypes, )
        print("model.encoder.embeddings: ", tensor_sizes(self.embeddings))

        # node types that needs a projection to align to the embedding_dim
        linear_dict = {}
        for ntype in dataset.node_types:
            if ntype in dataset.node_attr_shape and dataset.node_attr_shape[ntype] and \
                    dataset.node_attr_shape[ntype] != hparams.embedding_dim:

                # Row Sparse Matrix Multiplication
                if ntype in dataset.node_attr_sparse:
                    linear_dict[ntype] = nn.ParameterList([
                        nn.EmbeddingBag(dataset.node_attr_shape[ntype], hparams.embedding_dim,
                                        mode='sum', include_last_offset=True),
                        nn.Parameter(torch.zeros(hparams.embedding_dim)),
                        nn.ReLU(),
                        nn.Dropout(hparams.dropout if hasattr(hparams, 'dropout') else 0.0)
                    ])

                else:
                    linear_dict[ntype] = nn.Sequential(OrderedDict(
                        [('linear', nn.Linear(dataset.node_attr_shape[ntype], hparams.embedding_dim)),
                         ('relu', nn.ReLU()),
                         ('dropout', nn.Dropout(hparams.dropout if hasattr(hparams, 'dropout') else 0.0))]))

            elif ntype in self.embeddings and self.embeddings[ntype].weight.size(1) != hparams.embedding_dim:
                # Pretrained embeddings size doesn't match embedding_dim
                linear_dict[ntype] = nn.Sequential(OrderedDict(
                    [('linear', nn.Linear(self.embeddings[ntype].weight.size(1), hparams.embedding_dim)),
                     ('relu', nn.ReLU()),
                     ('dropout', nn.Dropout(hparams.dropout if hasattr(hparams, 'dropout') else 0.0))]))

        self.linear_proj = nn.ModuleDict(linear_dict)
        print("model.encoder.feature_projection: ", self.linear_proj)

        self.reset_parameters()

    def reset_parameters(self):
        for ntype, linear in self.linear_proj.items():
            if hasattr(linear, "weight"):
                nn.init.xavier_uniform_(linear.weight)

        for ntype, embedding in self.embeddings.items():
            if hasattr(embedding, "weight"):
                nn.init.xavier_uniform_(embedding.weight)

    def init_embeddings(self, embedding_dim: int,
                        num_nodes_dict: Dict[str, int],
                        in_channels_dict: Dict[str, int],
                        pretrain_embeddings: Dict[str, Tensor],
                        subset_ntypes: List[str] = None,
                        freeze=True) -> Dict[str, nn.Embedding]:
        # If some node type are not attributed, instantiate nn.Embedding for them
        if isinstance(in_channels_dict, dict):
            non_attr_node_types = (num_nodes_dict.keys() - in_channels_dict.keys())
        else:
            non_attr_node_types = []

        if subset_ntypes:
            non_attr_node_types = set(ntype for ntype in non_attr_node_types if ntype in subset_ntypes)

        embeddings = {}
        for ntype in non_attr_node_types:
            if pretrain_embeddings is None or ntype not in pretrain_embeddings:
                print("Initialized trainable embeddings: ", ntype)
                embeddings[ntype] = nn.Embedding(num_embeddings=num_nodes_dict[ntype],
                                                 embedding_dim=embedding_dim,
                                                 scale_grad_by_freq=True,
                                                 sparse=False)

                nn.init.xavier_uniform_(embeddings[ntype].weight)

            else:
                print(f"Pretrained embeddings freeze={freeze}", ntype)
                max_norm = pretrain_embeddings[ntype].norm(dim=1).mean()
                embeddings[ntype] = nn.Embedding.from_pretrained(pretrain_embeddings[ntype],
                                                                 freeze=freeze,
                                                                 scale_grad_by_freq=True,
                                                                 max_norm=max_norm)

        embeddings = nn.ModuleDict(embeddings)

        return embeddings

    def forward(self, feats: Dict[str, Tensor], global_node_index: Dict[str, Tensor]) -> Dict[str, Tensor]:
        h_dict = {k: v for k, v in feats.items()} if isinstance(feats, dict) else {}

        for ntype in global_node_index:
            if global_node_index[ntype].numel() == 0: continue

            if ntype not in h_dict and ntype in self.embeddings:
                h_dict[ntype] = self.embeddings[ntype](global_node_index[ntype]).to(global_node_index[ntype].device)

            # project to embedding_dim if node features are of not the same dimension
            if ntype in self.linear_proj and isinstance(h_dict[ntype], Tensor):
                if hasattr(self, "batchnorm"):
                    h_dict[ntype] = self.batchnorm[ntype].forward(h_dict[ntype])

                h_dict[ntype] = self.linear_proj[ntype].forward(h_dict[ntype])

            elif ntype in self.linear_proj and isinstance(h_dict[ntype], SparseTensor):
                self.linear_proj[ntype]: nn.ParameterList
                for module in self.linear_proj[ntype]:
                    if isinstance(module, nn.EmbeddingBag):
                        sparse_tensor: SparseTensor = h_dict[ntype]
                        h_dict[ntype] = module.forward(input=sparse_tensor.storage.col(),
                                                       offsets=sparse_tensor.storage.rowptr(),
                                                       per_sample_weights=sparse_tensor.storage.value())
                    elif isinstance(module, (nn.Parameter, Tensor)):
                        h_dict[ntype] = h_dict[ntype] + module
                    elif isinstance(module, nn.ReLU):
                        h_dict[ntype] = module.forward(h_dict[ntype])
                    elif isinstance(module, nn.Dropout):
                        h_dict[ntype] = module.forward(h_dict[ntype])

        return h_dict


class HeteroSequenceEncoder(nn.Module):
    def __init__(self, hparams: Namespace, dataset: HeteroNodeClfDataset) -> None:
        super().__init__()
        encoders = {}
        print("HeteroSequenceEncoder", list(dataset.seq_tokenizer.tokenizers.keys()))

        for ntype, tokenizer in dataset.seq_tokenizer.items():
            max_position_embeddings = dataset.seq_tokenizer.max_length[ntype]

            if hasattr(dataset.seq_tokenizer, 'esm_model'):
                encoders[ntype] = dataset.seq_tokenizer.esm_model

                # Freeze ESM pretrained layers
                for name, param in encoders[ntype].named_parameters():
                    if 'classifier' not in name:
                        param.requires_grad = False

            elif hasattr(hparams, "bert_config") and ntype in hparams.bert_config:
                if isinstance(hparams.bert_config[ntype], BertConfig):
                    bert_config = hparams.bert_config[ntype]
                    encoders[ntype] = BertForSequenceClassification(bert_config)

                    print("BertForSequenceClassification custom BertConfig", ntype)

                elif isinstance(hparams.bert_config[ntype], str):
                    encoders[ntype] = BertForSequenceClassification.from_pretrained(
                        hparams.bert_config[ntype],
                        num_labels=hparams.embedding_dim,
                        classifier_dropout=hparams.dropout, )

                    # Freeze BERT pretrained layers
                    for name, param in encoders[ntype].named_parameters():
                        if 'classifier' not in name:  # BERT emebedding/encoder/decoder layers
                            param.requires_grad = False

                    print("BertForSequenceClassification pretrained from:", hparams.bert_config[ntype])

            elif hasattr(hparams, "lstm_config"):
                lstm_config = hparams.lstm_config[ntype] if ntype in hparams.lstm_config else hparams.lstm_config
                encoders[ntype] = LSTMSequenceEncoder(vocab_size=tokenizer.vocab_size,
                                                      embedding_dim=hparams.embedding_dim,
                                                      hidden_dim=lstm_config["hidden_dim"],
                                                      kernel_size=lstm_config["kernel_size"],
                                                      num_layers=lstm_config["num_layers"],
                                                      dropout=lstm_config["dropout"], )

            else:
                bert_config = BertConfig(
                    vocab_size=tokenizer.vocab_size, hidden_size=128, max_position_embeddings=max_position_embeddings,
                    num_hidden_layers=2, num_attention_heads=4, intermediate_size=40, hidden_dropout_prob=0.1,
                    pad_token_id=tokenizer.vocab["[PAD]"], num_labels=hparams.embedding_dim,
                    position_embedding_type=None, use_cache=False, classifier_dropout=0.1)

                encoders[ntype] = BertForSequenceClassification(bert_config)
                print("BertForSequenceClassification default BertConfig", ntype)

        self.seq_encoders: Dict[str, Union[BertForSequenceClassification, LSTMSequenceEncoder, ESM2]] = \
            nn.ModuleDict(encoders)

    def forward(self, sequences: Dict[str, Dict[str, Tensor]], split_size: Union[float, int] = None) \
            -> Dict[str, Tensor]:
        """

        Args:
            sequences ():
            split_size ():

        Returns:

        """
        h_out = {}
        if isinstance(split_size, (int, float)):
            split_size = max(int(split_size), 1)
        else:
            split_size = None

        for ntype, inputs in sequences.items():
            model = self.seq_encoders[ntype]

            if isinstance(model, LSTMSequenceEncoder):
                # LTSM
                lengths = inputs["input_ids"].ne(0).sum(1)
                h_out[ntype] = model.forward(seqs=inputs["input_ids"], lengths=lengths)

            elif isinstance(model, ESM2):
                # ESM
                n_layers = len(model.layers)
                sequence_embs = []

                # Predict protein embeddings and generate per-sequence representation by averaging
                for batch_tokens, batch_lengths in zip(torch.split(inputs['batch_tokens'], split_size),
                                                       torch.split(inputs['batch_lengths'], split_size)):
                    results = model.forward(batch_tokens, repr_layers=[n_layers], return_contacts=False)
                    token_representations = results["representations"][n_layers]
                    for i, seq_len in enumerate(batch_lengths):
                        if seq_len >= token_representations.size(1) - 2:
                            sequence_embs.append(token_representations[i].mean(0))
                        else:
                            sequence_embs.append(token_representations[i, 1: seq_len + 1].mean(0))

                h_out[ntype] = torch.stack(sequence_embs, dim=0)

            elif isinstance(model, BertForSequenceClassification):
                # BERT
                if split_size:
                    batch_output = []
                    for input_ids, att_mask, token_type_ids in zip(torch.split(inputs["input_ids"], split_size),
                                                                   torch.split(inputs["attention_mask"], split_size),
                                                                   torch.split(inputs["token_type_ids"], split_size)):
                        out = model.forward(input_ids=input_ids, attention_mask=att_mask, token_type_ids=token_type_ids)
                        batch_output.append(out.logits)

                    h_out[ntype] = torch.cat(batch_output, dim=0)

                else:
                    out = model.forward(input_ids=inputs["input_ids"],
                                        attention_mask=inputs["attention_mask"],
                                        token_type_ids=inputs["token_type_ids"])
                    h_out[ntype] = out.logits

        return h_out

class LSTMSequenceEncoder(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int = 32, kernel_size: int = 13, num_layers=1,
                 dropout=0.3):
        super().__init__()
        self.vocab_size, self.embedding_dim, self.hidden_dim = vocab_size, embedding_dim, hidden_dim

        self.embedding = nn.Embedding(vocab_size, hidden_dim)

        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=self.kernel_size)
        self.maxpool = nn.MaxPool1d(kernel_size=self.kernel_size // 2)

        self.rnn = nn.LSTM(hidden_dim, embedding_dim, num_layers=num_layers,
                           batch_first=True, dropout=dropout, bidirectional=True)

        self.dropout = nn.Dropout(dropout)
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
