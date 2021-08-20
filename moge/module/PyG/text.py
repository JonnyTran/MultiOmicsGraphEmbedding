import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from moge.module.utils import tensor_sizes


class SequenceEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim=32):
        super().__init__()
        self.vocab_size, self.embedding_dim, self.hidden_dim = vocab_size, embed_dim, hidden_dim

        self.embedding = nn.Embedding(vocab_size, hidden_dim)

        self.kernel_size = 13
        self.conv = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=self.kernel_size)
        self.maxpool = nn.MaxPool1d(kernel_size=self.kernel_size // 2)

        self.rnn = nn.LSTM(hidden_dim, embed_dim, num_layers=1, batch_first=True, dropout=0.3, bidirectional=True)

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(embed_dim * 2, embed_dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.embedding.weight)
        nn.init.xavier_normal_(self.conv.weight)
        nn.init.xavier_normal_(self.fc.weight)

    def forward(self, seq, lengths):
        embs = self.embedding(seq)
        embs = self.conv(embs.transpose(2, 1))
        embs = self.maxpool(embs).transpose(2, 1)

        lengths = ((lengths - self.kernel_size) / (self.kernel_size // 2)).type_as(lengths)
        lengths = torch.maximum(lengths, torch.tensor(1))

        packed_input = pack_padded_sequence(embs, lengths.cpu(), batch_first=True, enforce_sorted=False)  # unpad
        packed_output, _ = self.rnn(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        out_forward = output[range(len(output)), lengths - 1, :self.embedding_dim]
        out_reverse = output[:, 0, self.embedding_dim:]
        out_reduced = torch.cat((out_forward, out_reverse), 1)

        text_fea = self.dropout(out_reduced)
        text_fea = self.fc(text_fea)

        return text_fea
