import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence, pad_packed_sequence
import numpy as np


class EncoderLSTM_Att(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, word_embedding_numpy=None):
        super(EncoderLSTM_Att, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size)

        if word_embedding_numpy is not None:
            self.embedding.weight.detach().copy_(torch.tensor(word_embedding_numpy.astype(np.float32)))

    def forward(self, input, length):
        embedded = self.embedding(input)
        output = embedded
        output = pack_padded_sequence(output, lengths=torch.tensor(length), batch_first=True, enforce_sorted=False)
        output, (hidden, cell) = self.lstm(output)
        output = pad_packed_sequence(output)[0]
        hidden = hidden.permute((1, 0, 2))  # [batch_size, layer * direction, hidden_size]
        hidden = hidden.reshape(hidden.shape[0], -1)  # [batch_size, hidden_size * layer * direction]
        return output, hidden, cell


class DecoderLSTM_Att(nn.Module):
    def __init__(self, embedding_size, hidden_size, output_size):
        super(DecoderLSTM_Att, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size)

        # Multiplicative attention을 위한 가중치 행렬 추가
        self.W_q = nn.Linear(hidden_size, hidden_size)  # 쿼리에 대한 가중치 행렬
        self.W_k = nn.Linear(hidden_size, hidden_size)  # 키에 대한 가중치 행렬

        self.softmax = nn.Softmax(dim=1)
        self.att_combine = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax_final = nn.LogSoftmax(dim=1)

    def forward(self, enc_output, input, hidden, cell):
        embed = self.embedding(input)
        output, (hidden, cell) = self.lstm(embed.unsqueeze(0), (hidden, cell))

        enc_output = enc_output.permute((1, 0, 2))  # encoder의 각 단어별 state
        # embed : [batch_size, embedding_size]
        # hidden : [1, batch_size, hidden_size]
        # enc_output : [batch_size, length, hidden_size]

        # Multiplicative Attention 적용
        query = self.W_q(hidden.permute((1, 0, 2)))  # 쿼리에 가중치 행렬 적용 [batch_size, 1, hidden_size]
        keys = self.W_k(enc_output)  # 키에 가중치 행렬 적용 [batch_size, length, hidden_size]

        att_score = torch.bmm(keys, query.permute((0, 2, 1)))  # 쿼리와 키의 내적을 통해 어텐션 점수 계산 [batch_size, length, 1]
        att_dist = self.softmax(att_score)  # 어텐션 분포 계산 [batch_size, length, 1]

        # 어텐션 값 계산
        att_output = enc_output * att_dist  # broadcasting
        att_output = torch.sum(att_output, dim=1).unsqueeze(0)  # [1, batch_size, hidden_size]

        # 어텐션 출력과 hidden 상태 결합
        hidden = torch.cat((hidden, att_output), dim=-1)  # [1, batch_size, hidden_size * 2]
        hidden = self.att_combine(hidden)  # [1, batch_size, hidden_size]

        # 출력 계산
        output = self.softmax_final(self.out(hidden[0]))
        return output, hidden, cell
