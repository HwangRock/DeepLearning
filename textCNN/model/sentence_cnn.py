import numpy as np
import torch
from torch import nn


class Convolution(nn.Module):
    def __init__(self, dimension, window, channel, dropout_rate):
        # 필터의 차원(열), 몇개의 수를 볼지 결정하는 window(행), 컨볼루션으로 볼 특징의 수, 특징의 dropout 확률을 인수로 받음.
        super().__init__()
        filters = []  # 필터들을 임시보관할 리스트
        for size, num in zip(window, channel):  # 차원은 동일하므로 윈도우와 채널에 따라서 필터들을 생성
            conv2d = nn.Conv2d(1, num, (size, dimension))  # 필터 생성
            nn.init.kaiming_normal_(conv2d.weight, mode='fan_out', nonlinearity='relu')  # 가중치 초기화(He)
            nn.init.zeros_(conv2d.bias)  # bias는 0으로 초기화
            filters.append(nn.Sequential(conv2d, nn.ReLU(inplace=True)))
        self.filters = nn.ModuleList(filters)  # 임시보관한 리스트를 modulelist로 변환
        self.window = window
        self.dropout_rate = nn.Dropout(dropout_rate)

    def forward(self, embedded_words):  # embedded_words: [batch, sentence length, embedding dimension]
        feature = []  # 컨볼루션의 결과를 담는 리스트에 넣음.
        for size, filters in zip(self.window, self.filters):
            conv_output = filters(embedded_words)  # convolution 진행
            conv_output = conv_output.squeeze(-1).max(dim=-1)[0]  # overfitting을 방지하기 위한 max pooling
            feature.append(conv_output)
            del conv_output  # 볼일없으면 메모리를 위해 제거.

        feature = torch.cat(feature, dim=1)
        feature = self.dropout(feature)

        return feature


class Sentencecnn(nn.Module):
    def __init__(self, classes, word_embedding_numpy, window, channel, dropout_rate):
        super().__init__()
        word_num = word_embedding_numpy.shape[0]  # 단어의 수를 받아옴.
        dimension = word_embedding_numpy.shape[1]  # 임베딩 벡터의 차원을 받아옴.

        self.word_embedding = nn.Embedding(word_num, dimension, padding_idx=0)  # 임베딩 행렬을 정의.

        # word2vec 활용
        self.word_embedding.weight.detach().copy_(torch.tensor(word_embedding_numpy.astype(np.float32)))

        self.convolution = Convolution(dimension, word_num, channel, dropout_rate) # convolution 정의

        filter_num = sum(channel)  # fliter의 수 (특징의 수와 연결됨.)
        self.linear = nn.Linear(filter_num, classes)  # fully connected layer를 통해서 차원 변환
        nn.init.kaiming_normal_(self.linear.weight, mode='fan_out', nonlinearity='relu')  # 가중치 초기화(He)
        nn.init.zeros_(self.linear.bias)  # bias는 0으로 초기화

    def forward(self, x):
        x = self.word_embedding(x)  # embedding layer
        x = self.convolution(x)  # convolution layer
        logits = self.linear(x)  # fully connected layer
        return logits  # crossentropy에 넣기위한 logits를 반환
