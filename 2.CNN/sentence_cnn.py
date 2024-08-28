import numpy as np
import torch
from torch import nn


class Convolution(nn.Module):
    def __init__(self, dimension, window, channel, dropout_rate):
        super().__init__()
        filters = []
        for size, num in zip(window, channel):
            conv2d = nn.Conv2d(1, num, (size, dimension))
            nn.init.kaiming_normal_(conv2d.weight, mode='fan_out', nonlinearity='relu')  # 가중치 초기화(He)
            nn.init.zeros_(conv2d.bias)  # bias는 0으로 초기화
            filters.append(nn.Sequentioal(conv2d, nn.ReLU(inplace=True)))
        self.filters = nn.ModuleList(filters)
        self.window = window
        self.dropout_rate = nn.Dropout(dropout_rate)

    def forward(self, embedded_words):  # embedded_words: [batch, sentence length, embedding dimension]
        feature = []
        for size, filters in zip(self.window, self.filters):
            conv_output = filters(embedded_words)
            conv_output = conv_output.squeeze(-1).max(dim=-1)[0]
            feature.append(conv_output)
            del conv_output

        feature = torch.cat(feature, dim=1)
        feature = self.dropout(feature)

        return feature

