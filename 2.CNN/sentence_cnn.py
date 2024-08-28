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


