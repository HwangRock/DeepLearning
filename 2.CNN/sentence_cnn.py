import numpy as np
from torch import nn

class Convolution(nn.Module):
    def __init__(self, dimension, window, channel, dropout_rate):
        super().__init__()
        filters = []
        for size,