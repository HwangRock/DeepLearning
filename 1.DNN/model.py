import torch.nn as nn
import torch.nn.functional as f


class MLP(nn.Module):
    def __init__(self, drop_prob=0.3):
        super().__init__()
        self.layer1 = nn.Linear(28*28, 1024)
        self.layer2 = nn.Linear(1024, 512)
        self.layer3 = nn.Linear(512, 256)
        self.layer4 = nn.Linear(256, 10)

        self.active = f.relu
        self.dropout_prob = drop_prob
        self._init_xavier()

    def _init_xavier(self):
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)
        nn.init.xavier_uniform_(self.layer3.weight)
        nn.init.xavier_uniform_(self.layer4.weight)

    def forward(self, x):
        x = self.active(self.layer1(x))
        x = f.dropout(x, training=self.training, p=self.dropout_prob)
        x = self.active(self.layer2(x))
        x = f.dropout(x, training=self.training, p=self.dropout_prob)
        x = self.active(self.layer3(x))
        x = f.dropout(x, training=self.training, p=self.dropout_prob)
        x = self.layer4(x)

        return x
