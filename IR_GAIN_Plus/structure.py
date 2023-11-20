import torch
import torch.nn as nn
from torch.nn.functional import relu

from IR_Net_Plus.structure import InputLayer


class Generator(nn.Module):
    def __init__(self, l_seq, n_feature, device, plus=True):
        super().__init__()
        self.plus = plus
        if plus:
            self.input_layer = InputLayer(l_seq, l_seq, device)
        else:
            self.input_layer = nn.Linear(l_seq, l_seq)
        self.layers = nn.Sequential(nn.Linear(l_seq, n_feature), nn.ReLU(),
                                    nn.Linear(n_feature, l_seq))

    def forward(self, x, m):
        # batch * l_seq * dim
        if self.plus:
            inputs = relu(self.input_layer(x.transpose(1, 2), m.transpose(1, 2)))  # batch * dim * l
        else:
            inputs = relu(self.input_layer(x.transpose(1, 2)))
        output = self.layers(inputs)
        return output.transpose(1, 2)


class Discriminator(nn.Module):
    def __init__(self, l_seq, n_feature):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(2 * l_seq, n_feature), nn.ReLU(),
                                    nn.Linear(n_feature, n_feature), nn.ReLU(),
                                    nn.Linear(n_feature, l_seq), nn.Sigmoid())

    def forward(self, x, m):
        x, m = x.transpose(1, 2), m.transpose(1, 2)
        h = torch.rand_like(m)
        mask = m.clone()
        mask[h > 0.3] = 0.5
        input = torch.cat([x, mask], dim=-1)
        output = self.layers(input)
        return output.transpose(1, 2)
