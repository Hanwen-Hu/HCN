import torch
import torch.nn as nn
from torch.nn.functional import leaky_relu

class InputLayer(nn.Module):
    def __init__(self, in_dim, out_dim, device):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, 1, in_dim, out_dim, device=device))
        self.bias = nn.Parameter(torch.zeros(1, 1, out_dim, device=device))
        self.scale = nn.Parameter(torch.ones(1, 1, out_dim, device=device))

    def forward(self, x, m):
        batch, dim = x.shape[0], x.shape[1]
        weight = self.weight.repeat(batch, dim, 1, 1)
        weight[m == 0] = 0
        weight = weight / (torch.sum(torch.abs(weight), dim=-2, keepdim=True) + 1e-5)
        return self.scale * torch.sum(weight * x.unsqueeze(-1), dim=2) + self.bias


class Net_Plus(nn.Module):
    def __init__(self, dim, length, device, plus=True):
        super().__init__()
        self.plus = plus
        if plus:
            self.input_layer = InputLayer(length, length, device)
        else:
            self.input_layer = nn.Linear(length, length)
        self.spatial_layers = nn.Sequential(nn.Linear(dim, dim), nn.LeakyReLU())
        self.temporal_layers = nn.Sequential(nn.Linear(length, length), nn.LeakyReLU())
        self.output_layer = nn.Linear(length, length)

    def forward(self, x, m):
        # batch * length * dim
        if self.plus:
            x = self.input_layer(x.transpose(1, 2), m.transpose(1, 2))
        else:
            x = self.input_layer(x.transpose(1, 2))
        x = leaky_relu(x).transpose(1, 2)  # batch * length * dim
        x = (self.spatial_layers(x) + x).transpose(1, 2)  # batch * dim * length
        x = self.temporal_layers(x) + x
        return self.output_layer(x).transpose(1, 2)
