import math

import torch
import torch.nn as nn


# Incomplete Representation Mechanism
class IRM(nn.Module):
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


class Layer(nn.Module):
    def __init__(self, length, hidden_dim, dropout):
        super().__init__()
        self.qkv_layer = nn.ModuleList([nn.Linear(length, hidden_dim) for _ in range(3)])
        self.fc = nn.Sequential(nn.Linear(hidden_dim, length), nn.Dropout(dropout))
        self.attn_drop = nn.Dropout(dropout)
        self.attn_norm = nn.LayerNorm(length)

        self.ffn = nn.Sequential(nn.Linear(length, hidden_dim), nn.LeakyReLU(0.2), nn.Dropout(dropout),
                                 nn.Linear(hidden_dim, length), nn.Dropout(dropout))
        self.ffn_norm = nn.LayerNorm(length)

    def attention(self, x):
        """
        :param x: (batch, dim, length)
        :return: (batch, dim, length)
        """
        query, key, value = [self.qkv_layer[i](x) for i in range(3)]
        attn = torch.matmul(query, key.transpose(1, 2)) / math.sqrt(query.shape[-1])
        attn = self.attn_drop(torch.softmax(attn, dim=-1))
        value = torch.matmul(attn, value)
        return self.fc(value)

    def forward(self, x):
        x = self.attn_norm(self.attention(x) + x)
        return self.ffn_norm(self.ffn(x) + x)


class Net(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.use_irm = args.use_irm
        if args.use_irm:
            self.input_layer = IRM(args.length, args.length, args.device)
        else:
            self.input_layer = nn.Linear(args.length, args.length)
        self.act_func = nn.Sequential(nn.LeakyReLU(0.2), nn.Dropout(args.dropout))
        self.layers = nn.ModuleList([Layer(args.length, args.length, args.dropout) for _ in range(args.n_layer)])
        self.output_layer = nn.Linear(args.length, args.length)

    def forward(self, x, m):
        # batch * length * dim
        if self.use_irm:
            x = self.input_layer(x.transpose(1, 2), m.transpose(1, 2))
        else:
            x = self.input_layer(x.transpose(1, 2))
        x = self.act_func(x)
        for layer in self.layers:
            x = layer(x)
        return self.output_layer(x).transpose(1, 2)


class Discriminator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(2 * args.length, args.length), nn.ReLU(),
                                    nn.Linear(args.length, args.length), nn.ReLU(),
                                    nn.Linear(args.length, args.length), nn.Sigmoid())

    def forward(self, x, m):
        x, m = x.transpose(1, 2), m.transpose(1, 2)
        h = torch.rand_like(m)
        mask = m.clone()
        mask[h > 0.3] = 0.5
        x = torch.cat([x, mask], dim=-1)
        return self.layers(x).transpose(1, 2)
