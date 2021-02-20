import numpy as np
import torch
from torch import nn as nn


class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                lim = 1 / self.in_features
            else:
                lim = np.sqrt(6 / self.in_features) / self.omega_0
            self.linear.weight.uniform_(-lim, lim)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))


class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, side_x, side_y, outermost_linear=True,
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        self.side_x = side_x
        self.side_y = side_y

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            with torch.no_grad():
                lim = np.sqrt(6 / hidden_features) / hidden_omega_0
                final_linear.weight.uniform_(-lim, lim)
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True)
        output = self.net(coords.cuda())
        return output.view(1, self.side_y, self.side_x, 3).permute(0, 3, 1, 2)  # .sigmoid_()