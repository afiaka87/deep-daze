
import torch.nn as nn
import clip
import glob
import os
import random

import imageio
import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import torchvision
import torchvision.transforms.functional as TF
from IPython import display
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"


# Load the model
perceptor, preprocess = clip.load('ViT-B/32')

im_shape = [512, 512, 3]
sideX, sideY, channels = im_shape

tx = clip.tokenize(TEXT)


def displ(img, pre_scaled=True):
    img = np.array(img)[:, :, :]
    img = np.transpose(img, (1, 2, 0))
    if not pre_scaled:
        img = scale(img, 48*4, 32*4)
    imageio.imwrite(str(3) + '.png', np.array(img))
    return display.Image(str(3)+'.png')


def card_padded(im, to_pad=3):
    return np.pad(np.pad(np.pad(im, [[1, 1], [1, 1], [0, 0]], constant_values=0), [[2, 2], [2, 2], [0, 0]], constant_values=1),
                  [[to_pad, to_pad], [to_pad, to_pad], [0, 0]], constant_values=0)


class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=True,
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features,
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                             np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True)
        output = self.net(coords.cuda())
        # .sigmoid_()
        return output.view(1, sideX, sideY, 3).permute(0, 3, 1, 2)

    def forward_with_activations(self, coords, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!'''
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)

                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()

                activations['_'.join(
                    (str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else:
                x = layer(x)

                if retain_grad:
                    x.retain_grad()

            activations['_'.join(
                (str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations


def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid


model = Siren(2, 256, 16, 3).cuda()
LLL = []
eps = 0

optimizer = torch.optim.Adam(model.parameters(), .00001)


def checkin(loss):
    print(loss)
    with torch.no_grad():
        al = nom(model(get_mgrid(sideX)).cpu()).numpy()
    for allls in al:
        displ(allls)
        display.display(display.Image(str(3)+'.png'))
        print('\n')
    output.eval_js(
        'new Audio("https://freesound.org/data/previews/80/80921_1022651-lq.ogg").play()')


def ascend_txt():
    out = model(get_mgrid(sideX))

    cutn = 64
    p_s = []
    for ch in range(cutn):
        size = torch.randint(int(.5*sideX), int(.98*sideX), ())
        offsetx = torch.randint(0, sideX - size, ())
        offsety = torch.randint(0, sideX - size, ())
        apper = out[:, :, offsetx:offsetx + size, offsety:offsety + size]
        apper = torch.nn.functional.interpolate(
            apper, (224, 224), mode='bilinear', align_corners=True)
        p_s.append(nom(apper))
    into = torch.cat(p_s, 0)

    iii = perceptor.encode_image(into)
    t = perceptor.encode_text(tx.cuda())
    return -100*torch.cosine_similarity(t, iii, dim=-1).mean()


def train(epoch, i):
    loss = ascend_txt()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if itt % 150 == 0:
        checkin(loss)


nom = torchvision.transforms.Normalize(
    (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))


itt = 0
for epochs in range(10000):
    for i in range(1000):
        train(eps, i)
        itt += 1
    eps += 1
