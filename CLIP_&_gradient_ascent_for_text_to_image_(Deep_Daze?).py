#!/usr/bin/env python
# coding: utf-8


## Description

# A Colab notebook for generating images using OpenAI's CLIP model.
#
# Heavily influenced by Alexander Mordvintsev's Deep Dream, this work uses CLIP to match an image learned by a SIREN network with a given textual description.
# As a good launching point for future directions and to find more related work, see https://distill.pub/2017/feature-visualization/
#
# If you have questions, please see my twitter at https://twitter.com/advadnoun
# This is all free! But if you're feeling generous, you can donate to my venmo @rynnn while your "a beautiful Waluigi" loads ;)

import torch.nn as nn
import clip
import subprocess
import glob
import torch
import numpy as np
import torchvision
import torchvision.transforms.functional as TF

import PIL
import matplotlib.pyplot as plt

import os
import random
import imageio

from IPython import get_ipython

# # CLIP

# Setup CLIP and set it to be the "perceptor" used to determine the loss for the SIREN network.
#
# Thanks to the authors below & OpenAI for sharing! https://github.com/openai/CLIP
#
# Alec Radford \* Jong Wook Kim \* Chris Hallacy Aditya Ramesh Gabriel Goh Sandhini Agarwal
# Girish Sastry Amanda Askell Pamela Mishkin Jack Clark Gretchen Krueger
# Ilya Sutskever
#

get_ipython().run_line_magic('cd', '/content/')
get_ipython().system('git clone https://github.com/openai/CLIP.git')
get_ipython().run_line_magic('cd', '/content/CLIP/')
get_ipython().system('pip install ftfy')


# Load the model
perceptor, preprocess = clip.load('ViT-B/32')


# # Params

# Determine the output dimensions of the image and the number of channels.
#
# Set the text to be matched

# In[ ]:


im_shape = [512, 512, 3]
sideX, sideY, channels = im_shape

tx = clip.tokenize("a beautiful Waluigi")


# # Define

# Define some helper functions

# In[ ]:


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


# # SIREN

# Thanks to the authors of SIREN! https://github.com/vsitzmann/siren
#
# @inproceedings{sitzmann2019siren,
#     author = {Sitzmann, Vincent
#               and Martel, Julien N.P.
#               and Bergman, Alexander W.
#               and Lindell, David B.
#               and Wetzstein, Gordon},
#     title = {Implicit Neural Representations
#               with Periodic Activation Functions},
#     booktitle = {arXiv},
#     year={2020}
# }
#
#
# The number of layers is 8 right now, but if the machine OOMs (runs out of RAM), it can naturally be tweaked. I've found that 16 layers for the SIREN works best, but I'm not always blessed with a V100 GPU.

# In[ ]:


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


siren_model = Siren(2, 256, 16, 3).cuda()
LLL = []
eps = 0

optimizer = torch.optim.Adam(siren_model.parameters(), .00001)

nom = torchvision.transforms.Normalize(
    (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

# # Train

# Train and output samples every 150 iterations
#
# We create batches of images at different resolutions in different parts of the SIREN image and resize them with bilinear upsampling. This seems to work very, very well as regularization for visualizing networks with larger images than their usual input resolution.

# In[ ]:


def checkin(loss):
    print(loss)
    with torch.no_grad():
        al = nom(siren_model(get_mgrid(sideX)).cpu()).numpy()
    for allls in al:
        displ(allls)
        display.display(display.Image(str(3)+'.png'))
        print('\n')
    output.eval_js(
        'new Audio("https://freesound.org/data/previews/80/80921_1022651-lq.ogg").play()')


def ascend_txt():
    out = siren_model(get_mgrid(sideX))

    cutn = 64
    p_s = []
    for ch in range(cutn):
        size = torch.randint(int(.5*sideX), int(.98*sideX), ())
        offsetx = torch.randint(0, sideX - size, ())
        offsety = torch.randint(0, sideX - size, ())
        apper = out[:, :, offsetx:offsetx + size, offsety:offsety + size]
        apper = torch.nn.functional.interpolate(
            apper, (224, 224), mode='bilinear')
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




itt = 0
for epochs in range(10000):
    for i in range(1000):
        train(eps, i)
        itt += 1
    eps += 1


# # Bot
