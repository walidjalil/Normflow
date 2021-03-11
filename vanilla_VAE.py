"""
VAE Class

@author: walidajalil
"""

import torch
import numpy as np
import torch.nn as nn
from loss import *


class VAE(nn.Module):

    def encoder(self, in_channels, out_channels, kernel_size):
        encode = nn.Sequential(nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels,
                                         padding=1, stride=2),
                               nn.BatchNorm2d(out_channels),
                               nn.LeakyReLU())
        return encode

    def decoder(self, in_channels, out_channels, kernel_size):
        decode = nn.Sequential(nn.ConvTranspose2d(kernel_size=kernel_size, in_channels=in_channels,
                                                  out_channels=out_channels, padding=1, stride=2, output_padding=1),
                               nn.BatchNorm2d(out_channels), nn.LeakyReLU())
        return decode

    def create_last_layer(self, in_channels, out_channels, kernel_size):
        last_layer = nn.Sequential(nn.ConvTranspose2d(kernel_size=kernel_size, in_channels=out_channels,
                                                      out_channels=out_channels, padding=1, stride=2, output_padding=1),
                                   nn.BatchNorm2d(out_channels), nn.LeakyReLU(),
                                   nn.Conv2d(kernel_size=kernel_size, in_channels=out_channels, out_channels=in_channels
                                             , padding=1))  # , nn.Tanh())
        return last_layer

    def reparam_trick(self, mu, log_var):
        norm = torch.randn(mu.size()).cuda()
        standard_dev = torch.exp(0.5 * log_var)

        return norm * standard_dev + mu

    def __init__(self, in_channels=1, out_channels=32, kernel_size=3, n_latent=None):
        super(VAE, self).__init__()
        super(VAE, self).__init__()

        self.e1 = self.encoder(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.e2 = self.encoder(in_channels=out_channels, out_channels=out_channels * 2, kernel_size=kernel_size)
        self.e3 = self.encoder(in_channels=out_channels * 2, out_channels=out_channels * 4, kernel_size=kernel_size)
        self.e4 = self.encoder(in_channels=out_channels * 4, out_channels=out_channels * 8, kernel_size=kernel_size)
        self.e5 = self.encoder(in_channels=out_channels * 8, out_channels=out_channels * 16, kernel_size=kernel_size)

        self.mu = nn.Linear(out_channels * 400, n_latent)
        self.var = nn.Linear(out_channels * 400, n_latent)

        self.decoder_input = nn.Linear(n_latent, out_channels * 400)

        self.d1 = self.decoder(in_channels=out_channels * 16, out_channels=out_channels * 8, kernel_size=kernel_size)
        self.d2 = self.decoder(in_channels=out_channels * 8, out_channels=out_channels * 4, kernel_size=kernel_size)
        self.d3 = self.decoder(in_channels=out_channels * 4, out_channels=out_channels * 2, kernel_size=kernel_size)
        self.d4 = self.decoder(in_channels=out_channels * 2, out_channels=out_channels, kernel_size=kernel_size)

        self.last_layer = self.create_last_layer(kernel_size=kernel_size, in_channels=in_channels,
                                                 out_channels=out_channels)

    def forward(self, x):
        # print("input shape:", x.shape)
        encoding1 = self.e1(x)
        # print("encoding1 shape:", x.shape)
        encoding2 = self.e2(encoding1)
        # print("encoding2 shape:", x.shape)
        encoding3 = self.e3(encoding2)
        # print("encoding3 shape:", x.shape)
        encoding4 = self.e4(encoding3)
        # print("encoding4 shape:", x.shape)
        # encoding5 = self.e5(encoding4)
        # print("encoding5 shape:", x.shape)
        flat = torch.flatten(encoding4, start_dim=1)
        # print("flatten shape:", x.shape)
        mu = self.mu(flat)
        log_var = self.var(flat)

        z = self.reparam_trick(mu, log_var)

        d_input = self.decoder_input(z)
        d_input = d_input.view(-1, 256, 5, 10)
        # decoding1 = self.d1(d_input)
        decoding2 = self.d2(d_input)
        decoding3 = self.d3(decoding2)
        decoding4 = self.d4(decoding3)

        output = self.last_layer(decoding4)
        # print("Output shape: ", output.shape)
        # print("gt_images=x shape: ", x.shape)
        # print("output shape: ", output.shape)
        loss, d_kl = get_loss(mu, log_var, gt_images=x, reconstructions=output.permute(0, 1, 3, 2))
        # print("Loss: ", loss)

        return loss, d_kl