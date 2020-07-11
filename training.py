#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 02:23:00 2020

@author: walidajalil
"""

import torch
import torch.nn as nn
import numpy as np
from VAE_test import VAE
import torchvision
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms

# ------ Initialize model
model = VAE(in_channels=3, out_channels=32, kernel_size=3, n_latent=100)
model = model.float()

# ------ Load Data
dataset = torchvision.datasets.CelebA("../input/celeba/celeba", split='train', transform=transforms.Compose([
    transforms.CenterCrop(148), transforms.Resize(64), transforms.ToTensor()]), download=False)


data_load = DataLoader(dataset, batch_size=144, drop_last=True, shuffle=True)

# ------ Initialize optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
print(" ")
print("Beginning training now:")
print(" ")
model.train()

loss_list = []
for epoch in range(1):
    for i, batch in enumerate(data_load):
        data_input = Variable(batch[0])
        optimizer.zero_grad()
        loss_output = model(data_input)
        loss_output.backward()
        loss_list.append(loss_output.item())
        optimizer.step()
        print(" ")
        print("Loss: ", loss_output.item())
        # if i == 300:
        # print("Reached", i, "iterations!")
        # break
