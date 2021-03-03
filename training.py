#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 02:23:00 2020

@author: walidajalil
"""
import os
import torch
import torch.nn as nn
import numpy as np
from VAE_test import VAE
import torchvision
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms
import sys

# ------ Initialize model
model = VAE(in_channels=3, out_channels=32, kernel_size=3, n_latent=128)
model = model.float()
model.cuda()

# ------ Load Data
dataset = torchvision.datasets.CelebA("/home/walid_abduljalil/Normflow", split='train',
                                      transform=transforms.Compose([transforms.RandomHorizontalFlip(),
                                                                    transforms.CenterCrop(148), transforms.Resize(64),
                                                                    transforms.ToTensor()]), download=False)
# /kaggle/input/celeba
#sys.exit("Check if download is complete")

data_load = DataLoader(dataset, batch_size=144, drop_last=True, shuffle=True)

# ------ Initialize optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.90)
print(" ")
print("Beginning training now:")
print(" ")
model.train()

if not os.path.isdir("/home/walid_abduljalil/Normflow/saved models"):
    os.makedirs("/home/walid_abduljalil/Normflow/saved models")

loss_list = []
d_kl_list = []
for epoch in range(21):
    for i, batch in enumerate(data_load):
        data_input = Variable(batch[0]).cuda()
        optimizer.zero_grad()
        loss_output, d_kl = model(data_input)
        loss_output.backward()
        loss_list.append(loss_output.item())
        d_kl_list.append(d_kl.item())
        optimizer.step()
        scheduler.step()
        if i % 100 == 0:
            print(torch.min(data_input,3))
            print(" ")
            print("Loss: ", loss_output.item())
            print(" ")
            print("D_KL: ", d_kl.item())
            print("i: ", i)
            print("-----------------------------")
            # if i == 300:
            # print("Reached", i, "iterations!")
            # break
    if epoch % 1 == 0:
        save_prefix = os.path.join("/home/walid_abduljalil/Normflow/saved models")
        path = "/home/walid_abduljalil/Normflow/saved models/model" + str(epoch) + ".pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss_output, 'D_KL': d_kl,
        }, path)
