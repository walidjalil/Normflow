#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 02:23:00 2020

@author: walidajalil
"""
import os
import sys
import torch
from VAE_test import VAE
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from data import *

writer = SummaryWriter()

# ------ Initialize model
model = VAE(in_channels=1, out_channels=32, kernel_size=3, n_latent=128)
model = model.float()
model.cuda()

# ------ Initialize optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1.0)
print(" ")
print("Beginning training now:")
print(" ")

if not os.path.isdir("/home/walid_abduljalil/Normflow/saved models"):
    os.makedirs("/home/walid_abduljalil/Normflow/saved models")

epoch_loss_list = []
epoch_d_kl_list = []
epoch_val_loss_list = []
epoch_val_d_kl_list = []

for epoch in range(100):
    iteration_loss_list = []
    iteration_d_kl_list = []

    val_iteration_loss_list = []
    val_iteration_d_kl_list = []

    for i, batch in enumerate(train_loader):

        model.train()
        data_input = Variable(batch).cuda()

        optimizer.zero_grad()

        loss_output, d_kl = model(data_input)
        loss_output.backward()

        iteration_loss_list.append(loss_output.item())
        iteration_d_kl_list.append(d_kl.item())

        optimizer.step()
        scheduler.step()

    epoch_loss_list.append(mean(iteration_loss_list))
    epoch_d_kl_list.append(mean(iteration_d_kl_list))
    print("train loss: ", mean(iteration_loss_list))

    with torch.no_grad():
        for j, val_batch in enumerate(validation_loader):

            model.eval()
            data_input = Variable(val_batch).cuda()

            val_loss_output, val_d_kl = model(data_input)
            val_iteration_loss_list.append(val_loss_output.item())
            val_iteration_d_kl_list.append(val_d_kl.item())

        epoch_val_loss_list.append(val_iteration_loss_list.mean())
        epoch_val_d_kl_list.append(val_iteration_d_kl_list.mean())

    if epoch % 100 == 0:
        save_prefix = os.path.join("/home/walid_abduljalil/Normflow/saved models")
        path = "/home/walid_abduljalil/Normflow/saved models/model" + str(epoch) + ".pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss_list[epoch], 'D_KL': epoch_d_kl_list[epoch],
        }, path)
