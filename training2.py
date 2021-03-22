#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 02:23:00 2020

@author: walidajalil
"""

import os
import sys
import torch
from vanilla_VAE import VAE
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from data import *
import math

writer3 = SummaryWriter()
writer4 = SummaryWriter()

# ------ Initialize model
model = VAE(in_channels=1, out_channels=32, kernel_size=3, n_latent=256)
model.train()
model = model.float()
model.cuda()

# ------ Initialize optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1.0)
print(" ")
print("Beginning training now:")
print(" ")
model.train()
if not os.path.isdir("/home/walid_abduljalil/Normflow/saved models"):
    os.makedirs("/home/walid_abduljalil/Normflow/saved models")

epoch_loss_list = []
epoch_d_kl_list = []
epoch_val_loss_list = []
epoch_val_d_kl_list = []
for epoch in range(71):
    iteration_loss_list = []
    iteration_d_kl_list = []

    val_iteration_loss_list = []
    val_iteration_d_kl_list = []
    model.train()
    for i, batch in enumerate(train_loader):
        data_input = Variable(batch).cuda()
        optimizer.zero_grad()

        loss_output, d_kl = model(data_input)
        loss_output.backward()
        iteration_loss_list.append(loss_output.item())
        iteration_d_kl_list.append(d_kl.item())

        optimizer.step()
        scheduler.step()

    epoch_loss_list.append(np.mean(iteration_loss_list))
    epoch_d_kl_list.append(np.mean(iteration_d_kl_list))
    writer3.add_scalar("Loss/train", np.mean(iteration_loss_list), epoch)
    print("train loss: ", np.mean(iteration_loss_list))

    with torch.no_grad():
        model.eval()
        for i, val_batch in enumerate(validation_loader):

            val_data_input = Variable(val_batch).cuda()

            val_loss_output, val_d_kl = model(val_data_input)

            val_iteration_loss_list.append(val_loss_output.item())
            val_iteration_d_kl_list.append(val_d_kl.item())

        epoch_val_loss_list.append(np.mean(val_iteration_loss_list))
        epoch_val_d_kl_list.append(np.mean(val_iteration_d_kl_list))
        writer4.add_scalar("Loss/val", np.mean(val_iteration_loss_list), epoch)
        print("val loss: ", np.mean(val_iteration_loss_list))
        print("-------------------------------------------------")

    if epoch % 5 == 0:
        save_prefix = os.path.join("/home/walid_abduljalil/Normflow/saved_models")
        path = "/home/walid_abduljalil/Normflow/saved_models/model" + str(epoch) + ".pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss_list, 'D_KL': epoch_d_kl_list,
        }, path)