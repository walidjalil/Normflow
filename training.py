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

writer = SummaryWriter()
writer2 = SummaryWriter()
device = torch.device("cuda")
# ------ Initialize model
model = VAE(in_channels=1, out_channels=32, kernel_size=3, n_latent=128)
checkpoint = torch.load('/home/walid_abduljalil/Normflow/model325.pt',map_location="cuda:0")
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1.0)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#epoch_start = checkpoint['epoch']
#loss = checkpoint['loss']
model.train()
model = model.float()
model.cuda()

# ------ Initialize optimizer
#optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
#scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1.0)
print(" ")
print("Beginning training now:")
print(" ")
model.train()
if not os.path.isdir("/home/walid_abduljalil/Normflow/saved_models_new_loss"):
    os.makedirs("/home/walid_abduljalil/Normflow/saved_models_new_loss")

epoch_loss_list = []
epoch_d_kl_list = []
epoch_val_loss_list = []
epoch_val_d_kl_list = []
for epoch in range(1):
    iteration_loss_list = []
    iteration_d_kl_list = []

    val_iteration_loss_list = []
    val_iteration_d_kl_list = []
    val_iteration_reconloss_list = []
    model.train()
    # for i, batch in enumerate(train_loader):
    #     data_input = Variable(batch).cuda()
    #     optimizer.zero_grad()
    #
    #     loss_output, d_kl = model(data_input)
    #     loss_output.backward()
    #     iteration_loss_list.append(loss_output.item())
    #     iteration_d_kl_list.append(d_kl.item())
    #
    #     optimizer.step()
    #     scheduler.step()
    #
    # epoch_loss_list.append(np.mean(iteration_loss_list))
    # epoch_d_kl_list.append(np.mean(iteration_d_kl_list))
    # writer.add_scalar("Loss/train", np.mean(iteration_loss_list), epoch)
    # print("train loss: ", np.mean(iteration_loss_list))

    with torch.no_grad():
        model.eval()
        for i, val_batch in enumerate(validation_loader):

            val_data_input = Variable(val_batch).cuda()

            val_loss_output, val_d_kl, recon_loss = model(val_data_input)

            val_iteration_loss_list.append(val_loss_output.item())
            val_iteration_d_kl_list.append(val_d_kl.item())
            val_iteration_reconloss_list.append(recon_loss.item())

        epoch_val_loss_list.append(np.mean(val_iteration_loss_list))
        epoch_val_d_kl_list.append(np.mean(val_iteration_d_kl_list))
        #writer2.add_scalar("Loss/val", np.mean(val_iteration_loss_list), epoch)
        print("loglikelihood: ",np.mean(val_iteration_reconloss_list))
        print("-------------------------------------------------")

    # if epoch % 5 == 0:
    #     save_prefix = os.path.join("/home/walid_abduljalil/Normflow/saved_models_new_loss")
    #     path = "/home/walid_abduljalil/Normflow/saved_models_new_loss/model" + str(epoch) + ".pt"
    #     torch.save({
    #         'epoch': epoch,
    #         'model_state_dict': model.state_dict(),
    #         'optimizer_state_dict': optimizer.state_dict(),
    #         'loss': epoch_loss_list, 'D_KL': epoch_d_kl_list,
    #     }, path)
