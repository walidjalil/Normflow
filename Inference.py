#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 02:23:00 2020

@author: walidajalil
"""

import os
import sys
import torch
from Inference_NET import VAE
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
#from data import *
import math

model_PATH = '/home/walid_abduljalil/Normflow/model325.pt'
mel_PATH = '/home/walid_abduljalil/Normflow/data/121201.pt'
mel_save_PATH = '/home/walid_abduljalil/Normflow/reconstructed_poop2_325.pt'
samples_save_PATH = '/home/walid_abduljalil/Normflow/VAE_sample.pt'

# ------ Initialize model
model = VAE(in_channels=1, out_channels=32, kernel_size=3, n_latent=128)
model.load_state_dict(torch.load(model_PATH)['model_state_dict'])
model.eval()
model = model.float()
model.cuda()

print(" ")
print("Beginning inference now:")
print(" ")


for i in range(1):
    inf_data_input = torch.load(mel_PATH).T
    inf_data_input = inf_data_input.unsqueeze(dim=0)
    inf_data_input = inf_data_input.unsqueeze(dim=0)
    inf_data_input = inf_data_input.cuda()

    print(inf_data_input.shape)
    print(inf_data_input)
    print("------------------------------------------------------")
    inference_loss, inference_d_kl, reconstruction = model(inf_data_input)
    print(reconstruction.permute(0,1,3,2))
    print("shape of output: ", reconstruction.shape)

    reconstruction = reconstruction.squeeze(dim=0)
    reconstruction = reconstruction.squeeze(dim=0)
    reconstruction = reconstruction.detach().cpu()
    torch.save(reconstruction, mel_save_PATH)


    samples = model.sample()
    print("samples shape:", samples.shape)
    samples = samples.squeeze(dim=0)
    samples = samples.squeeze(dim=0)
    samples = samples.detach().cpu()
    torch.save(samples, samples_save_PATH)
