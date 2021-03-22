#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 02:23:00 2020

@author: walidajalil
"""

import os
import sys
import torch
from iwae_inference_NET import IWAE
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
import math



model_PATH = '/home/walid_abduljalil/Normflow/iwae_saved_models/model140.pt'
mel_PATH = '/home/walid_abduljalil/Normflow/data/'
mel_number = '121201.pt'
mel_save_PATH = '/home/walid_abduljalil/waveglow/mel_spectrograms/IWAE_reconstruction_'
samples_save_PATH = '/home/walid_abduljalil/waveglow/mel_spectrograms/iwae_sample'
ending = '.pt'

# ------ Initialize model
model = IWAE(in_channels=1, out_channels=32, kernel_size=3, n_latent=128, n_samples=6)
model.load_state_dict(torch.load(model_PATH)['model_state_dict'])
model.eval()
model = model.float()
model.cuda()

# ------ Initialize optimizer
print(" ")
print("Beginning Inference now:")
print(" ")
model.train()


for i in range(1):
    inf_data_input = torch.load(mel_PATH+mel_number).T
    inf_data_input = inf_data_input.unsqueeze(dim=0)
    inf_data_input = inf_data_input.unsqueeze(dim=0)
    inf_data_input = inf_data_input.cuda()

    print(inf_data_input.shape)
    print(inf_data_input)
    print("------------------------------------------------------")
    inference_loss, reconstruction = model(inf_data_input)
    #reconstruction = reconstruction.permute(0,1,2,4,3)
    print("shape of output: ", reconstruction.shape)

    reconstruction = reconstruction.squeeze(dim=0)
    reconstruction = reconstruction.squeeze(dim=1)
    reconstruction = reconstruction.detach().cpu()
    torch.save(reconstruction[0,:,:], mel_save_PATH+mel_number)

for j in range(10):
    samples = model.sample()
    print("samples shape:", samples.shape)
    samples = samples.squeeze(dim=0)
    samples = samples.squeeze(dim=0)
    samples = samples.detach().cpu()
    torch.save(samples, samples_save_PATH+str(j+1)+ending)