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

model_PATH = '/home/walid_abduljalil/Normflow/model45.pt'
mel_PATH = '/home/walid_abduljalil/Normflow/data/8542.pt'
mel_save_PATH = '/home/walid_abduljalil/Normflow/reconstructed_mel.pt'

# ------ Initialize model
model = VAE(in_channels=1, out_channels=32, kernel_size=3, n_latent=128)
model.load_state_dict(torch.load(model_PATH)['model_state_dict'])
model.eval()
model = model.float()
model.cuda()

print(" ")
print("Beginning inference now:")
print(" ")

if not os.path.isdir("/home/walid_abduljalil/Normflow/saved_inference_output"):
    os.makedirs("/home/walid_abduljalil/Normflow/saved_inference_output")
print("hej")
    with torch.no_grad():

        for i in range(1):
            print("hej")
            x = torch.load('/home/walid_abduljalil/Normflow/data/8542.pt')
            print(x.shape)
            inf_data_input = torch.load(mel_PATH).cuda()
            print(inf_data_input.shape)
            inference_loss, inference_d_kl, reconstruction = model(inf_data_input)
            print(reconstruction)
            print("shape of output: ", reconstruction.shape)
            torch.save(reconstruction, mel_save_PATH)
