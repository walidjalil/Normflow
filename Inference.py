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
from data import *
import math

model_PATH = 'Put the path to your model HERE. File should end with .pt'
mel_PATH = 'PUT the path to your mel spectrogram HERE. File should also end with .pt'

# ------ Initialize model
model = VAE(in_channels=1, out_channels=32, kernel_size=3, n_latent=128)
model.load_state_dict(torch.load(model_PATH))
model.eval()
model = model.float()
model.cuda()

print(" ")
print("Beginning inference now:")
print(" ")

if not os.path.isdir("/home/walid_abduljalil/Normflow/saved_inference_output"):
    os.makedirs("/home/walid_abduljalil/Normflow/saved_inference_output")

    with torch.no_grad():

        for i in range(1):
            inf_data_input = torch.load(mel_PATH).cuda()

            inference_loss, inference_d_kl, reconstruction = model(val_data_input)
