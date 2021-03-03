#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 23:23:17 2021

@author: walidajalil
"""
import os
import sys
import glob
import torch
import numpy as np

#if not os.path.exists('/data'):
    #os.mkdir('/data)')


def mel_split(x, n_seconds, padding = False):
    
    # Returns a list of pytorch tensors.
    
    frames_per_split = n_seconds*80
    n_splits = x.shape[1] / frames_per_split
    
    idx_array = np.arange(start = 0, stop = frames_per_split*np.floor(n_splits),step = frames_per_split,dtype = int)
    n_subarrays = np.ceil(n_splits).astype(int)
    
    
    subarray_list = [None] * n_subarrays
    
    if (x.shape[1] < frames_per_split) == True:
        subarray_list[0] = x

    else:
        
        for i, sub_array_idx in enumerate(idx_array):
            subarray_list[i] = x[:,sub_array_idx:sub_array_idx+frames_per_split]
    
    if (x.shape[1] < frames_per_split) == False:
        if (x.shape[1] % frames_per_split) != 0:
            subarray_list[-1] = x[:,idx_array[-1]+frames_per_split:]
    
    
    if padding == True:
        
        last_subarray_size = subarray_list[-1].shape[1]
        pad_idx = frames_per_split - last_subarray_size
        pad_data = x[:,:pad_idx]
        subarray_list[-1] = torch.cat((subarray_list[-1],pad_data),dim=1)
        
    elif (x.shape[1] % frames_per_split) != 0:
        if (x.shape[1] < frames_per_split) == False:
            subarray_list.pop()
        
    return subarray_list

n_files = len(glob.glob('/home/walid_abduljalil/tacotron2/mels/*.pt'))

for i, filepath in enumerate(glob.glob('/home/walid_abduljalil/tacotron2/mels/*.pt')):
    
    data = torch.load(filepath)
    subarray_list = mel_split(data, n_seconds = 2, padding = False)
    for j, subarray in enumerate(subarray_list):
        torch.save(subarray,'/home/walid_abduljalil/Normflow/data/' + str(i)+str(j+1) + '.pt')
        
    if (i % 100) == 0:
        print("# of files split: " +str(i+1)+"/"+str(n_files))
        


