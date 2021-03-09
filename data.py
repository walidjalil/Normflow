import numpy as np
import torch
import glob
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import random
import os
from torch.utils.data.sampler import SubsetRandomSampler

my_path = "/home/walid_abduljalil/Normflow/data/"


class MelDataset(Dataset):
    """ class for dataloader"""

    def __init__(self, path, n_seconds):
        self.frame_size = 80 * n_seconds
        self.height = 80
        print("Initializing dataset...")
        for i, filepath in enumerate(glob.glob(path + "/*.pt")):
            if i%100 == 0:
                print(i)
            self.data = []
            x = torch.load(filepath)
            x = x.T
            x = torch.unsqueeze(x, dim=0)
            self.data.append(x)
        print("Dataset initialized")

    def __getitem__(self, index):
        mel_sample = self.data[index]

        return mel_sample

    def __len__(self):
        return len(self.data)


dataset = MelDataset(my_path, n_seconds=2)
dataset_size = len(dataset)
indices = list(range(dataset_size))
validation_split = .15
shuffle_dataset = True
batch_size = 64
split = int(np.floor(validation_split * dataset_size))

if shuffle_dataset:
    np.random.seed(57)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                           sampler=train_sampler, drop_last=True)
validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                sampler=valid_sampler, drop_last=True)
