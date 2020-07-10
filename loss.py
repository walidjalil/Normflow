"""
VAE - Loss

@author: walidajalil
"""

import torch
import numpy as np


def get_loss(mu, log_var, gt_images, reconstructions, dataset_size=None):
    d_kl = 0.5 * torch.sum(1 + (2 * log_var) - (mu ** 2) - (torch.exp(log_var) ** 2), dim=1)
    recon_loss = torch.nn.functional.mse_loss(reconstructions, gt_images)
    loss = 0.0004 * torch.mean(-1 * d_kl + recon_loss, dim=0)
    return loss
