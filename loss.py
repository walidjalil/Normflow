"""
VAE - Loss

@author: walidajalil
"""

import torch
from torch.nn import functional as F


def get_loss(mu, log_var, gt_images, reconstructions, dataset_size=None):
    d_kl = 0.5 * torch.sum(1 + (2 * log_var) - (mu ** 2) - (torch.exp(log_var) ** 2), dim=1)
    recon_loss = torch.nn.functional.mse_loss(reconstructions, gt_images)
    loss = torch.mean((-0.001 * d_kl) + recon_loss, dim=0)
    return loss, torch.mean(-1 * d_kl, dim=0)


def loss_function(mu, log_var, gt_images, reconstructions, dataset_size=None):
    kld_weight = 0.001
    recons_loss = F.mse_loss(reconstructions, gt_images)

    kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
    loss = recons_loss + kld_weight * kld_loss

    return loss, kld_loss
    # return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss}


def get_iwae_loss(mu, log_var, gt_images, reconstructions, n_samples):
    gt_images = gt_images.repeat(n_samples, 1, 1, 1, 1).permute()
    kld_weight = 0.001208

    reconstruction_loss = ((reconstructions - gt_images) ** 2).flatten(2).mean(-1)
    kld_loss = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=2)
    log_weight = (reconstruction_loss + kld_weight * kld_loss)
    weight = F.softmax(log_weight, dim=-1)
    loss = torch.mean(torch.sum(weight * log_weight, dim=-1), dim = 0)
    return loss
