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
    """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """

    kld_weight = 0.001
    recons_loss = F.mse_loss(reconstructions, gt_images)

    kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
    loss = recons_loss + kld_weight * kld_loss

    return loss, kld_loss
    # return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss}
