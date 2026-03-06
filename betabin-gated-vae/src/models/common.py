__all__ = [
    'activations',
    'reconstruction_losses'
]

import torch
import torch.nn.functional as F

def mse_reconstruction_loss(recon_x, x):
    return F.mse_loss(recon_x, x, reduction='sum') / x.shape[0]

def bce_reconstruction_loss(recon_x, x):
    return F.binary_cross_entropy(recon_x, x, reduction='sum') / x.shape[0]


activations = {
    'relu': torch.relu_,
    'leaky_relu': torch.nn.functional.leaky_relu,
    'silu': torch.nn.functional.silu,
    'sigmoid': torch.sigmoid_,
    'softplus': torch.nn.functional.softplus,
    'tanh': torch.tanh
}

reconstruction_losses = {
    'mse': mse_reconstruction_loss,
    'bce': bce_reconstruction_loss
}

