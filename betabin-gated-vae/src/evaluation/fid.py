"""
Accelerated version for https://github.com/mseitzer/pytorch-fid
Wrapped as a context manager.
"""

__all__ = ['FIDEvaluator']

import gc
from typing import Iterable
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from pytorch_fid.inception import InceptionV3
from torch.nn.functional import adaptive_avg_pool2d

from src.utils import logger

# =======
# helpers
# =======

def get_activations(
    dataloader,
    inception_model,
    dims = 2048,
    device = 'cuda',
    pbar: bool = False
):
    """
    Get InceptionV3 activations of a given dataloader
    (modified from pytorch-fid)
    """
    pbar = tqdm(dataloader) if pbar else dataloader

    inception_model.eval()
    
    # pred_arr = np.empty((dataloader.n_samples, dims), dtype=np.float32)
    pred_arr = torch.empty((dataloader.n_samples, dims))
    start_idx = 0

    for batch_in, batch_lbl in pbar:
        batch_in = batch_in.to(device)

        with torch.no_grad():
            pred = inception_model(batch_in)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
        
        pred = pred.squeeze(3).squeeze(2) # .cpu().numpy()
        pred_arr[start_idx:start_idx + pred.shape[0]] = pred
        start_idx = start_idx + pred.shape[0]
    return pred_arr

# Get mean, cov of given activations
# support batched (by class) input and non-batched input
def get_activation_statistics(activations: torch.Tensor, device='cuda'):
    act_gpu = activations.to(device) # shape (n_classes, samples_per_class, inception_dims)
    mu = act_gpu.mean(dim=-2, keepdim=True) # shape (n_classes, 1, inception_dims)

    centered = act_gpu - mu
    samples_per_class = act_gpu.shape[-2]
    sigma = 1 / (samples_per_class - 1) * (centered.transpose(-2, -1) @ centered)
    return mu.squeeze_(-2), sigma

# Calculate FID using torch
def get_frechet_distance_torch(mu1, sigma1, mu2, sigma2, eps=1e-6):
    with torch.no_grad():
        diff = mu1 - mu2

        # ensure positive definite
        eps_eye = torch.eye(sigma1.size(0), device=sigma1.device) * eps
        sigma1 = sigma1 + eps_eye
        sigma2 = sigma2 + eps_eye

        # sqrt(sigma1)
        eigvals, eigvecs = torch.linalg.eigh(sigma1)
        sqrt_sigma1 = eigvecs @ torch.diag(torch.sqrt(torch.clamp(eigvals, min=0))) @ eigvecs.T

        # sqrt(sqrt_sigma1 * sigma2 * sqrt_sigma1)
        inner = sqrt_sigma1 @ sigma2 @ sqrt_sigma1
        eigvals2, eigvecs2 = torch.linalg.eigh(inner)
        sqrt_inner = eigvecs2 @ torch.diag(torch.sqrt(torch.clamp(eigvals2, min=0))) @ eigvecs2.T

        tr_covmean = torch.trace(sqrt_inner)

        fid = diff.dot(diff) + torch.trace(sigma1) + torch.trace(sigma2) - 2 * tr_covmean
        return fid.item()

# ==============
# FID Eval Class
# ==============

class FIDEvaluator:
    def __init__(
        self,
        dataloader_real: DataLoader,
        n_classes: int,
        emb_dim: int = 2048,
        device: str = 'cuda'
    ):
        # prepare configs
        self.dataloader_real = dataloader_real
        self.n_classes = n_classes
        self.emb_dim = emb_dim
        self.device = device

    def __enter__(self):
        """
        prepare to memory:
        - inception model
        - real data statistics
        """
        logger.info("Preparing FID evaluator... ")
        # prepare models
        logger.info("Loading InceptionV3 model...")
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[self.emb_dim]
        self.inception = InceptionV3([block_idx]).to(self.device)
        self.inception.eval()
        # get real data statistics
        logger.info("Pre-computing real data statistics...")
        act_real = get_activations(
            self.dataloader_real, self.inception, self.emb_dim, self.device, False
        )
        # aggregated statistics
        self.mu_real, self.sigma_real = get_activation_statistics(act_real.unsqueeze_(0), self.device)
        self.mu_real.squeeze_(0)
        self.sigma_real.squeeze_(0)
        # class-wise statistics
        act_real = act_real.reshape((self.n_classes, -1, self.emb_dim))
        self.mu_real_classwise, self.sigma_real_classwise = get_activation_statistics(act_real, self.device)
        logger.info("Done.")
        return self
    
    def evaluate(self, dataloader_gen: Iterable, pbar: bool=False):
        act_gen = get_activations(
            dataloader_gen, self.inception, self.emb_dim, self.device, pbar
        )
        # aggregated FID
        mu_gen, sigma_gen = get_activation_statistics(act_gen.unsqueeze_(0), self.device)
        fid_aggregated = get_frechet_distance_torch(
            mu_gen[0], sigma_gen[0], self.mu_real, self.sigma_real
        )
        
        # class-wise FID
        act_gen = act_gen.reshape((self.n_classes, -1, self.emb_dim))
        mu_gen_classwise, sigma_gen_classwise = get_activation_statistics(act_gen, self.device)
        fid_classwise = np.zeros(self.n_classes)
        for i in range(self.n_classes):
            fid_classwise[i] = get_frechet_distance_torch(
                mu_gen_classwise[i], sigma_gen_classwise[i], self.mu_real_classwise[i], self.sigma_real_classwise[i]
            )
        return fid_aggregated, fid_classwise

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.inception = None
            self.mu_real = None
            self.sigma_real = None
            self.mu_real_classwise = None
            self.sigma_real_classwise = None

            gc.collect()
            if "cuda" in str(self.device):
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"[WARN] FIDEvaluator.__exit__ cleanup error: {e}")
        return False

