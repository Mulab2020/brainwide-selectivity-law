__all__ = ['VAE', 'ClassAwareGatedVAE']

from typing import Literal, Tuple, Optional, Union, overload
import numpy as np
from numpy.typing import ArrayLike
import torch
from torch import nn
from torch.nn.functional import one_hot

from src.models.classaware_gate import ClassAwareGate

# ==============================
# Basic variational autoencoder
# =============================


class VAE(nn.Module):
    def __init__(
            self,
            downward: nn.Module,
            upward: nn.Module,
            latent_dim: int,
            conditional: bool, # whether to use conditional VAE
            n_classes: Optional[int] = None, # number of classes if conditional
    ):
        super().__init__()
        assert hasattr(downward, 'output_dim')
        emb_dim = downward.output_dim
        self.downward = downward
        self.upward = upward
        self.conditional = conditional
        self.latent_dim = latent_dim
        
        if conditional:
            assert n_classes is not None, "`n_classes` must be specified for conditional VAE"
            self.n_classes = n_classes
            self.fc_mean = nn.Linear(emb_dim + n_classes, latent_dim)
            self.fc_logvar = nn.Linear(emb_dim + n_classes, latent_dim)
            self.fc_latent = nn.Linear(latent_dim + n_classes, emb_dim)
        else:
            self.fc_mean = nn.Linear(emb_dim, latent_dim)
            self.fc_logvar = nn.Linear(emb_dim, latent_dim)
            self.fc_latent = nn.Linear(latent_dim, emb_dim)
        
    def sample_latent(self, mean, logvar):
        # mask: (batch_size, latent_dim)
        std = torch.exp(0.5 * logvar) # (batch_size, latent_dim)
        epsilon = torch.randn(size=std.shape).to(self.fc_mean.weight.device)
        z = mean + epsilon * std
        return z
    
    def encode(self, x, onehot_label=None):
        emb = self.downward(x)
        if self.conditional:
            emb = torch.cat([emb, onehot_label], dim=1)
        mean = self.fc_mean(emb)
        logvar = self.fc_logvar(emb)
        z = self.sample_latent(mean, logvar)
        return z, mean, logvar
    
    def decode(self, z, onehot_label=None):
        if self.conditional:
            z = torch.cat([z, onehot_label], dim=1)
        return self.upward(self.fc_latent(z))
    
    def forward(self, input):
        if self.conditional:
            x, label = input
            onehot_label = one_hot(label, self.n_classes)
        else:
            x = input
            onehot_label = None
        z, mean, logvar = self.encode(x, onehot_label)
        x_hat = self.decode(z, onehot_label)
        return x_hat, mean, logvar
    
    def kld_loss(self, mean, logvar, label=None):
        kld = (torch.square(mean) + torch.exp(logvar) - logvar - 1) * 0.5 # (batch_size, latent_dim)
        return torch.sum(kld) / kld.shape[0]
    
    def generate_images(
            self,
            arg: Union[int, torch.Tensor, ArrayLike],
            device='cuda'
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Sample images from latent.
        
        Parameters
        ----------
        arg: int or torch.Tensor or ArrayLike
            - If int, sample n_images=`arg` from latent.
            - If torch.Tensor or ArrayLike, sample images from latent with given labels=`arg`.
        """
        if self.conditional:
            if isinstance(arg, int):
                n_images = arg
                label = torch.randint(0, self.n_classes, (n_images,)).to(device)
            else:
                n_images = len(arg)
                label = arg.to(device) if isinstance(arg, torch.Tensor) else torch.tensor(arg).to(device)
            onehot_label = one_hot(label, self.n_classes)
            # generation samples
            gen_z_norm = torch.randn((n_images, self.latent_dim), device=device)
            gen_x_norm = self.decode(gen_z_norm, onehot_label)
            return gen_x_norm, label
        else:
            assert isinstance(arg, int), "`arg` should be the number of images to generate when `model.conditional` is False."
            gen_z_norm = torch.randn((n_images, self.latent_dim), device=device)
            gen_x_norm = self.decode(gen_z_norm)
            return gen_x_norm


# ==============================
# Variational autoencoder with class-aware masking
# ==============================

class ClassAwareGatedVAE(VAE):
    @overload # arg group 1
    def __init__(
            self,
            downward: nn.Module,
            upward: nn.Module,
            latent_dim: int,
            conditional: bool,
            n_classes: int,
            class_profile: np.ndarray,
    ):
        """
        Initialize with a pre-defined class profile.

        Parameters
        ----------
        downward : nn.Module
            The downward module of the VAE.
        upward : nn.Module
            The upward module of the VAE.
        latent_dim : int
            Dimensionality of the latent space.
        conditional : bool
            Whether the VAE is conditional.
        class_profile : np.ndarray, shape (n_classes, n_units)
            Pre-defined class profile matrix.
        """
        ...
    
    @overload # arg group 2
    def __init__(
            self,
            downward: nn.Module,
            upward: nn.Module,
            latent_dim: int,
            conditional: bool,
            n_classes: int,
            alpha: Optional[float] = None,
            beta: Optional[float] = None,
            shuffle_units: bool = False,
            random_control: bool = False
    ):
        """
        Initialize with parameters to generate a class profile dynamically.

        Parameters
        ----------
        downward : nn.Module
            The downward module of the VAE.
        upward : nn.Module
            The upward module of the VAE.
        latent_dim : int
            Dimensionality of the latent space.
        conditional : bool
            Whether the VAE is conditional.
        n_classes : int
            Number of classes for the mask.
        alpha, beta : float, optional
            Beta-distribution parameters for class_profile generation.
            An all-one dummy class_profile is generated if `alpha` or `beta` are not provided.
        shuffle_units : bool, optional
            Whether to shuffle units in the generated class profile.
        """
        ...
    
    def __init__(
            self,
            downward: nn.Module,
            upward: nn.Module,
            latent_dim: int,
            conditional: bool, # vae is conditional or not 
            n_classes: int,
            *args, **kwargs
    ):
        group = 0
        if len(args) == 1 and isinstance(args[0], np.ndarray): # class_profile
            group = 1
            class_profile = args[0]
        elif 'class_profile' in kwargs:
            class_profile = kwargs['class_profile']
            if class_profile is None: group = 2
            else: group = 1
        
        if group == 1:
            gate = ClassAwareGate(class_profile)
            assert gate.n_classes == n_classes, "`class_profile` and `n_classes` mismatch."
        else:
            gate = ClassAwareGate(latent_dim, n_classes, *args, **kwargs)
        
        # n_classes = gate.n_classes
        super().__init__(downward, upward, latent_dim, conditional, n_classes)
        self.gate = gate
    
    def get_representation(self, input, sampling=False):
        x, label = input
        onehot_label = one_hot(label, self.n_classes) if self.conditional else None
        if sampling:
            z, _, _ = self.encode(x, onehot_label)
        else:
            _, z, _ = self.encode(x, onehot_label) # use mean
        z = self.gate(z, label)
        return z
    
    def generate_images(
            self,
            arg: Union[int, torch.Tensor, ArrayLike],
            clamp_values: bool = True,
            device='cuda'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample images from latent.
        
        Parameters
        ----------
        arg: int or torch.Tensor or ArrayLike
            - If int, sample n_images=`arg` from latent.
            - If torch.Tensor or ArrayLike, sample images from latent with given labels=`arg`.
        clamp_values : bool, optional
            Whether to clamp values to [0, 1].
        Returns
        -------
            - gen_x_norm: torch.Tensor, shape (n_images, n_channels, height, width)
            - label: torch.Tensor, shape (n_images,)
        """
        if isinstance(arg, int):
            n_images = arg
            labels = torch.randint(0, self.n_classes, (n_images,)).to(device)
        else:
            n_images = len(arg)
            labels = arg.to(device) if isinstance(arg, torch.Tensor) else torch.tensor(arg).to(device)

        # generation samples
        gen_z_norm = torch.randn((n_images, self.latent_dim), device=device)
        gen_x_norm = self.gated_decode(gen_z_norm, labels)
        if clamp_values:
            gen_x_norm = gen_x_norm.clamp(0, 1)
        return gen_x_norm, labels

    def gated_decode(self, z, label):
        z = self.gate(z, label)
        onehot_label = one_hot(label, self.n_classes) if self.conditional else None
        return self.decode(z, onehot_label)
    
    def forward(self, input):
        x, label = input
        onehot_label = one_hot(label, self.n_classes) if self.conditional else None
        z, mean, logvar = self.encode(x, onehot_label)
        z = self.gate(z, label)
        x_hat = self.decode(z, onehot_label)
        return x_hat, mean, logvar
    
    def kld_loss(self, mean, logvar, label):
        kld = (torch.square(mean) + torch.exp(logvar) - logvar - 1) * 0.5 # (batch_size, latent_dim)
        kld = self.gate(kld, label)
        kld = torch.sum(kld, dim=1)
        return torch.mean(kld)

