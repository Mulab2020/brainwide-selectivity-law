__all__ = ['make_convnet_pair', 'get_gated_vae_from_config', 'get_vae_helpers']

import numpy as np
from typing import Literal, Tuple, overload
from torch import nn

from src.configs import ConvConfig
from src.configs import ModelConfig
from src.models.common import reconstruction_losses
from src.models.conv_backbone import ConvNet
from src.models.cvae import VAE, ClassAwareGatedVAE


# ==============================
# ConvNet utils
# ==============================

# reshape module
class Reshape(nn.Module):
    @overload
    def __init__(self, shape: Tuple[int, ...]) -> None:
        ...
    
    @overload
    def __init__(self, *shape: int) -> None:
        ...
    
    def __init__(self, *args) -> None:
        super().__init__()
        if len(args) == 1:
            self.shape = args[0]
        else:
            self.shape = args
    
    def forward(self, x):
        return x.reshape(-1, *self.shape)

# padding utils

def get_padding_same(in_size, ksize, stride):
    return (int(np.ceil(in_size / stride) - 1) * stride + ksize - in_size) // 2

def get_conv_outsize(in_size, ksize, stride, padding):
    return (in_size - ksize + 2 * padding) // stride + 1

def get_deconv_outsize(in_size, ksize, stride, padding, output_padding=0):
    return (in_size - 1) * stride - 2 * padding + ksize + output_padding

def get_padding_sizes(in_size:int, ksizes, strides, mode: Literal['conv', 'deconv']):
    if mode == 'conv':
        get_outsize = get_conv_outsize
    elif mode == 'deconv':
        get_outsize = get_deconv_outsize
    else:
        raise ValueError(f'Invalid mode: {mode}')

    io_sizes = [in_size,]
    paddings = []
    for i in range(len(ksizes)):
        in_size = io_sizes[i]
        ksize = ksizes[i]
        stride = strides[i]
        padding = get_padding_same(in_size, ksize, stride)
        paddings.append(padding)
        
        out_size = get_outsize(in_size, ksize, stride, padding)
        io_sizes.append(out_size)
    return paddings, io_sizes


# symmetric convnet pair

def make_convnet_pair(cfg: ConvConfig):
    assert (len(cfg.channels) - 1) == len(cfg.kernel_sizes) == len(cfg.strides), \
        "Channels# should be one more than kernel_sizes# (or strides#) specified."

    # check kernel sizes, strides suitable for image_size
    min_size = np.prod(cfg.strides)
    assert (cfg.img_size[0] % min_size == 0) and (cfg.img_size[1] % min_size == 0), \
        "Product of strides should be a divisor of image size."
    assert not np.any(np.subtract(cfg.kernel_sizes, cfg.strides) % 2), \
        "Difference between kernel_sizes and strides should be even."
    
    # down
    down_paddings, down_out_sizes = get_padding_sizes(cfg.img_size[0], cfg.kernel_sizes, cfg.strides, mode='conv')
    down_cnn = ConvNet(
        cfg.channels,
        cfg.kernel_sizes,
        cfg.strides,
        down_paddings,
        cfg.activation,
        transposed = False
    )
    # downward = nn.Sequential(down_cnn, nn.AdaptiveAvgPool2d(1), nn.Flatten())
    downward = nn.Sequential(down_cnn, nn.Flatten())

    #up
    up_ksizes = cfg.kernel_sizes[::-1]
    up_strides = cfg.strides[::-1]

    up_paddings, _ = get_padding_sizes(down_out_sizes[-1], up_ksizes, up_strides, mode='deconv')
    up_cnn = ConvNet(
        cfg.channels[::-1],
        up_ksizes,
        up_strides,
        up_paddings,
        cfg.activation,
        cfg.final_activation,
        transposed = True
    )
    
    # dummy_input = torch.zeros(size=(1, channels[0], *img_size))
    # fm_shape = down_cnn(dummy_input).shape[1:]
    fm_shape = (cfg.channels[-1], down_out_sizes[-1], down_out_sizes[-1])

    # upward = nn.Sequential(InvAvgPool(*fm_shape), up_cnn)
    # downward.output_dim = downward(dummy_input).shape[1]
    upward = nn.Sequential(Reshape(fm_shape), up_cnn)
    downward.output_dim = np.prod(fm_shape)

    return downward, upward


# ==============================
# VAE utils
# ==============================

def get_gated_vae_from_config(model_cfg: ModelConfig, class_profile: np.ndarray | None = None):
    down, up = make_convnet_pair(model_cfg.convnet)
    vae_cfg = model_cfg.vae
    if class_profile is not None:
        # return ClassAwareGatedVAE(down, up, **model_cfg, class_profile=class_profile)
        return ClassAwareGatedVAE(
            down, up,
            latent_dim = vae_cfg.latent_dim,
            conditional = vae_cfg.conditional,
            n_classes = vae_cfg.n_classes,
            class_profile = class_profile
        )
    else:
        return ClassAwareGatedVAE(
            down, up,
            latent_dim = vae_cfg.latent_dim,
            conditional = vae_cfg.conditional,
            n_classes = vae_cfg.n_classes
        )


def get_vae_helpers(
        vae_model: VAE | ClassAwareGatedVAE | None = None,
        reconstruction_loss: Literal['mse', 'bce'] = 'mse',
        kld_weight: float = 1.0
):
    """Get collate_fn, loss_fn, eval_fn for a VAE models.
    (Adapters for the trainer.)
    """
    recon_loss = reconstruction_losses[reconstruction_loss]

    # helper functions are decided by input-target format
    # flag: whether input and target format is (input, label) or input-only
    io_with_label = isinstance(vae_model, ClassAwareGatedVAE) or vae_model.conditional
    
    if io_with_label:
        # when label input is necessary, i.e. format is (input, label)
        def vae_collate_fn(batch_input, batch_label, device):
            batch_input = batch_input.to(device)
            batch_label = batch_label.to(device)
            return (batch_input, batch_label), (batch_input, batch_label)
        
        def vae_loss_fn(output, batch_target):
            batch_recon, mean, logvar = output
            batch_img, batch_label = batch_target
            return recon_loss(batch_recon, batch_img) + kld_weight * vae_model.kld_loss(mean, logvar, batch_label)
        
        def vae_eval_fn(output, batch_target):
            batch_recon, mean, logvar = output
            batch_img, batch_label = batch_target
            return np.array([
                recon_loss(batch_recon, batch_img).item(),
                vae_model.kld_loss(mean, logvar, batch_label).item()
            ])
    else:
        # when only image input is necessary, i.e. format is input-only
        def vae_collate_fn(batch_input, batch_label, device):
                batch_input = batch_input.to(device)
                return batch_input, batch_input
        
        def vae_loss_fn(output, batch_target):
            batch_recon, mean, logvar = output
            return recon_loss(batch_recon, batch_target) + kld_weight * vae_model.kld_loss(mean, logvar)
        
        def vae_eval_fn(output, batch_target):
            batch_recon, mean, logvar = output
            return np.array([
                recon_loss(batch_recon, batch_target),
                kld_weight * vae_model.kld_loss(mean, logvar)
            ])
    
    return vae_collate_fn, vae_loss_fn, vae_eval_fn

