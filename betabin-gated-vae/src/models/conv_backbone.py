from typing import List, Optional, Literal
from torch import nn

from src.models.common import activations


# ==============================
# ConvNet
# ==============================

def _make_conv_layer(in_channels, out_channels, kernel_size, stride, padding):
    """ conv + bn """
    # if padding is None:
    #     padding = (kernel_size - 1) // 2 # ceil((k - s) / 2) <= p < floot(k / 2)
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels)
    )

def _make_deconv_layer(in_channels, out_channels, kernel_size, stride, padding):
    """ convT + bn """
    # if padding is None:
    #     padding = (kernel_size - 1) // 2 # ceil((k - s) / 2) <= p < floot(k / 2)
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels)
    )

class ConvNet(nn.Module):
    def __init__(
            self,
            channels: List,
            kernel_sizes: List,
            strides: List,
            paddings: List,
            activation: Literal['relu', 'leaky_relu', 'sigmoid', 'softplus'] = 'relu',
            final_activation: Optional[Literal['relu', 'leaky_relu', 'sigmoid', 'softplus']] = None,
            transposed: bool = False
    ):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        
        if paddings is None:
            paddings = [None] * len(kernel_sizes)
        for i, padding in enumerate(paddings):
            if padding is None:
                assert strides[i] == 1, "padding must be specified for layers with stride > 1."
                paddings[i] = (kernel_sizes[i] - 1) // 2

        make_layer = _make_conv_layer if not transposed else _make_deconv_layer
        curr_channels = channels[0]
        for next_channels, kernel_size, stride, padding in zip(channels[1:], kernel_sizes, strides, paddings):
            self.conv_layers.append(make_layer(curr_channels, next_channels, kernel_size, stride, padding))
            curr_channels = next_channels
        self.activation = activations[activation]
        if final_activation is not None:
            self.final_activation = activations[final_activation]
        else:
            self.final_activation = self.activation
    
    def forward(self, x):
        for conv in self.conv_layers[:-1]:
            x = self.activation(conv(x))
        x = self.final_activation(self.conv_layers[-1](x))
        return x

