__all__ = ['show_imgs']

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from torchvision.utils import make_grid
from matplotlib.figure import Figure

from src.visualization.helpers import exportable_plot

@exportable_plot
def show_imgs(
    imgs: torch.Tensor | np.ndarray, 
    labels: List[int] = None, 
    classes: List[str] = None, 
    channel_first: bool = True, 
    imgs_per_row = 10, 
    dpi = 100,
    padding = 5
) -> Figure:
    """Visualize a batch of images with optional labels displayed below each image.
    
    Parameters
    ----------
    imgs : np.ndarray | torch.Tensor, shape (n, *image_shape)
        A batch of images.
    labels: list of indices, optional
        Index of the class name to be shown below each image. Must be the same length as imgs if provided.
    classes: list of str, optional
        Class name list.
    channel_first : bool, default True
        If True, the imgs are given in `(n c h w)`. Otherwise, `(n h w c)`.
    imgs_per_row : int, default 10
        Number of images to display per row.
    dpi : int, default 100
        Dots per inch for the figure.
    padding : int, default 5
        Padding between images.
    """
    # standardize input
    if isinstance(imgs, np.ndarray):
        imgs = torch.from_numpy(imgs)
    
    if len(imgs.shape) == 3: # (c h w) -> (1 c h w)
        imgs = imgs.unsqueeze(0)
    
    if not channel_first: # (n h w c) -> (n c h w)
        imgs = imgs.permute(0, 3, 1, 2)
    
    # make grid view
    n, c, h, w = imgs.shape
    n_cols = min(n, imgs_per_row)
    n_rows = int(np.ceil(n / n_cols))
    
    grid = make_grid(imgs, nrow=n_cols, padding=padding, normalize=True) # n_row: number of images per row -> n_cols
    grid_np = grid.permute(1, 2, 0).cpu().numpy()
    
    # plot
    fig, ax = plt.subplots(figsize=(n_cols, n_rows), dpi=dpi)
    ax.imshow(grid_np)
    ax.axis('off')

    # labeling
    if labels is not None and classes is not None:
        for i in range(len(imgs)):
            if i >= n: break
            
            # get index
            row_idx = i // n_cols
            col_idx = i % n_cols
            
            # position
            x_pos = col_idx * (w + padding) + padding
            y_pos = row_idx * (h + padding) + padding
            
            # add label
            label_text = classes[labels[i]] if i < len(labels) else ""
            ax.text(
                x_pos, y_pos - 1, label_text, 
                color='yellow', fontsize=8, fontweight='bold',
                ha='left', va='bottom', 
                # bbox=dict(facecolor='black', alpha=0.5, lw=0, pad=0.5) # background
            )

    plt.tight_layout(pad=0)
    return fig

