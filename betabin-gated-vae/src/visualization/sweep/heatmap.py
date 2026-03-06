__all__ = ['plot_sweep_heatmap']

from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import patches
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb, ListedColormap

from src.visualization.helpers import COLORS
from src.visualization.helpers import exportable_plot

# ================
# Colormap helpers
# ================

def adjust_cmap_saturation(cmap, saturation=1.0):
    colors = np.asarray(
        cmap(np.linspace(0, 1, cmap.N))[:, :3],
        dtype=float
    )

    hsv = rgb_to_hsv(colors)
    hsv[:, 1] = np.clip(hsv[:, 1] * saturation, 0.0, 1.0)

    return ListedColormap(hsv_to_rgb(hsv))


# cmap 'BlWtRd'
def get_blue_white_red_centered_cmap(n=256, data_min=None, data_max=None):
    if data_min is None or data_max is None:
        raise ValueError("data_min and data_max must be provided")

    zero_pos = abs(data_min) / (abs(data_min) + data_max)
    zero_index = int(round(zero_pos * (n - 1)))

    bottom = np.array([0.0, 0.0, 1.0])
    middle = np.array([1.0, 1.0, 1.0])
    top = np.array([1.0, 0.0, 0.0])

    r1 = np.linspace(bottom[0], middle[0], zero_index)
    g1 = np.linspace(bottom[1], middle[1], zero_index)
    b1 = np.linspace(bottom[2], middle[2], zero_index)

    r2 = np.linspace(middle[0], top[0], n - zero_index)
    g2 = np.linspace(middle[1], top[1], n - zero_index)
    b2 = np.linspace(middle[2], top[2], n - zero_index)

    colors = np.vstack([
        np.column_stack([r1, g1, b1]),
        np.column_stack([r2, g2, b2])
    ])

    return ListedColormap(colors)

# ============
# Plot heatmap
# ============


@exportable_plot
def plot_sweep_heatmap(
        node_centers: np.array,
        node_sizes: np.array,
        node_values: Optional[np.array] = None,
        value_name: str = None,
        title: str = None,
        cmap: str = 'Oranges_r',
        font_size: int = 10,
        clim: Optional[Tuple[float, float]] = None,
        fig_size: Tuple[float, float] = (5, 5),
        saturation: float = 0.7,
        show_grid: bool = False,
        dpi: int = 100
) -> Figure:
    """
    Display value heatmap at sweeping areas.

    Parameters
    ----------
    node_centers : np.array
        Centers of nodes.
    node_sizes : np.array
        Sizes of nodes.
    node_values : np.array
        Values associated with each node.
    value_name : str, default=None
    title: str, default = None
    cmap : str, default='Oranges_r'
    dst_path : str | Path | None, optional
        Path to save the figure. If None, the figure is shown instead.
    bbox_inches : str, optional
        Passed to fig.savefig(), default is 'tight'.
    """
    
    label_size = int(font_size * 1.2)
    title_size = int(font_size * 1.2)
    
    if show_grid:
        grid_edge_color = COLORS['black']
        grid_edge_width = 1
    else:
        grid_edge_color = None
        grid_edge_width = 0
    
    with plt.rc_context({
        'font.size': font_size,
        'axes.labelsize': label_size,
        'axes.titlesize': title_size
    }):
        fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
        
        # for colormap
        draw_nodes_only = node_values is None
        if not draw_nodes_only:
            if cmap is None:
                cmap = plt.colormaps['Oranges_r']
            elif cmap == 'BlWtRd':
                cmap = get_blue_white_red_centered_cmap(
                    n = 256,
                    data_min = node_values.min(),
                    data_max = node_values.max()
                )
            else:
                cmap = plt.colormaps[cmap]
            
            # adjust saturation
            cmap = adjust_cmap_saturation(cmap, saturation)
            
            if clim is not None:
                norm = plt.Normalize(*clim)
            else:
                norm = plt.Normalize(node_values.min(), node_values.max())
            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            colors = sm.to_rgba(node_values)

            get_color = lambda color: colors[i]
        else:
            get_color = lambda color: 'white'
        
        corners = node_centers - node_sizes / 2
        
        for i, (corner, size) in enumerate(zip(corners, node_sizes)):
            rect = patches.Rectangle(
                corner, *size, facecolor=get_color(i),
                linewidth=grid_edge_width, edgecolor=grid_edge_color
            )
            ax.add_patch(rect)
        
        ax.set_aspect('equal')
        
        if not draw_nodes_only:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(sm, label=value_name, cax=cax)
        
        ax.set_title(title, pad=font_size)

        ax.set_xlabel('α')
        ax.set_ylabel('β')
        
        left, lower = np.min(corners, axis=0)
        right_idx, upper_idx = np.argmax(corners, axis=0)
        right = corners[right_idx, 0] + node_sizes[right_idx, 0]
        upper = corners[upper_idx, 1] + node_sizes[upper_idx, 1]
        ax.set_xlim(left, right)
        ax.set_ylim(lower, upper)

        return fig

