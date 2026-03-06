__all__ = ['plot_mixing_level_distribution', 'plot_mld']

import numpy as np
import torch
import scipy.stats as sst
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.figure import Figure
from typing import overload, Tuple
from numbers import Number

from src.utils import parse_values
from src.visualization.helpers import COLORS, exportable_plot

# ======
# Helper
# ======

def make_bins(n_bins, left=0.0, right=1.0):
    """
    Make bins for histogram.

    Parameters
    ----------
    n_bins : int
        Number of bins.
    left : float
        Left edge of the first bin (inclusive).
    right : float
        Right edge of the last bin (inclusive).

    Returns
    -------
    bin_centers : array, shape (n_bins,)
        Bin centers.
    bins : array, shape (n_bins + 1,)
        Bin edges.
    """
    bin_centers = np.linspace(left, right, n_bins)
    bandwidth = (bin_centers[1] - bin_centers[0]) / 2
    bins = np.zeros(n_bins + 1, dtype=float)
    bins[:-1] = bin_centers - bandwidth
    bins[-1] = right + bandwidth
    return bin_centers, bins


# ===
# API
# ===

# overloads for plot_mixing_level_distribution:

@overload # arg group case 1
def plot_mixing_level_distribution(
    mixing_levels: np.ndarray,
    n_classes: int,
    dpi: int | None = None,
    fig_size: Tuple[float, float] | None = None,
    font_size_base: float | None = None
) -> Figure:
    """Plot mixing-level distribution from samples.

    Parameters
    ----------
    mixing_levels : 1d ndarray
        Shape (n_units,). Either counts or ratios.
    n_classes : int
    dpi : int, optional
    fig_size : tuple, optional
    font_size_base : float, optional
    dst_path : str, optional
    bbox_inches : str, optional
    """
    ...

@overload # arg group case 2
def plot_mixing_level_distribution(
    class_profile: np.ndarray | torch.Tensor,
    alpha: float | None = None,
    beta: float | None = None,
    dpi: int | None = None,
    fig_size: Tuple[float, float] | None = None,
    font_size_base: float | None = None
) -> Figure:
    """Plot mixing-level distribution from a class profile.

    Parameters
    ----------
    class_profile : 2d ndarray | torch.Tensor
        Shape (n_classes, n_units)。
    alpha, beta : float, optional
        Beta-binomial parameters.
    dpi : int, optional
    fig_size : tuple, optional
    font_size_base : float, optional
    dst_path : str, optional
    bbox_inches : str, optional
    """
    ...

@overload # arg group case 3
def plot_mixing_level_distribution(
    alpha: float,
    beta: float,
    n_classes: int,
    dpi: int | None = None,
    fig_size: Tuple[float, float] | None = None,
    font_size_base: float | None = None
) -> Figure:
    """Plot the theoretical Beta-binomial mixing-level distribution.

    Parameters
    ----------
    alpha, beta : float
        Beta-binomial parameters.
    n_classes : int
    dpi : int, optional
    fig_size : tuple, optional
    font_size_base : float, optional
    dst_path : str, optional
    bbox_inches : str, optional
    """
    ...

@exportable_plot
def plot_mixing_level_distribution(*args, **kwargs) -> Figure:
    # parse args
    fig_size, font_size_base, dpi, mixing_levels, n_classes, alpha, beta = (
        None, None, None, None, None, None, None
    )
    case = 0
    if args:
        arg0 = args[0]
        if isinstance(arg0, np.ndarray) or isinstance(arg0, torch.Tensor):
            if arg0.ndim == 1: case = 1
            elif arg0.ndim == 2: case = 2
        elif isinstance(arg0, Number):
            case = 3
    else:
        if 'mixing_levels' in kwargs and 'n_classes' in kwargs: case = 1
        elif 'class_profile' in kwargs: case = 2
        elif 'alpha' in kwargs and 'beta' in kwargs and 'n_classes' in kwargs: case = 3
    
    if case == 1:
        mixing_levels, alpha, beta, n_classes, dpi, fig_size, font_size_base = parse_values(
            args, kwargs,
            'mixing_levels', 'alpha', 'beta', 'n_classes', 'dpi', 'fig_size', 'font_size_base',
        )
        for param in [mixing_levels, n_classes]:
            assert param is not None
        
        if isinstance(mixing_levels, torch.Tensor):
            mixing_levels = mixing_levels.detach().cpu().numpy()
        if mixing_levels.max() <= 1.0: # handle ratio input
            mixing_levels = mixing_levels * n_classes
    elif case == 2:
        class_profile, alpha, beta, dpi, fig_size, font_size_base = parse_values(
            args, kwargs,
            'class_profile', 'alpha', 'beta', 'dpi', 'fig_size', 'font_size_base'
        )
        assert class_profile is not None
        
        if isinstance(class_profile, torch.Tensor):
            class_profile = class_profile.detach().cpu().numpy()
        mixing_levels = np.sum(class_profile, axis=0)
        n_classes = class_profile.shape[0]
    elif case == 3:
        alpha, beta, n_classes, dpi, fig_size, font_size_base = parse_values(
            args, kwargs,
            'alpha', 'beta', 'n_classes', 'dpi', 'fig_size', 'font_size_base'
        )
        for param in [alpha, beta, n_classes]:
            assert param is not None
    else:
        raise ValueError('Invalid arguments.')
    
    fig_size = (5, 5) if fig_size is None else fig_size
    font_size_base = 12 if font_size_base is None else font_size_base
    
    return _plot_mixing_level_distribution_impl(fig_size, font_size_base, dpi, mixing_levels, n_classes, alpha, beta)

plot_mld = plot_mixing_level_distribution # alias


# =====================
# Actual implementation
# =====================

def _plot_mixing_level_distribution_impl(
    fig_size: Tuple[float, float],
    font_size_base,
    dpi: int | None = None,
    mixing_levels: np.ndarray | None = None,
    n_classes: int | None = None,
    alpha: float | None = None,
    beta: float | None = None
) -> Figure:
    label_font_size = font_size_base * (7 / 6)
    title_font_size = font_size_base * (9 / 6)
    
    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)

    if alpha is not None:
        k_locs = np.arange(1, n_classes + 1)
        bbm = sst.betabinom(n_classes, alpha, beta)
        bbm_pm = bbm.pmf(np.arange(1, n_classes + 1)) / (1 - bbm.pmf(0))
        ax.plot(
            k_locs, bbm_pm,
            linewidth=2, linestyle=':', color=COLORS['orange'],
            label=f'Beta-binomial\n(n={n_classes}, α={alpha:.3f}, β={beta:.3f})'
        )
        ax.legend(fontsize=font_size_base)

    if mixing_levels is not None:
        locs, edges = make_bins(n_bins=n_classes, left=1, right=n_classes)
        ax.hist(
            mixing_levels, bins=edges, density=True, color=COLORS['blue'],
            linewidth=0.5, edgecolor=COLORS['black']
        )

    ax.set_xlim((0.5, n_classes + 0.5))
    ax.set_xticks((1 , n_classes))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.tick_params(axis='both', labelsize=font_size_base)

    ax.set_title(f'Mixing-level distribution', fontsize=title_font_size)

    ax.set_xlabel('Mixing-level', fontsize=label_font_size)
    ax.set_ylabel('Fraction', fontsize=label_font_size)

    for sp in ax.spines:
        ax.spines[sp].set_linewidth(1)
    
    return fig

