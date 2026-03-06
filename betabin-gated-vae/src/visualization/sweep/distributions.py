__all__ = ['plot_sweep_distributions']

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import betabinom, binom, beta
from scipy.special import digamma
from scipy.optimize import root
from typing import Literal, Set, Tuple

from src.visualization.helpers import COLORS, exportable_plot

EPSILON = 1e-6
EPSILON2 = 1e-12

# =======
# Helpers
# =======

# parameter inversion for G-space

def gxgy_from_ab(a, b):
    gx = np.exp(digamma(a) - digamma(a + b))
    gy = np.exp(digamma(b) - digamma(a + b))
    return gx, gy


def ab_from_gxgy(gx, gy, s0=5.0):
    a0 = max(gx, EPSILON) * s0
    b0 = max(gy, EPSILON) * s0
    u0, v0 = np.log(a0 + EPSILON2), np.log(b0 + EPSILON2)

    def F(z):
        u, v = z
        a, b = np.exp(u), np.exp(v)
        gx_hat = np.exp(digamma(a) - digamma(a + b))
        gy_hat = np.exp(digamma(b) - digamma(a + b))
        return [gx_hat - gx, gy_hat - gy]

    sol = root(F, x0=[u0, v0], method='hybr')
    if not sol.success:
        u1, v1 = np.log(max(gx, 1e-6) * 20.0), np.log(max(gy, 1e-6) * 20.0)
        sol = root(F, x0=[u1, v1], method='hybr')

    u, v = sol.x
    a, b = float(np.exp(u)), float(np.exp(v))
    return max(a, 1e-6), max(b, 1e-6)


# plot in a grid cell

def _plot_distribution_in_cell(
        ax: plt.Axes,
        x0, x1, y0, y1,
        mode: Literal['beta', 'betabin', 'binom', 'delta', 'diff'],
        distribution_color: str,
        cell_ylim: Tuple[float, float] | None = None,
        remove_silent: bool = True,
        line_width_scaling: float = 1.0,
        plot_mode: Literal['curve', 'bar'] = 'curve',
        **distribution_kwargs
):
    if mode == 'delta':
        linewidth = 1.5 * line_width_scaling
        p = distribution_kwargs['p']
        dx = x1 - x0
        dy = y1 - y0
        x_p = x0 + dx * p
        ax.arrow(
            x_p, y0, 0, dy = dy * 0.75, 
            head_width = 0.1 * dx, head_length = dy * 0.2, 
            fc = distribution_color, ec = distribution_color, 
            linewidth = linewidth,
            length_includes_head = True
        )
        return
    
    N = distribution_kwargs.get('N', 100)
    k = np.arange(1, N + 1) if remove_silent else np.arange(0, N + 1)
    if mode == 'beta':
        a, b = distribution_kwargs['a'], distribution_kwargs['b']
        pmf = beta.pdf(k / N, a, b) / N
        pmf /= pmf[np.isfinite(pmf)].sum()
    elif mode == 'betabin':
        a, b = distribution_kwargs['a'], distribution_kwargs['b']
        pmf = betabinom.pmf(k, N, a, b)
        pmf /= pmf[np.isfinite(pmf)].sum()
    elif mode == 'binom':
        p = distribution_kwargs['p']
        pmf = binom.pmf(k, N, p)
        pmf /= pmf[np.isfinite(pmf)].sum()
    elif mode == 'diff':
        a, b = distribution_kwargs['a'], distribution_kwargs['b']
        p = distribution_kwargs['p']
        pmf1 = betabinom.pmf(k, N, a, b)
        pmf1 /= pmf1[np.isfinite(pmf1)].sum()
        pmf2 = binom.pmf(k, N, p)
        pmf2 /= pmf2[np.isfinite(pmf2)].sum()
        pmf = pmf1 - pmf2
    
    # rescale to fit into the cell
    if cell_ylim is None:
        finite_max = np.max(pmf[np.isfinite(pmf)])
        finite_min = np.min(pmf[np.isfinite(pmf)])
        cell_ylim = (finite_min, finite_max) # not tested
    range_size = cell_ylim[1] - cell_ylim[0]
    pmf_rescaled = (pmf - cell_ylim[0]) / range_size
    pmf_rescaled = np.minimum(pmf_rescaled, 1.0) # truncate out-of-box part

    if plot_mode == 'curve':
        x_vals = np.linspace(x0, x1, len(k))
        y_vals = y0 + pmf_rescaled * (y1 - y0)
        y_xaxis = y0 - cell_ylim[0] / range_size * (y1 - y0) # position of rescaled x-axis (for fill_between)
        
        ax.plot(x_vals, y_vals, color=distribution_color, lw=line_width_scaling)
        ax.fill_between(x_vals, y_xaxis, y_vals, color=distribution_color, alpha=0.5, linewidth=0)
    elif plot_mode == 'bar':
        n_points = len(k)
        bar_width = (x1 - x0) / n_points
        x_vals = np.linspace(x0, x1, n_points + 1)[:-1] + bar_width / 2
        y_vals = pmf_rescaled * (y1 - y0)
        
        ax.bar(
            x=x_vals, height=y_vals, bottom=y0, width=bar_width,
            color=distribution_color, alpha=0.75, linewidth=0
        )


# ====
# plot
# ====

@exportable_plot
def plot_sweep_distributions(
    mode: Literal['beta', 'betabin', 'binom', 'delta', 'diff'],
    n_classes: int,
    w_steps: int,
    h_steps: int,
    max_alpha: float,
    max_beta: float,
    cell_ylim: Tuple[float, float] | float | None = None,
    distribution_color: str = COLORS['light-green'],
    space: Literal['ab', 'g'] = 'ab',
    font_size: int = 20,
    fig_size: tuple[int, int] = (12, 12),
    dpi: int = 100,
    title: str = None,
    highlight_cells: Set[tuple[int, int]] | None = None,
    line_width_scaling: float = 1.0,
    force_curve: bool = False, # plot curves instead of bars
    remove_silent: bool = True
):
    label_size = int(font_size * 1.2)
    title_size = int(font_size * 1.2)
    
    if (cell_ylim is not None) and (not isinstance(cell_ylim, tuple)):
        cell_ylim = (0, cell_ylim)

    # set font sizes & colors
    sns.set_context(rc = {
        'font.size': font_size,
        'axes.labelsize': label_size,
        'axes.titlesize': title_size,
    })
    sns.set_style(rc = {
        'axes.edgecolor': COLORS['black'],
        'axes.labelcolor': COLORS['black'],
        'text.color': COLORS['black'],
        'xtick.color': COLORS['black'],
        'ytick.color': COLORS['black'],
        'legend.edgecolor': COLORS['black'],
        'text.color': COLORS['black'],
    })
    
    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)

    if space == 'ab':
        dx = max_alpha / w_steps
        dy = max_beta / h_steps

        ax.set_xlim(0, max_alpha)
        ax.set_ylim(0, max_beta)
        ax.set_xticks(range(int(max_alpha) + 1))
        ax.set_yticks(range(int(max_beta) + 1))
        ax.set_xlabel(r'$\alpha$')
        ax.set_ylabel(r'$\beta$')
    elif space == 'g':
        dx = 1.0 / w_steps
        dy = 1.0 / h_steps

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel(r'$G_p = \mathrm{e}^{\mathrm{E}[\ln p]}$')
        ax.set_ylabel(r'$G_{1-p} = \mathrm{e}^{\mathrm{E}[\ln(1-p)]}$')

        sns.lineplot(x=[0, 1], y=[1, 0],
                        ax=ax, color=COLORS['black'],
                        linewidth=2 * line_width_scaling, linestyle='dashed', alpha=0.5)
    else:
        raise ValueError('Unknown space')
    
    # vertical lines
    for i in range(w_steps + 1):
        x_pos = i * dx
        ax.axvline(x=x_pos, color=COLORS['black'], linewidth=line_width_scaling, zorder=5)

    # horizontal lines
    for j in range(h_steps + 1):
        y_pos = j * dy
        ax.axhline(y=y_pos, color=COLORS['black'], linewidth=line_width_scaling, zorder=5)
    
    # decide curve or bar plot
    if mode in {'betabin', 'binom'} and not force_curve:
        plot_mode = 'bar'
    else:
        plot_mode = 'curve'

    color = distribution_color
    for i in range(w_steps):
        for j in range(h_steps):
            x0, x1 = i * dx, (i + 1) * dx
            y0, y1 = j * dy, (j + 1) * dy
            xc = x0 + 0.5 * dx
            yc = y0 + 0.5 * dy

            cell_mode = mode
            if space == 'ab':
                a, b = xc, yc
            else:
                _condition = xc + yc - 1
                if _condition >= EPSILON2:
                    ax.fill_between([x0, x1], [y0, y0], [y1, y1], color=COLORS['black'])
                    continue
                elif 0 <= _condition < EPSILON2:
                    xc -= EPSILON2
                    yc -= EPSILON2
                    if mode == 'beta':
                        cell_mode = 'delta'
                a, b = ab_from_gxgy(xc, yc)
            
            p = a / (a + b)
            if remove_silent:
                p /= 1 - betabinom.pmf(0, n_classes, a, b)
            
            if highlight_cells is not None:
                if (i, j) in highlight_cells:
                    color = distribution_color
                    ax.fill_between([x0, x1], [y0, y0], [y1, y1], color=color, alpha=0.1)
                else:
                    color = 'gray' # 'lightgray'
            
            _plot_distribution_in_cell(
                ax, x0, x1, y0, y1,
                cell_mode, color, cell_ylim, remove_silent,
                line_width_scaling=line_width_scaling,
                N=n_classes, a=a, b=b, p=p,
                plot_mode=plot_mode
            )

            if cell_ylim is not None and (i == w_steps - 1) and (j == 0):
                ax.text(x1 + 0.05 * dx, y0, f'{cell_ylim[0]}', ha='left', va='center',
                        fontsize=font_size, color=color)
                ax.text(x1 + 0.05 * dx, y0 + dy, f'{cell_ylim[1]}',
                        ha='left', va='center', fontsize=font_size, color=color)

    if title is None:
        if mode == 'beta':
            dist_name = 'Beta PDFs'
        elif mode == 'betabin':
            dist_name = f'Beta-Binomial PMFs ($N_{{stims}}=${n_classes})'
        elif mode == 'binom':
            dist_name = 'Binomial PMFs'
        elif mode == 'delta':
            dist_name = 'Delta PDFs'
        elif mode == 'diff':
            dist_name = 'Diff[BetaBin, Binomial]'
        title = f'{dist_name}\nin the Parameter Space'
        
    for spine in ax.spines.values():
        spine.set_linewidth(line_width_scaling)
    ax.tick_params(axis='both', which='major', width=line_width_scaling, length=4 * line_width_scaling)
    ax.set_title(title)
    
    # sns.despine()
    return fig

