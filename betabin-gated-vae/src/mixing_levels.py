"""
Utils for mixing level distribution

Concepts
--------
class profile : (n_stims, n_units) binary matrix,
                where each row is a stim's neural responding profile.
"""

__all__ = ['generate_mixing_levels', 'generate_class_profile']


from typing import overload
import numpy as np
from scipy import stats

from src.utils import parse_values


# ==============================
# Generate mixing level samples & class profiles
# ==============================

def generate_mixing_levels(
        n_classes: int,
        alpha: float,
        beta: float,
        n_units: int,
        zero_truncated: bool = True,
        shuffle_units: bool = False
):
    distribution = stats.betabinom(n_classes, alpha, beta)
    x = np.arange(n_classes + 1)
    p = distribution.pmf(x)
    if zero_truncated:
        x = x[1:]
        p = p[1:] / (1 - p[0])
    targets = p * n_units
    counts = np.round(targets).astype(int)
    # rounding errors correction
    remainder = n_units - counts.sum()
    delta_sorted_idx = np.argsort(targets - counts)
    if remainder > 0:
        counts[delta_sorted_idx[-remainder:]] += 1
    else:
        counts[delta_sorted_idx[:-remainder]] -= 1
    
    mixing_levels = x.repeat(counts)
    if shuffle_units:
        np.random.shuffle(mixing_levels)
    return mixing_levels

@overload
def generate_class_profile(
        n_classes: int,
        mixing_levels: np.ndarray
):
    ...

@overload
def generate_class_profile(
        n_classes: int,
        alpha: float,
        beta: float,
        n_units: int,
        zero_truncated: bool = True,
        shuffle_units: bool = False,
        random_control: bool = False
):
    ...

def generate_class_profile(n_classes, *args, **kwargs):
    if len(args) == 1 or 'mixing_levels' in kwargs:
        mixing_levels, = parse_values(args, kwargs, 'mixing_levels')
    else:
        alpha, beta, n_units, zero_truncated, shuffle_units, random_control = parse_values(
            args, kwargs, 'alpha', 'beta', 'n_units', 'zero_truncated', 'shuffle_units', 'random_control'
        )
        zero_truncated = True if zero_truncated is None else zero_truncated # fix default
        mixing_levels = generate_mixing_levels(
            n_classes, alpha, beta, n_units, zero_truncated, shuffle_units
        )
        random_control = bool(random_control)
    
    if not random_control:
        # generate class_profile with intra-class structure
        units_profile = mixing_levels.reshape(-1, 1) > np.arange(n_classes) # (n_units, n_classes)
        for unit in units_profile:
            np.random.shuffle(unit)
        class_profile = units_profile.T.astype(float)
    else:
        # generate a class_profile without intra-class structure, but with the same sparsity of the structured one
        mean_activation = np.round(mixing_levels.sum() / n_classes).astype(int)
        class_profile = np.zeros((n_classes, n_units), dtype=float)
        class_profile[:, :mean_activation] = 1
        for class_ in class_profile:
            np.random.shuffle(class_)
    return class_profile

