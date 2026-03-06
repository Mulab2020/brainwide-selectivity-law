__all__ = [
    'logger',               # global logger
    'parse_values',         # helper for overriding args
    'set_global_seed',             # for reproducibility
]

import logging
from typing import Tuple, List, Any

import random
import numpy as np
import torch

# ==============================
# Setup logger
# ==============================

logger = logging.getLogger()
logger.setLevel(logging.INFO)

formatter = logging.Formatter('[%(levelname)s]: %(message)s')
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)

# ==============================
# Parser for overriding args
# ==============================

def parse_values(
        src_args: Tuple[Any],
        src_kwargs: dict,
        *arg_names: str,
        allow_none: bool=True,
        default_value: Any=None
) -> List[Any]:
    """Extract values from a args and kwargs according to the given keys.
    
    Parameters
    ----------
    src_args : Tuple[Any]
        Positional arguments.
    src_kwargs : dict
        Keyword arguments.
    *arg_names : str
        Keys to extract from the source dictionary.
    allow_none : bool, default=True
        Whether to allow None values in the output list.
    default_value : any, default=None
        Value to use if the key is not found in the source dictionary.
        
    Returns:
    list
        List of values corresponding to the specified keys.
    """
    if allow_none:
        return list(src_args) + [src_kwargs.get(k, default_value) for k in arg_names[len(src_args):]]
    
    values = list(src_args)
    for k in arg_names[len(values):]:
        if k not in src_kwargs:
            raise ValueError(f'Arg `{k}` required but not found.')
        values.append(src_kwargs[k])
    return values


# ==============================
# Reproducibility
# ==============================

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

