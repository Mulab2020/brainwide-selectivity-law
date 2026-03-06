from typing import overload, Optional
import numpy as np
import torch
from torch import nn

from src.utils import parse_values
from src.mixing_levels import generate_class_profile

class ClassAwareGate(nn.Module):
    @overload
    def __init__(self, class_profile: np.ndarray) -> None:
        """
        Initialize a mask with a given class profile numpy matrix.
        """
        ...
    
    @overload
    def __init__(
            self,
            n_units: int,
            n_classes: int,
            alpha: Optional[float] = None,
            beta: Optional[float] = None,
            shuffle_units: bool = False,
            random_control: bool = False
    ) -> None:
        """
        Initialize a mask with units following a certain mixing level distribution.
        """
        ...
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        if len(args) == 1:
            if isinstance(args[0], np.ndarray):
                class_profile = args[0]
            elif isinstance(args[0], int):
                class_profile = self._gen_class_profile(args, kwargs)
        elif 'class_profile' in kwargs:
            class_profile = kwargs['class_profile']
        else:
            class_profile = self._gen_class_profile(args, kwargs)
            
        self.class_profile = nn.Parameter(
            torch.tensor(class_profile, dtype=torch.float32), requires_grad=False
        )
    
    @property
    def n_classes(self) -> int:
        return self.class_profile.shape[0]
    
    def _gen_class_profile(self, src_args, src_kwargs) -> np.ndarray:
        n_units, n_classes, alpha, beta, shuffle_units, random_control = parse_values(
            src_args, src_kwargs, 'n_units', 'n_classes', 'alpha', 'beta', 'shuffle_units', 'random_control'
        )
        if alpha is None or beta is None:
            # Init an empty mask for loading saved mask models
            return np.ones((n_classes, n_units))
        else:
            return generate_class_profile(
                n_classes, alpha, beta, n_units, shuffle_units=shuffle_units, random_control=random_control
            )
        
    def forward(self, x: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # x : (batch_size, n_units), label : (batch_size,) or (batch_size, 1)
        return x * self.class_profile[label.flatten()]
    
    def get_dim_selector(self, label:int):
        dims = torch.where(self.class_profile[label])[0]
        def dim_selector(x: torch.Tensor):
            return x[..., dims]
        
        return dim_selector, len(dims)

