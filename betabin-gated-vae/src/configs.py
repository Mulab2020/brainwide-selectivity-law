__all__ = [
    'get_model_config',
    'get_sweep_config',
    'compare_sweep_configs',
    'ModelConfig',
    'SweepConfig',
]


from dataclasses import dataclass, asdict
from typing import List, Tuple, Literal
import json
from pathlib import Path
import numpy as np

from src.utils import logger


# ==============================
# Model Config
# ==============================

@dataclass
class VAEConfig:
    latent_dim: int
    conditional: bool
    n_classes: int
    @classmethod
    def from_dict(cls, config: dict) -> 'VAEConfig':
        return cls(**config)

@dataclass
class ConvConfig:
    img_size: Tuple[int, int]
    channels: List[int]
    kernel_sizes: List[int]
    strides: List[int]
    activation: Literal['relu', 'leaky_relu', 'silu', 'sigmoid', 'softplus', 'tanh']
    final_activation: Literal['relu', 'leaky_relu', 'sigmoid', 'softplus'] | None = 'sigmoid'
    
    @classmethod
    def from_dict(cls, config: dict) -> 'ConvConfig':
        return cls(**config)

@dataclass
class ModelConfig:
    vae: VAEConfig
    convnet: ConvConfig
    @classmethod
    def from_dict(cls, config: dict) -> 'ModelConfig':
        vae = VAEConfig.from_dict(config['vae'])
        convnet = ConvConfig.from_dict(config['convnet'])
        return cls(vae, convnet)
    
    def to_dict(self) -> dict:
        return dict_to_json_serializable(asdict(self))
    
    def save(self, dst_path: str, overwrite: bool = False):
        dst_path = Path(dst_path)
        if dst_path.exists() and not overwrite:
            raise FileExistsError(f'{dst_path} already exists. Pass `overwrite=True` to overwrite.')
        with open(dst_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)

# ==============================
# Sweeping Config
# ==============================

@dataclass
class TrainConfig:
    batch_size: int
    lr: float
    n_epochs: int
    eval_every_n_epochs: int
    save_every_n_epochs: int
    
    num_workers_train: int
    num_workers_val: int

    @classmethod
    def from_dict(cls, config: dict) -> 'TrainConfig':
        return cls(**config)

@dataclass
class FIDEvalConfig:
    batch_size: int
    inception_dims: int
    samples_per_class: int
    num_workers: int
    
    @classmethod
    def from_dict(cls, config: dict) -> 'FIDEvalConfig':
        return cls(**config)
    

@dataclass
class SweepConfig:
    random_seed: int
    is_control: bool

    out_dir: str
    model_cfg_path: str
    dataset_dir: str
    
    range_alpha: Tuple[float, float]
    range_beta: Tuple[float, float]
    n_nodes: Tuple[int, int]
    
    train: TrainConfig
    fid_eval: FIDEvalConfig
    
    device: str
    
    @classmethod
    def from_dict(cls, config: dict) -> 'SweepConfig':
        config = dict(config)
        train = TrainConfig.from_dict(config.pop('train'))
        fid_eval = FIDEvalConfig.from_dict(config.pop('fid_eval'))
        return cls(train=train, fid_eval=fid_eval, **config)

    def to_dict(self) -> dict:
        return dict_to_json_serializable(asdict(self))
    
    def save(self, dst_path: str, overwrite: bool = False):
        dst_path = Path(dst_path)
        if dst_path.exists() and not overwrite:
            raise FileExistsError(f'{dst_path} already exists. Pass `overwrite=True` to overwrite.')
        with open(dst_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)

    
# ==============================
# For json serialization
# ==============================

_registered_typemap = {
    np.ndarray: lambda x: x.tolist(),
    Path: lambda x: str(x)
}

def dict_to_json_serializable(dataclass_obj: dict):
    res = dataclass_obj.copy()
    for k, v in res.items():
        if isinstance(v, dict):
            res[k] = dict_to_json_serializable(v)
        for t in _registered_typemap:
            if isinstance(v, t):
                res[k] = _registered_typemap[t](v)
    return res

# ==============================
# APIs
# ==============================

def get_model_config(model_cfg_path: str) -> ModelConfig:
    model_cfg_path = Path(model_cfg_path)
    if model_cfg_path.is_dir():
        model_cfg_path = model_cfg_path / 'model_config.json'
    with open(model_cfg_path, 'r') as f:
        model_cfg = json.load(f)
    return ModelConfig.from_dict(model_cfg)


def get_sweep_config(sweep_cfg_path: str) -> SweepConfig:
    sweep_cfg_path = Path(sweep_cfg_path)
    if sweep_cfg_path.is_dir():
        sweep_cfg_path = sweep_cfg_path / 'sweeping_config.json'
    with open(sweep_cfg_path, 'r') as f:
        sweep_cfg = json.load(f)
    return SweepConfig.from_dict(sweep_cfg)


def compare_sweep_configs(cfg1: SweepConfig, cfg2: SweepConfig) -> bool:
    # sweeping_cfg
    if not (
        cfg1.random_seed == cfg2.random_seed and
        np.all(cfg1.range_alpha == np.array(cfg2.range_alpha)) and
        np.all(cfg1.range_beta == np.array(cfg2.range_beta)) and
        np.all(cfg1.n_nodes == np.array(cfg2.n_nodes)) and
        cfg1.is_control == cfg2.is_control
    ):
        logger.warning('Sweeping config mismatch.')
        return False
    # model cfg
    if not cfg1.model_cfg_path == cfg2.model_cfg_path:
        with open(cfg1.model_cfg_path, 'r') as f1, open(cfg2.model_cfg_path, 'r') as f2:
            model_cfg1 = ModelConfig.from_dict(json.load(f1))
            model_cfg2 = ModelConfig.from_dict(json.load(f2))
        if model_cfg1 != model_cfg2:
            logger.warning('Model config mismatch.')
            return False
    return True

