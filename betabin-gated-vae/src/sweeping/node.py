__all__ = ['NodeInfo', 'TrainRecord', 'load_node_model']

from pathlib import Path
from dataclasses import dataclass

from src.utils import logger
from src.configs import ModelConfig, TrainConfig, FIDEvalConfig

# =========
# Node info
# =========

@dataclass
class NodeInfo:
    id: int
    dir: str
    
    alpha: float
    beta: float
    n_classes: int

    is_control: bool
    n_epochs: int

    model_cfg: ModelConfig
    
    def __post_init__(self):
        self.dir = Path(self.dir)
        self.ckpt_dir = self.dir / 'model'
        if not self.ckpt_dir.exists():
            self.ckpt_dir.mkdir(parents=True)
            logger.info(f'Created new model directory: {self.ckpt_dir}.')

    def find_latest_ckpt(self) -> int | None:
        saved_epochs = [
            int(str(pth.name).split('EP')[1].split('.pt')[0])
            for pth in self.ckpt_dir.glob('EP*.pt')
        ]
        if saved_epochs:
            return max(saved_epochs)
        return None

    def full_ckpt_path(self, epoch: int) -> Path:
        return self.ckpt_dir / f'EP{epoch}.pt'


# ==========
# Node utils
# ==========

import torch

from src.configs import get_model_config
from src.models import ClassAwareGatedVAE, get_gated_vae_from_config

def load_node_model(
        node: NodeInfo,
        ckpt_epochs: int | None = None,
        model: ClassAwareGatedVAE | None = None,
        eval_mode: bool = True,
        device: str = 'cuda'
):
    if (model is None) or (not isinstance(model, ClassAwareGatedVAE)):
        model_cfg = get_model_config(node.dir / '../model_config.json')
        model = get_gated_vae_from_config(model_cfg)
    
    if ckpt_epochs is None:
        ckpt_epochs = node.find_latest_ckpt()
        if ckpt_epochs is None:
            raise ValueError('No checkpoint found.')
    
    ckpt_path = node.full_ckpt_path(ckpt_epochs)
    model.load_state_dict(torch.load(ckpt_path))
    model.to(device)
    
    if eval_mode:
        model.eval()
    return model

