__all__ = ['MetricsBase', 'TrainMetrics', 'EvalMetrics']

from typing import List
from dataclasses import dataclass, asdict, fields

@dataclass
class MetricsBase:
    node_id: int
    alpha: float
    beta: float
    mean_activation: float
    
    @classmethod
    def get_field_names(cls) -> List[str]:
        return [f.name for f in fields(cls)]
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrainMetrics(MetricsBase):
    train_loss: float
    val_recon: float
    val_kld: float


@dataclass
class EvalMetrics(MetricsBase):
    rmse: float
    fid: float