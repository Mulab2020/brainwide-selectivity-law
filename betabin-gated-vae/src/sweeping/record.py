__all__ = ['TrainRecord', 'FIDRecord']

import numpy as np
from numpy.typing import NDArray
from typing import List, overload, Any
from pathlib import Path
from dataclasses import dataclass

from src.utils import parse_values
from src.sweeping.node import NodeInfo

# ============
# Train record
# ============

@dataclass
class TrainRecord:
    mean_activation: float
    train_loss: List[float] = None
    val_recon: List[float] = None
    val_kld: List[float] = None
    
    def __post_init__(self):
        # Note:
        # if train record is None, then all records are None
        # but if val record is None, train record is not necessarily None
        init = (self.train_loss is None)
        if init:
            self.train_loss = []
            self.val_recon = []
            self.val_kld = []
        else:
            self.train_loss = list(self.train_loss)
            self.val_recon = list(self.val_recon)
            self.val_kld = list(self.val_kld)
    
    @staticmethod
    def get_path(node: NodeInfo) -> Path:
        return node.dir / 'train_record.npz'
    
    @classmethod
    def from_npz(cls, node: NodeInfo):
        record = np.load(TrainRecord.get_path(node))
        return cls(
            mean_activation = record['mean_activation'],
            train_loss = record['train_loss'],
            val_recon = record['val_recon'],
            val_kld = record['val_kld'],
        )
    
    def to_npz(self, node: NodeInfo):
        np.savez(
            file = self.get_path(node),
            mean_activation = self.mean_activation,
            train_loss = self.train_loss,
            val_recon = self.val_recon,
            val_kld = self.val_kld,
        )
    
    @overload # arg group 1
    def extend(
        self,
        train_record: List[float],
        val_recon: List[float],
        val_kld: List[float]
    ):
        ...
    
    @overload # arg group 2
    def extend(
        self,
        train_record: List[float],
        val_record: List[float]
    ):
        ...
    
    def extend(self, *args, **kwargs):
        # decide arg group
        group = 2 if len(args) == 2 or ('val_record' in kwargs) else 1
        
        if group == 1:
            train_record, val_recon, val_kld = parse_values(
                args, kwargs, 'train_record', 'val_recon', 'val_kld', allow_none=False
            )
        else:
            train_record, val_record = parse_values(
                args, kwargs, 'train_record', 'val_record', allow_none=False
            )
            assert 'val_recon' not in kwargs and 'val_kld' not in kwargs
            assert isinstance(val_record, np.ndarray) and val_record.shape[1] == 2, \
                f"val_record must be np.ndarray with shape (N, 2)"
            val_recon, val_kld = val_record.T
        
        self.train_loss.extend(train_record)
        self.val_recon.extend(val_recon)
        self.val_kld.extend(val_kld)


# ==========
# FID record
# ==========

@dataclass
class FIDRecord:
    aggregated: float
    classwise: NDArray[np.floating[Any]] # (n_classes,)
    
    @staticmethod
    def get_path(node: NodeInfo) -> Path:
        return node.dir / 'fid_record.npz'
    
    @classmethod
    def from_npz(cls, node: NodeInfo):
        record = np.load(cls.get_path(node))
        return cls(
            aggregated = record['aggregated'],
            classwise = record['classwise']
        )
    
    def to_npz(self, node: NodeInfo):
        np.savez(
            file = FIDRecord.get_path(node),
            aggregated = self.aggregated,
            classwise = self.classwise
        )

