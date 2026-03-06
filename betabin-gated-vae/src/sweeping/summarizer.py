__all__ = ['Summarizer']

import pandas as pd
from pathlib import Path
from typing import List, Type, TypeVar, Set, Generic

from src.utils import logger
from src.sweeping.session import SweepSession
from src.sweeping.metrics import MetricsBase

# ==================
# Summarizer context
# ==================

T = TypeVar('T', bound=MetricsBase)

class Summarizer(Generic[T]):

    def __init__(
            self,
            session: SweepSession,
            summary_type: Type[T],
            _buffer_size: int = 100
    ):
        self.session: SweepSession = session
        
        self.path: Path = None
        
        columns: List[str] = summary_type.get_field_names()
        self._data: pd.DataFrame | None = None
        self._columns = set(columns) # for field-match checking
        
        self.buffer_size = _buffer_size
        self._data_buffer = []
        
        self._prepared: bool = False
    
    
    def __enter__(self):
        self._check_prepared()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.dump()
        
    
    # public api
    
    def new(self, file_name: str, overwrite: bool = False):
        if not self.session.is_attached:
            raise RuntimeError('Sweep session must be attached.')
        if self._prepared:
            raise RuntimeError('Summarizer already prepared.')
        
        self.path = Path(self.session.dir) / file_name
        if (not overwrite) and self.path.exists():
            raise FileExistsError(f'{self.path} already exists.')
        self._prepared = True
        return self
    
    def load(self, file_name: str):
        self.path = Path(self.session.dir) / file_name
        
        data = pd.read_csv(self.path, dtype={'node_id': int})
        self._check_fields_match(set(data.columns))
        self._data = data
        self._prepared = True
        return self
    
    def get_data(self) -> pd.DataFrame:
        self._check_prepared()
        self._flush_buffer()
        return self._data
    
    def dump(self):
        self._check_prepared()
        self._flush_buffer()
        if self._data is None:
            logger.warning('No data to dump.')
        else:
            self._data.to_csv(self.path, index=False)
    
    def append(self, summary: MetricsBase, force_flush: bool = False):
        self._check_prepared()
        self._data_buffer.append(summary.to_dict())
        if len(self._data_buffer) >= self.buffer_size or force_flush:
            self._flush_buffer()

    
    # internal helpers
    
    def _flush_buffer(self):
        if len(self._data_buffer) == 0:
            return
        _new_df = pd.DataFrame(self._data_buffer)
        self._check_fields_match(set(_new_df.columns))
        if self._data is None:
            self._data = _new_df
        else:
            self._data = pd.concat([self._data, _new_df], ignore_index=True)
        self._data_buffer = []

    def _check_prepared(self):
        if not self._prepared:
            raise RuntimeError('Summarizer must be prepared.')

    def _check_fields_match(self, columns: Set[str]):
        if columns != self._columns:
            missing = self._columns - columns
            extra = columns - self._columns
            raise ValueError(
                f"Data columns mismatch!\n"
                f"Missing: {missing}\n"
                f"Extra: {extra}"
            )

