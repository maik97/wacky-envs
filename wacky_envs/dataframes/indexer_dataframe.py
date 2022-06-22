import numpy as np
from dataclasses import dataclass

from wacky_envs.dataframes import BaseDataframe


@dataclass
class IndexerDataframe(BaseDataframe):
    value: np.ndarray
    idx: int

    def __init__(self, df, indexer, dtype=float):
        super(IndexerDataframe, self).__init__(df, dtype)
        self._indexer = indexer

    @property
    def indexer(self):
        return self._indexer

    @property
    def idx(self):
        return self._indexer.value

    @property
    def value(self):
        self._value = self.df.iloc[self.idx].to_numpy().astype(self.dtype)
        return self._value
