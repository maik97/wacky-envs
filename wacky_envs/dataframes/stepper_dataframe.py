import numpy as np
from dataclasses import dataclass

from wacky_envs.dataframes import StepperDataframe

@dataclass
class FixStepperDataframe(StepperDataframe):
    """Dataframe is indexed by step counts, assuming that step timeframes are fixed."""

    value: np.ndarray
    idx: int

    def __init__(self, df, dtype=float):
        super(FixStepperDataframe, self).__init__(df, dtype)
        self._idx = 0

    def reset(self) -> None:
        super(FixStepperDataframe, self).reset()
        self._idx = 0

    @property
    def idx(self):
        return self._idx

    @property
    def init_value(self):
        return self.df.iloc[0].to_numpy().astype(self.dtype)

    @property
    def value(self):
        return self.df.iloc[self.idx].to_numpy().astype(self.dtype)

    def step(self, t, _, ___):
        self._idx = t

    def take_step(self, _, t, __, ___):
        self._idx = t
        return self.value

