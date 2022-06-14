import numpy as np
from dataclasses import dataclass


@dataclass
class DataframeFixStepper:

    value: np.ndarray
    dtype: object = np.ndarray
    t: int = 0

    def __init__(self, df):
        self.df = df
        self.dtype = np.ndarray
        self.reset()

    def reset(self):
        self.t = 0
        self()

    def __call__(self):
        return self.df.iloc[self.t].to_numpy()

    @property
    def value(self):
        return self.df.iloc[self.t].to_numpy()

    @property
    def float(self):
        return self.df.iloc[self.t].to_numpy().astype(float)

    def step(self, t, _):
        self.t = t

    def take_step(self, _, t, __):
        self.t = t
        return self.value

