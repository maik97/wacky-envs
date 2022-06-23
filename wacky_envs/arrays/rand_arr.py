import numpy as np
from wacky_envs.arrays import BaseArray


class EpisodeRandIntArray(BaseArray):
    """Randomizes value when resetting."""

    def __init__(self, shape, low, high):
        super(EpisodeRandIntArray, self).__init__(low=low, high=high)
        self._shape = shape
        self.reset()

    def reset(self):
        self._value = np.random.randint(self.low, self.high, self.shape)


class StepRandIntArray(BaseArray):
    """Randomizes value when resetting and on step() calling."""

    def __init__(self, shape, low, high):
        super(StepRandIntArray, self).__init__(low=low, high=high)
        self._shape = shape
        self.reset()

    def reset(self):
        self._value = np.random.randint(self.low, self.high, self.shape)

    def step(self, _, __, ___) -> None:
        self.reset()


class EpisodeRandFloatArray(BaseArray):
    """Randomizes value when resetting."""

    def __init__(self, shape, low=0.0, high=1.0):
        super(EpisodeRandFloatArray, self).__init__(low=low, high=high)
        self._shape = shape
        self.reset()

    def reset(self):
        self._value = self.low + np.random.random(self.shape) * (self.high - self.low)


class StepRandFloatArray(BaseArray):
    """Randomizes value when resetting and on step() calling."""

    def __init__(self, shape, low=0.0, high=1.0):
        super(StepRandFloatArray, self).__init__(low=low, high=high)
        self._shape = shape
        self.reset()

    def reset(self):
        self._value = self.low + np.random.random(self.shape) * (self.high - self.low)

    def step(self, _, __, ___) -> None:
        self.reset()
