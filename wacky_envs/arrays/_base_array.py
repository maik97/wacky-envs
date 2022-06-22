import numpy as np
from gym import spaces
from wacky_envs import ValueEnvModule


class BaseArray(ValueEnvModule):

    @property
    def dtype(self) -> type:
        return np.ndarray

    def __init__(
            self,
            value: np.ndarray = None,
            low = None,
            high = None,
    ):
        """Test"""
        super(BaseArray, self).__init__(value)
        if value is not None:
            self._shape = value.shape
        self._low = low if low is not None else 0.0
        self._high = high if high is not None else np.inf

    @property
    def value(self):
        return self._value

    @property
    def shape(self):
        return self._shape

    @property
    def low(self):
        return self._low

    @property
    def high(self):
        return self._high

    def act(self, input):
        self.set(input)

    @property
    def obs(self):
        return self.value

    @property
    def token_dict(self):
        return {
            'id': self.id,
            'module_type': self.__class__.__name__,
            'dtype': self.dtype,
            'value': self.value,
            'init_value': self.init_value,
            'prev_value': self.prev_value,
            'delta_x': self.delta_x,
            'low': self.low,
            'high': self.high,
        }

    @property
    def space(self):
        return spaces.Box(low=self.low, high=self.high, shape=self.shape)

    def __getitem__(self, key):
        return self.value.__getitem__(key)

    def __len__(self):
        return self.value.__len__()
