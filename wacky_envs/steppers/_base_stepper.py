import numpy as np
from dataclasses import dataclass
from gym import spaces

from wacky_envs import ValueEnvModule
from typing import Any


@dataclass
class BaseStepper(ValueEnvModule):
    """Base module for steppers."""

    t: int
    delta_t: float
    max_t: int

    @property
    def dtype(self) -> type:
        """Datatype of :attr:`BaseStepper.value`"""
        return float

    def __init__(self, delta_t: Any = None,  max_t: int = None, init_t: int = 0):
        """
        Parent class for all steppers.

        :param delta_t(Any):
        :param max_t(int):
        :param init_steps(int):
        """
        if delta_t is not None:
            super(BaseStepper, self).__init__(init_value=init_t * delta_t)
        else:
            super(BaseStepper, self).__init__()

        self._delta_t = delta_t
        self._max_t = max_t
        self._init_t = init_t
        self._t = 0
        self._total_t = 0

    def next(self):
        """Next step."""
        self._t += 1
        self._total_t += 1
        if self.delta_t is not None:
            self.set(self.value + self.delta_t)

    def reset(self):
        """Reset `value` and episode step counter `t`"""
        self._t = 0
        super(BaseStepper, self).reset()

    @property
    def t(self):
        """Episode step counter"""
        return self._t

    @property
    def init_t(self):
        """Each episode starts at this step."""
        return self._init_t

    @property
    def max_t(self):
        """Each episode terminates at this step."""
        return self._max_t

    @property
    def delta_t(self):
        """Timeframe between two steps."""
        return self._delta_t

    @property
    def total_t(self):
        """Total step counter. Does not reset after episode termination."""
        return self._total_t

    @property
    def episode_delta_t(self):
        """Sum of delta t for current episode."""
        return self.value

    @property
    def value(self):
        """Current value of episode delta t."""
        return self._value

    @property
    def done(self):
        """Episode termination signal if max steps are reached."""
        if self.max_t is None:
            return False
        else:
            return self.t >= self.max_t

    @property
    def obs(self) -> np.ndarray:
        """Timeframe since episode start and/ or episode step counter"""
        if self.delta_t is not None:
            return np.squeeze([self.value, self.t]).reshape(-1)
        else:
            return np.array(self.t)

    @property
    def token_dict(self):
        """Infos for token construction"""
        return {
            'id': self.id,
            'module_type': self.__class__.__name__,
            'dtype': self.dtype,
            'value': self.value,
            'init_value': self.init_value,
            'prev_value': self.prev_value,
            'delta_x': self.delta_x,
            't': self.t,
            'delta_t': self.delta_t,
        }

    @property
    def space(self) -> spaces.Box:
        """Gym space"""
        return spaces.Box(low=0.0, high=np.inf, shape=self.obs.shape)

    def reset_t(self):
        self._t = self._init_t

    def step(self, _, __, ___) -> None:
        raise AttributeError(f"{self.__class__.__name__} has no method 'step'.")

    def take_step(self, _, __, ___, ____):
        raise AttributeError(f"{self.__class__.__name__} has no method 'take_step'.")

    def act(self, input):
        raise AttributeError(f"{self.__class__.__name__} has no method 'act'.")

