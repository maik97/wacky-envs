from abc import ABC, abstractmethod
from typing import Any

class WackyNumber(ABC):
    """Parent class for :class:`wacky_envs.constraints.WackyFloat` and :class:`wacky_envs.constraints.WackyInt`"""

    value: [int, float]
    init_value: [int, float]
    prev_value: [int, float]
    delta_x: [int, float]
    dtype: type

    @property
    @abstractmethod
    def dtype(self) -> type:
        """Datatype for all object values (e.g., `value`, `init_value`, `prev_value`, `delta_value`, ect.)"""
        return Any

    def __init__(self, init_value: Any):
        """
        Sets `init_value` and calls reset to assign `init_value` to `value`.

        :param init_value:
        """
        self.set_init(init_value)
        self.reset()

    def set_init(self, init_value: Any) -> None:
        """
        Updates `init_value` after checking if the value is the right datatype.

        :param init_value: Set new `init_value`
        :return: None
        """
        if not isinstance(init_value, self.dtype):
            raise TypeError(f"Expected type {self.dtype}, got {type(init_value)} instead")
        self._init_value = init_value

    def set(self, value: Any) -> None:
        """
        Updates `value` and `prev_value` after checking if the value is the right datatype.

        :param value: Set new value
        :return: None
        """
        if not isinstance(value, self.dtype):
            raise TypeError(f"Expected type {self.dtype}, got {type(value)} instead")
        self._prev_value = self.value
        self._value = value

    def reset(self) -> None:
        """Assigns `init_value` to value and `prev_value`."""
        self._prev_value = self.init_value
        self._value = self.init_value

    @property
    def value(self) -> Any:
        """Current value"""
        return self._value

    @property
    def init_value(self) -> Any:
        """Used for resetting the current value to the initial value."""
        return self._init_value

    @property
    def prev_value(self) -> Any:
        """Value from before the :func:`wacky_envs.constraints.WackyNumber.set` method was called."""
        return self._prev_value

    @property
    def delta_value(self) -> Any:
        """
        Difference between new value and previous value
        after calling the :func:`wacky_envs.constraints.WackyNumber.set` method.
        """
        return self.value - self.prev_value

    def step(self, _, __) -> None:
        """
        Placeholder method, which might be called in :func:`wacky_envs.WackyEnv.step`

        :param _: Placeholder for the parameter `t` (current step)
        :param __: Placeholder for the parameter `delta_t` (current timeframe)
        :return: None
        """
        pass

    def read_other(self, x) -> [int, float]:
        """
        Used to make pythons math work.Reads the value of other :class:`wacky_envs.constraints.WackyNumber`
        objects as integer or float.
        """
        if isinstance(x, WackyNumber):
            return x.value
        else:
            return x

    def __call__(self, *args, **kwargs) -> Any:
        """Alternative for getting the current value. Might be useful for some properties in
        :class:`wacky_envs.constraints.WackyNumber`. (Example: See rewards for the peak-shaver)"""
        return self.value
