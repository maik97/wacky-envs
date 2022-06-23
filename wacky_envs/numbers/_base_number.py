from abc import abstractmethod
from typing import Any
from wacky_envs.env_module import ValueEnvModule


class WackyNumber(ValueEnvModule):
    """
    Base module for numbers (int, float).

    Note:
        Subclasses must implement a :func:`dtype` property.
    """

    @property
    @abstractmethod
    def dtype(self) -> type:
        """Datatype for `value`, `init_value`, `prev_value`, `delta_value`, ect."""
        return Any

    def __init__(self, init_value: Any):
        r"""
        Sets :attr:`init_value` and calls :func:`reset`.

        :param init_value: The initial value. Resets to this value at the start of each episode.
        """
        super(WackyNumber, self).__init__(init_value)

    @staticmethod
    def read_other(x) -> [int, float]:
        """
        Makes pythons math work. Reads the value of other :class:`wacky_envs.numbers.WackyNumber`
        objects as integer or float.
        """
        if isinstance(x, WackyNumber):
            return x.value
        else:
            return x
