from abc import abstractmethod
from typing import Any
from wacky_envs.env_module import ValueEnvModule


class WackyNumber(ValueEnvModule):
    """
    Parent class for :class:`wacky_envs.numbers.WackyFloat` and :class:`wacky_envs.numbers.WackyInt`

    """

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
        super(WackyNumber, self).__init__(init_value)
        self.set_init(init_value)
        self.reset()


    def read_other(self, x) -> [int, float]:
        """
        Used to make pythons math work.Reads the value of other :class:`wacky_envs.numbers.WackyNumber`
        objects as integer or float.
        """
        if isinstance(x, WackyNumber):
            return x.value
        else:
            return x
