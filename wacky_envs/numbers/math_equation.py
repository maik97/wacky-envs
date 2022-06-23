# TODO: Hacking! https://docs.python.org/3/library/ast.html#ast.literal_eval

import copy
from typing import Dict, Type

from wacky_envs.env_module import ValueEnvModule


class WackyMath(ValueEnvModule):
    """Allows implementing math."""

    def __init__(self, equation: str, var_dict: Dict, dtype: Type = None, name: str = None):
        """
        Sets attributes.

        :param equation: An equation in string format. Can have custom variables or may use
            the variables `value`, `t`, `delta_t` or `episode_delta_t`. See the methods for more information.
        :type equation: str

        :param var_dict: Dictionary, where keys are variable names of the equation.
        :type var_dict: dict

        :param dtype: Converts outputs to :attr:`dtype` if specified (optional).
        :type dtype: type
        """
        super(WackyMath, self).__init__()
        self._name = name
        self._var_dict = var_dict
        self._equation = equation
        self._dtype = dtype

    def __call__(self, additional_vars: dict = None) -> [float, int, bool]:
        """
        Returns output of the equation. Optionally, considers variables passed in :attr:`additional_vars`.

        :param additional_vars: Additional or changed variables (optional)
        :type additional_vars: dict

        :return: Output of the equation
        :rtype: Any or :attr:`dtype`
        """
        temp_dict = copy.deepcopy(self._var_dict)
        if additional_vars is not None:
            temp_dict.update(additional_vars)
        return self._eval(copy.deepcopy(self._var_dict))

    @property
    def equation(self) -> str:
        """An equation in string format. Can have custom variables or may use
        the variables `value`, `t`, `delta_t` or `total_delta_t`.
        See the methods for more information.
        """
        return self._equation

    @property
    def var_dict(self) -> dict:
        """Dictionary, where keys are variable names of the equation."""
        return self._var_dict

    @property
    def dtype(self) -> type:
        """Converts outputs of the equation if not `None`."""
        return self._dtype

    def _eval(self, temp_dict) -> [float, int, bool]:
        """
        Calculates output of the equation.

        :param temp_dict: (Modified) copy of :attr:`var_dict`

        :return: Output of the equation
        :rtype: Any or :attr:`dtype`
        """
        if self._dtype is None:
            return eval(self.equation, temp_dict)
        elif self._dtype is int or self._dtype is float:
            return self._dtype(eval(self.equation, temp_dict))
        elif isinstance(self._dtype, ValueEnvModule):
            self._dtype.set(eval(self.equation, temp_dict))
            return self._dtype.value
        else:
            raise TypeError(f'Unknown dtype: {self._dtype}.')

    @property
    def value(self) -> [float, int, bool]:
        return self._eval(copy.deepcopy(self._var_dict))

    def set(self,  equation: str = None, var_dict: Dict = None):
        """Update :attr:`equation` or :attr:`var_dict`"""
        if equation is not None:
            self._equation = equation
        if var_dict is not None:
            self._var_dict = var_dict

    def step(self, t, delta_t, episode_delta_t) -> None:
        """
        Returns output of the equation. Can consider variables `t`, `delta_t` and `episode_delta_t`.

        :param t: Current episode step count
        :type t: int

        :param delta_t: Current step timeframe
        :type delta_t: float

        :param episode_delta_t: Current episode timeframe (so far)
        :type episode_delta_t: float

        :return: Output of the equation
        :rtype: Any or :attr:`dtype`
        """
        temp_dict = self._var_dict.copy()
        temp_dict['episode_delta_t'] = episode_delta_t
        temp_dict['delta_t'] = delta_t
        temp_dict['t'] = t
        self._eval(copy.deepcopy(temp_dict))

    def take_step(self, value, t, delta_t, episode_delta_t) -> [float, int, bool]:
        """
        Returns output of the equation. Can consider variables `value`, `t`, `delta_t` and `episode_delta_t`.

        :param value: A value passed from another instance during its step operation.
        :type value: number

        :param t: Current episode step count
        :type t: int

        :param delta_t: Current step timeframe
        :type delta_t: float

        :param episode_delta_t: Current episode timeframe (so far)
        :type episode_delta_t: float

        :return: Output of the equation
        :rtype: Any or :attr:`dtype`
        """
        temp_dict = self._var_dict.copy()
        temp_dict['value'] = value
        temp_dict['delta_t'] = delta_t
        temp_dict['episode_delta_t'] = episode_delta_t
        temp_dict['t'] = t
        return self._eval(copy.deepcopy(temp_dict))

    def __repr__(self):
        return f"{self.__class__.__name__}({self._equation}, {self._var_dict})"

    @property
    def token_dict(self):
        temp_dict = self._var_dict.copy()
        for v, k in temp_dict.items():
            if isinstance(v, ValueEnvModule):
                temp_dict[k] = v.id

        if isinstance(self.dtype, ValueEnvModule):
            dtype = self.dtype.id
        else:
            dtype = self._dtype

        return {
            'id': self.id,
            'module_type': self.__class__.__name__,
            'equation:': self._equation,
            'var_dict:': temp_dict,
            'dtype': dtype,
        }
