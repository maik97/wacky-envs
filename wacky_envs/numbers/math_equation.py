# TODO: Hacking! https://docs.python.org/3/library/ast.html#ast.literal_eval

import copy
from typing import Dict, Type

from wacky_envs.env_module import ValueEnvModule


class WackyMath(ValueEnvModule):

    def __init__(self, equation: str, var_dict: Dict, dtype: Type = None):
        super(WackyMath, self).__init__()
        self._var_dict = var_dict
        self._equation = equation
        self._dtype = dtype

    def __call__(self, additional_vars: dict = None) -> [float, int, bool]:
        temp_dict = copy.deepcopy(self._var_dict)
        if additional_vars is not None:
            temp_dict.update(additional_vars)
        return self._eval(copy.deepcopy(self._var_dict))

    @property
    def equation(self) -> str:
        return self._equation

    @property
    def var_dict(self) -> dict:
        return self._var_dict

    @property
    def dtype(self) -> type:
        return self._dtype

    def _eval(self, temp_dict) -> [float, int, bool]:
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
        if equation is not None:
            self._equation = equation
        if var_dict is not None:
            self._var_dict = var_dict

    def step(self, delta_t, t, episode_delta_t) -> None:
        temp_dict = self._var_dict.copy()
        temp_dict['episode_delta_t'] = episode_delta_t
        temp_dict['delta_t'] = delta_t
        temp_dict['t'] = t
        self._eval(copy.deepcopy(temp_dict))

    def take_step(self, value, delta_t, t, episode_delta_t) -> [float, int, bool]:
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
