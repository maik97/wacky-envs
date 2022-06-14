import copy
from typing import Dict


class WackyMath:

    def __init__(self, equation: str, var_dict: Dict):
        self.var_dict = var_dict

        # TODO:
        #  Schutz vor hacking beachten!!!
        #  https://docs.python.org/3/library/ast.html#ast.literal_eval
        self.equation = equation  # ast.literal_eval(equation)

    def __call__(self, additional_vars: dict = None) -> [float, int, bool]:
        temp_dict = copy.deepcopy(self.var_dict)
        if additional_vars is not None:
            temp_dict.update(additional_vars)
        #print(self.equation)
        #print(temp_dict)
        return eval(self.equation, temp_dict)

    @property
    def value(self):
        return eval(self.equation, copy.deepcopy(self.var_dict))

    def set(self,  equation: str = None, var_dict: Dict = None):
        if equation is not None:
            self.equation = equation
        if var_dict is not None:
            self.var_dict = var_dict

    def step(self, delta_t, t):
        self.var_dict['delta_t'] = delta_t
        self.var_dict['t'] = t

    def take_step(self, value, delta_t, t):
        temp_dict = self.var_dict.copy()
        temp_dict['value'] = value
        temp_dict['delta_t'] = delta_t
        temp_dict['t'] = t
        return eval(self.equation, temp_dict)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.equation}, {self.var_dict})"
