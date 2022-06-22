import numpy as np
from dataclasses import dataclass, field
from collections import deque

from gym import spaces

from wacky_envs.numbers import WackyFloat, WackyMath
from wacky_envs.dataframes import FixStepperDataframe
from _old.decisions import ByIndex


@dataclass
class FloatConstr(WackyFloat):

    # parameter:
    name: str
    value: float
    init_value: float

    # properties:
    is_operating: bool
    is_waiting: bool
    error_signal: bool
    op_name: str
    op_id: int

    # numbers:
    upperbound: [WackyMath, WackyFloat, float] = field(default=None)
    lowerbound: [WackyMath, WackyFloat, float] = field(default=None)
    rate_add: [WackyMath, WackyFloat, float] = field(default=None)
    rate_sub: [WackyMath, WackyFloat, float] = field(default=None)
    func_time: WackyMath = field(default=None)

    # settings:
    action_lock: bool = field(default=False)

    # operations:
    errors: np.ndarray = field(default=np.zeros(2))
    op_x: float = field(default=0)
    op_time: float = field(default=0.0)
    delta_op_x: float = field(default=0.0)

    def __init__(
            self,
            init_value: float,
            upperbound: [WackyMath, WackyFloat, float] = None,
            lowerbound: [WackyMath, WackyFloat, float] = None,
            rate_add: [WackyMath, WackyFloat, float] = None,
            rate_sub: [WackyMath, WackyFloat, float] = None,
            func_time: [WackyMath, FixStepperDataframe] = None,
            action_lock: bool = False,
            name: str = None,
    ) -> None:
        """Subclass of :class:`wacky_envs.numbers.WackyFloat`"""

        self._prev_step_values = deque(maxlen=2)
        super().__init__(init_value)

        self.name = name if name is not None else self.__class__.__name__
        self.upperbound = self._init_val(upperbound)
        self.lowerbound = self._init_val(lowerbound)
        self.rate_add = self._init_val(rate_add)
        self.rate_sub = self._init_val(rate_sub)
        self.func_time = self._init_func_time(func_time)
        self.action_lock = action_lock

    @property
    def is_operating(self) -> bool:
        """Checks if value will change during the current step."""
        return self.op_x != 0.0

    @property
    def is_waiting(self) -> bool:
        """Might not be used in the future. Is true if value does not change in the current step,
        but the operation timeframe is not zero."""
        return self.op_x == 0.0 and self.op_time != 0.0

    @property
    def error_signal(self) -> bool:
        """Checks if anything was invalid when :method:`wacky_envs.numbers.IntConstr.delta` was called"""
        return bool(np.any(self.errors))

    @property
    def op_name(self) -> str:
        """
        Name of the current operation when :method:`wacky_envs.numbers.IntConstr.step` is called.

        - 'None': Nothing happens. Assigning a new operation is valid.
        - 'add': Some amount will be added to the current value.
        - 'sub': Some amount will be subtracted to the current value.
        - 'wait': Nothing happens, but assigning a new operation is not valid.
        """
        if self.op_x == 0.0 and self.op_time == 0.0:
            return 'None'
        elif self.op_x > 0.0 and self.op_time != 0.0:
            return 'add'
        elif self.op_x < 0.0 and self.op_time != 0.0:
            return 'sub'
        elif self.op_x == 0.0 and self.op_time != 0.0:
            return 'wait'

    @property
    def op_id(self) -> int:
        """
        Id of the current operation when :method:`wacky_envs.numbers.IntConstr.step` is called.

        - 0: 'None'
        - 1: 'add'
        - 2: 'sub'
        - 3: 'wait'
        """
        if self.op_x == 0.0 and self.op_time == 0.0:
            return 0
        elif self.op_x > 0.0 and self.op_time != 0.0:
            return 1
        elif self.op_x < 0.0 and self.op_time != 0.0:
            return 2
        elif self.op_x == 0.0 and self.op_time != 0.0:
            return 3

    @staticmethod
    def _init_val(val) -> [None, WackyFloat, WackyMath]:
        '''Checks if parameter values are the right type. Converts integers to WackyFloat.'''
        if val is None:
            return None
        elif isinstance(val, float):
            return WackyFloat(val)
        elif isinstance(val, (WackyFloat, WackyMath)):
            return val
        else:
            raise TypeError(f'Expected type: float, WackyFloat, WackyMath. Got {type(val)} instead.')

    @staticmethod
    def _init_func_time(val) -> [None, WackyMath, FixStepperDataframe]:
        '''Checks if parameter func_time is the right type.'''
        if val is None:
            return None
        elif isinstance(val, (WackyMath, FixStepperDataframe, ByIndex)):
            return val
        else:
            raise TypeError(f'Expected type: WackyMath, DataframeFixStepper. Got {type(val)} instead.')

    def reset(self) -> None:
        super(FloatConstr, self).reset()

        for _ in range(self._prev_step_values.maxlen):
            self._prev_step_values.append(self.init_value)

        self.errors = np.zeros(shape=2)
        self.op_x = 0.0
        self.op_time = 0.0
        self.delta_op_x = 0.0

    @property
    def delta_step(self) -> float:
        """
        Total value difference between current and last step.
        Usually: delta_step >= delta_op >= delta_value
        """
        return self._prev_step_values[-1] - self._prev_step_values[-2]

    @property
    def delta_op(self) -> float:
        """
        Amount from the last operation.
        Usually: delta_step >= delta_op >= delta_value
        """
        return self.delta_op_x

    @property
    def delta_value(self) -> float:
        """
        Difference between new value and previous value after calling the set() method.
        Usually: delta_step >= delta_op >= delta_value
        """
        return super(FloatConstr, self).delta_value

    def delta(self, x: float) -> None:
        """Sets up a possible value change based on the restrictions. Must be confirmed in the accept method."""

        if self.upperbound is not None and x > 0.0:
            if (self.value + x) > self.upperbound.value:
                self.errors[0] = 1
                x = self.upperbound.value - self.value

        if self.lowerbound is not None and x < 0.0:
            if (self.value - x) < self.lowerbound.value:
                self.errors[0] = 1
                x = self.lowerbound.value - self.value

        if self.action_lock and (self.is_operating or self.is_waiting):
            self.errors[1] = 1
            return None

        if x > 0:
            if self.rate_add is not None:
                self.to_accept_op_time = self.rate_add.value * x
            else:
                self.to_accept_op_time = 0.0
            self.to_accept_op_x = x

        elif x < 0:
            if self.rate_sub is not None:
                self.to_accept_op_time = self.rate_sub.value * abs(x)
            else:
                self.to_accept_op_time = 0.0
            self.to_accept_op_x = x

        else:
            self.to_accept_op_time = 0.0
            self.to_accept_op_x = x

    def accept(self, x, delta_t):
        """Accept the value change with the corresponding timeframe"""
        self.op_time = delta_t

        if delta_t == 0.0:
            self.set(self.value + x)
            self.delta_op_x = x
            self.op_x = 0.0
        else:
            self.op_x = x
            self.delta_op_x = 0.0

    def wait(self, delta_t: float) -> None:
        """Set up waiting time with delta_t as the timeframe"""
        if not self.is_waiting and not self.is_operating:
            self.op_time = delta_t

    def step(self, t: float, delta_t: float, episode_delta_t: float) -> None:
        """Complete the accepted operation if current timeframe delta_t is <= the required operation time."""

        if self.is_operating:
            if self.op_time <= delta_t:
                self.delta_op_x = self.op_x
                self.set(self.op_x + self.value)
                self.op_x = 0.0
                self.op_time = 0.0
            else:
                self.op_time -= delta_t

        if self.func_time is not None:
            self.set(float(self.func_time.take_step(self.value, t, delta_t, episode_delta_t)))

        if self.lowerbound is not None:
            x_low = max(self.value, self.lowerbound.value)
        else:
            x_low = self.value

        if self.upperbound is not None:
            x_up = min(self.value, self.upperbound.value)
        else:
            x_up = self.value

        self.set(max(x_low, x_up))
        self.errors = np.zeros(shape=2)

        self._prev_step_values.append(self.value)

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
        }

    @property
    def space(self):

        if self.lowerbound is not None:
            low = self.lowerbound.value
        else:
            low = -np.inf

        if self.upperbound is not None:
            high = self.upperbound.value
        else:
            high = np.inf

        return spaces.Box(low=low, high=high, shape=1)


def main():
    from dataclasses import asdict, astuple
    test = FloatConstr(12.0)
    print(test)
    print(test + 2)
    print(test - 2)
    print(test * 2)
    print(test / 2)
    print(10 + test)
    print(10 - test)
    print(10 * test)
    print(10 / test)
    print(asdict(test))
    print(astuple(test))

    math_test = WackyMath('11 + a * 3 -b', {'a': 1, 'b': test})
    print(math_test)
    print(math_test.value)
    test.set(1.0)
    print(math_test)
    print(math_test.value)
    test.reset()
    print(math_test)
    print(math_test.value)

if __name__ == '__main__':
    main()
