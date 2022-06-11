import numpy as np
from dataclasses import dataclass, field
from wacky_envs.constraints import WackyFloat, WackyMath


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

    # constraints:
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

    def __init__(
            self,
            init_value: float,
            upperbound: [WackyMath, WackyFloat, float] = None,
            lowerbound: [WackyMath, WackyFloat, float] = None,
            rate_add: [WackyMath, WackyFloat, float] = None,
            rate_sub: [WackyMath, WackyFloat, float] = None,
            func_time: WackyMath = None,
            action_lock: bool = False,
            name: str = None,
    ) -> None:
        self._init_value = init_value
        super().__init__(init_value)
        self.reset()

        self.name = name if name is not None else self.__class__.__name__
        self.upperbound = self._init_val(upperbound)
        self.lowerbound = self._init_val(lowerbound)
        self.rate_add = self._init_val(rate_add)
        self.rate_sub = self._init_val(rate_sub)
        self.func_time = self._init_func_time(func_time)
        self.action_lock = action_lock

    @property
    def value(self) -> float:
        return self._value

    @property
    def init_value(self) -> float:
        return self._init_value

    @property
    def is_operating(self) -> bool:
        return self.op_x != 0.0

    @property
    def is_waiting(self) -> bool:
        return self.op_x == 0.0 and self.op_time != 0.0

    @property
    def error_signal(self) -> bool:
        return bool(np.any(self.errors))

    @property
    def op_name(self) -> str:
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
        if val is None:
            return None
        elif isinstance(val, float):
            return WackyFloat(val)
        elif isinstance(val, (WackyFloat, WackyMath)):
            return val
        else:
            raise TypeError(f'Expected type: float, WackyFloat, WackyMath. Got {type(val)} instead.')

    @staticmethod
    def _init_func_time(val) -> [None, WackyMath]:
        if val is None:
            return None
        elif isinstance(val, WackyMath):
            return val
        else:
            raise TypeError(f'Expected type: WackyMath. Got {type(val)} instead.')

    def set_init_value(self, init_value) -> None:
        if not isinstance(init_value, float):
            raise TypeError(f"Expected type float, got {type(init_value)} instead")
        self._init_value = init_value

    def set(self, value) -> None:
        if not isinstance(value, float):
            raise TypeError(f"Expected type float, got {type(value)} instead")
        self._value = value

    def reset(self) -> None:
        self.set(self.init_value)
        self.errors = np.zeros(shape=2)
        self.op_x = 0.0
        self.op_time = 0.0

    def delta(self, x: float) -> None:

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
            if self.rate_add.value is not None:
                self.to_accept_op_time = self.rate_add.value * x
            else:
                self.to_accept_op_time = 0.0
            self.to_accept_op_x = x

        elif x < 0:
            if self.rate_sub.value is not None:
                self.to_accept_op_time = self.rate_sub.value * abs(x)
            else:
                self.to_accept_op_time = 0.0
            self.to_accept_op_x = x

    def accept(self, x, delta_t):
        self.op_time = delta_t
        if delta_t == 0.0:
            self.set(self.value + x)
            self.op_x = 0.0
        else:
            self.op_x = x

    def wait(self, delta_t: float) -> None:
        if not self.is_waiting and not self.is_operating:
            self.op_time = delta_t

    def step(self, t: float, delta_t: float) -> None:

        if self.is_operating:
            if self.op_time <= delta_t:
                self.set(self.op_x + self.value)
                self.op_x = 0.0
                self.op_time = 0.0
            else:
                self.op_time -= delta_t

        if self.func_time is not None:
            self.set(self.func_time.step(self.value, t, delta_t))

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
