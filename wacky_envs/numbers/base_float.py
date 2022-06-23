import sys
import numpy as np
from typing import Tuple, overload
from gym import spaces
from wacky_envs.numbers import WackyNumber


class WackyFloat(WackyNumber):
    """Instance of changeable float."""

    @property
    def dtype(self) -> type:
        """Datatype float for `value`, `init_value`, `prev_value`, `delta_value`, ect."""
        return float

    def __init__(self, init_value: float, name:str = None):
        super(WackyFloat, self).__init__(init_value)
        self._name = name

    def set_init(self, init_value: float) -> None:
        """Update initial value."""
        super(WackyFloat, self).set_init(init_value)

    def set(self, value: float) -> None:
        """Update current value."""
        super(WackyFloat, self).set(value)

    @property
    def value(self) -> float:
        return super(WackyFloat, self).value

    @property
    def init_value(self) -> float:
        return super(WackyFloat, self).init_value

    @property
    def prev_value(self) -> float:
        return super(WackyFloat, self).prev_value

    @property
    def delta_value(self) -> float:
        return super(WackyFloat, self).delta_value

    @property
    def space(self):
        return spaces.Box(low=-np.inf, high=np.inf, shape=1)

    def as_integer_ratio(self) -> Tuple[int, int]:
        """(From python float) Returns a pair of integers whose ratio is exactly equal to
        the original float and with a positive denominator."""
        return self._value.as_integer_ratio()

    def hex(self) -> str:
        """(From python float) Converts value to the corresponding hexadecimal number in string form and returns it."""
        return self._value.hex()

    def is_integer(self) -> bool:
        """(From python float) Returns True if the float instance is finite with integral value, and False otherwise."""
        return self._value.is_integer()

    @classmethod
    def fromhex(cls, s: str) -> float:
        """(From python float) Returns a new bytearray object initialized from a string of hex numbers."""
        return float.fromhex(s)

    @property
    def real(self) -> float:
        """(From python float) Returns the real part."""
        return self._value.real

    @property
    def imag(self) -> float:
        """(From python float) Returns the imaginary part."""
        return self._value.imag

    def conjugate(self) -> float:
        """(From python float) Returns the complex conjugate."""
        return self._value.conjugate()

    def __add__(self, x: float) -> float:
        return self._value.__add__(self.read_other(x))

    def __sub__(self, x: float) -> float:
        return self._value.__sub__(self.read_other(x))

    def __mul__(self, x: float) -> float:
        return self._value.__mul__(self.read_other(x))

    def __floordiv__(self, x: float) -> float:
        return self._value.__floordiv__(self.read_other(x))

    if sys.version_info < (3,):
        def __div__(self, x: float) -> float: return self._value.__div__(self.read_other(x))

    def __truediv__(self, x: float) -> float:
        return self._value.__truediv__(self.read_other(x))

    def __mod__(self, x: float) -> float:
        return self._value.__mod__(self.read_other(x))

    def __divmod__(self, x: float) -> Tuple[float, float]:
        return self._value.__divmod__(self.read_other(x))

    def __pow__(self, x: float) -> float:
        return self._value.__pow__(self.read_other(x))  # In Python 3, returns complex if self is negative and x is not whole

    def __radd__(self, x: float) -> float:
        return self._value.__radd__(self.read_other(x))

    def __rsub__(self, x: float) -> float:
        return self._value.__rsub__(self.read_other(x))

    def __rmul__(self, x: float) -> float:
        return self._value.__rmul__(self.read_other(x))

    def __rfloordiv__(self, x: float) -> float:
        return self._value.__rfloordiv__(self.read_other(x))

    if sys.version_info < (3,):
        def __rdiv__(self, x: float) -> float: return self._value.__rdiv__(self.read_other(x))

    def __rtruediv__(self, x: float) -> float:
        return self._value.__rtruediv__(self.read_other(x))

    def __rmod__(self, x: float) -> float:
        return self._value.__rmod__(self.read_other(x))

    def __rdivmod__(self, x: float) -> Tuple[float, float]:
        return self._value.__rdivmod__(self.read_other(x))

    def __rpow__(self, x: float) -> float:
        return self._value.__rpow__(self.read_other(x))

    def __getnewargs__(self) -> Tuple[float]:
        return self._value.__getnewargs__()

    def __trunc__(self) -> int:
        return self._value.__trunc__()

    if sys.version_info >= (3,):
        @overload
        def __round__(self, ndigits: None = ...) -> int: return self._value.__round__(ndigits)

        @overload
        def __round__(self, ndigits: int) -> float: return self._value.__round__(ndigits)

    def __eq__(self, x: object) -> bool:
        return self._value.__eq__(self.read_other(x))

    def __ne__(self, x: object) -> bool:
        return self._value.__ne__(self.read_other(x))

    def __lt__(self, x: float) -> bool:
        return self._value.__lt__(self.read_other(x))

    def __le__(self, x: float) -> bool:
        return self._value.__le__(self.read_other(x))

    def __gt__(self, x: float) -> bool:
        return self._value.__gt__(self.read_other(x))

    def __ge__(self, x: float) -> bool:
        return self._value.__ge__(self.read_other(x))

    def __neg__(self) -> float:
        return self._value.__neg__()

    def __pos__(self) -> float:
        return self._value.__pos__()

    '''def __str__(self) -> str:
        return self._value.__str__()'''

    def __int__(self) -> int:
        return self._value.__int__()

    def __float__(self) -> float:
        return self._value.__float__()

    def __abs__(self) -> float:
        return self._value.__abs__()

    def __hash__(self) -> int:
        return self._value.__hash__()

    if sys.version_info >= (3,):
        def __bool__(self) -> bool:
            return self._value.__bool__()
    else:
        def __nonzero__(self) -> bool:
            return self._value.__nonzero__()

    def __repr__(self):
        return f"{self.__class__.__name__}({self._value})"


def main():
    import numpy as np

    test_float = WackyFloat(3.0)
    print(test_float)
    print(str(test_float))

    print(test_float + 2)
    print(test_float - 2)
    print(test_float * 2)
    print(test_float / 2)
    print(10 + test_float)
    print(10 - test_float)
    print(10 * test_float)
    print(10 / test_float)

    test_arr = np.array([test_float], dtype=float)
    print(test_arr)
    print(test_arr.shape)
    print(test_arr.dtype)

    print(WackyFloat(1.0) - test_float)
    print(WackyFloat(1.0) * test_float)
    print(WackyFloat(1.0) / test_float)
    print(WackyFloat(1.0) + test_float)

    test_float.set(1.0)
    print(test_float.value)
    try:
        test_float.set(1)
    except Exception as e:
        print('test exception:')
        print(e)


if __name__ == '__main__':
    main()
