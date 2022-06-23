import sys
from typing import Optional, Tuple, Any

import numpy as np
from gym import spaces

from wacky_envs.numbers import WackyNumber


class WackyInt(WackyNumber):
    """Instance of changeable integer."""

    @property
    def dtype(self) -> type:
        """Datatype for `value`, `init_value`, `prev_value`, `delta_value`, ect."""
        return int

    def __init__(self, init_value: int, name:str = None):
        super(WackyInt, self).__init__(init_value)
        self._name = name

    def set_init(self, init_value: int) -> None:
        """Update initial value."""
        super(WackyInt, self).set_init(init_value)

    def set(self, value: int) -> None:
        """Update current value."""
        super(WackyInt, self).set(int(value))

    @property
    def value(self) -> int:
        return super(WackyInt, self).value

    @property
    def init_value(self) -> int:
        return super(WackyInt, self).init_value

    @property
    def prev_value(self) -> int:
        return super(WackyInt, self).prev_value

    @property
    def delta_value(self) -> int:
        return super(WackyInt, self).delta_value

    @property
    def space(self):
        return spaces.Box(low=-np.inf, high=np.inf, shape=1)

    @property
    def real(self) -> int:
        """(From python integer) Returns the real part."""
        return self._value.real

    @property
    def imag(self) -> int:
        """(From python integer) Returns the imaginary part."""
        return self._value.imag

    @property
    def numerator(self) -> int:
        """(From python integer) Integers are their own numerators."""
        return self._value.numerator

    @property
    def denominator(self) -> int:
        """(From python integer) Integers have a denominator of 1."""
        return self._value.denominator

    def conjugate(self) -> int:
        """(From python integer) Returns the complex conjugate."""
        return self._value.conjugate()

    def bit_length(self) -> int:
        """(From python integer) Returns the number of bits necessary to represent an integer in binary, excluding the sign and leading zeros."""
        return self._value.bit_length()

    if sys.version_info >= (3,):
        def to_bytes(self, *args, **kwargs) -> bytes:
            """(From python integer) Return an array of bytes representing an integer."""
            return self._value.to_bytes(*args, **kwargs)

        @classmethod
        def from_bytes(cls, *args, **kwargs) -> int:
            """(From python integer) Return the integer represented by the given array of bytes."""
            return int.from_bytes(*args, **kwargs)

    def __add__(self, x: int) -> int:
        return self._value.__add__(self.read_other(x))

    def __sub__(self, x: int) -> int:
        return self._value.__sub__(self.read_other(x))

    def __mul__(self, x: int) -> int:
        return self._value.__mul__(self.read_other(x))

    def __floordiv__(self, x: int) -> int:
        return self._value.__floordiv__(self.read_other(x))

    if sys.version_info < (3,):
        def __div__(self, x: int) -> int: return self._value.__div__(x)

    def __truediv__(self, x: int) -> float:
        return self._value.__truediv__(self.read_other(x))

    def __mod__(self, x: int) -> int:
        return self._value.__mod__(self.read_other(x))

    def __divmod__(self, x: int) -> Tuple[int, int]:
        return self._value.__divmod__(self.read_other(x))

    def __radd__(self, x: int) -> int:
        return self._value.__radd__(self.read_other(x))

    def __rsub__(self, x: int) -> int:
        return self._value.__rsub__(self.read_other(x))

    def __rmul__(self, x: int) -> int:
        return self._value.__rmul__(self.read_other(x))

    def __rfloordiv__(self, x: int) -> int:
        return self._value.__rfloordiv__(self.read_other(x))

    if sys.version_info < (3,):
        def __rdiv__(self, x: int) -> int: return self._value.__rdiv__(self.read_other(x))

    def __rtruediv__(self, x: int) -> float:
        return self._value.__rtruediv__(self.read_other(x))

    def __rmod__(self, x: int) -> int:
        return self._value.__rmod__(self.read_other(x))

    def __rdivmod__(self, x: int) -> Tuple[int, int]:
        return self._value.__rdivmod__(self.read_other(x))

    def __pow__(self, __x: int, __modulo: Optional[int] = ...) -> Any:
        return self._value.__pow__(self.read_other(__x))  # Return type can be int or float, depending on x.

    def __rpow__(self, x: int) -> Any:
        return self._value.__rpow__(self.read_other(x))

    def __and__(self, n: int) -> int:
        return self._value.__and__(self.read_other(n))

    def __or__(self, n: int) -> int:
        return self._value.__or__(self.read_other(n))

    def __xor__(self, n: int) -> int:
        return self._value.__xor__(self.read_other(n))

    def __lshift__(self, n: int) -> int:
        return self._value.__lshift__(self.read_other(n))

    def __rshift__(self, n: int) -> int:
        return self._value.__rshift__(self.read_other(n))

    def __rand__(self, n: int) -> int:
        return self._value.__rand__(self.read_other(n))

    def __ror__(self, n: int) -> int:
        return self._value.__ror__(self.read_other(n))

    def __rxor__(self, n: int) -> int:
        return self._value.__rxor__(self.read_other(n))

    def __rlshift__(self, n: int) -> int:
        return self._value.__rlshift__(self.read_other(n))

    def __rrshift__(self, n: int) -> int:
        return self._value.__rrshift__(self.read_other(n))

    def __neg__(self) -> int:
        return self._value.__neg__()

    def __pos__(self) -> int:
        return self._value.__pos__()

    def __invert__(self) -> int:
        return self._value.__invert__()

    def __trunc__(self) -> int:
        return self._value.__trunc__()

    if sys.version_info >= (3,):
        def __ceil__(self) -> int: return self._value.__ceil__()

        def __floor__(self) -> int: return self._value.__floor__()

        def __round__(self, ndigits: Optional[int] = ...) -> int: return self._value.__round__()

    def __getnewargs__(self) -> Tuple[int]:
        return self._value.__getnewargs__()

    def __eq__(self, x: object) -> bool:
        return self._value.__eq__(self.read_other(x))

    def __ne__(self, x: object) -> bool:
        return self._value.__ne__(self.read_other(x))

    def __lt__(self, x: int) -> bool:
        return self._value.__lt__(self.read_other(x))

    def __le__(self, x: int) -> bool:
        return self._value.__le__(self.read_other(x))

    def __gt__(self, x: int) -> bool:
        return self._value.__gt__(self.read_other(x))

    def __ge__(self, x: int) -> bool:
        return self._value.__ge__(self.read_other(x))

    '''def __str__(self) -> str:
        return self._value.__str__()'''

    def __float__(self) -> float:
        return self._value.__float__()

    def __int__(self) -> int:
        return self._value.__int__()

    def __abs__(self) -> int:
        return self._value.__abs__()

    def __hash__(self) -> int:
        return self._value.__hash__()

    if sys.version_info >= (3,):
        def __bool__(self) -> bool:
            return self._value.__bool__()
    else:
        def __nonzero__(self) -> bool:
            return self._value.__nonzero__()

    def __index__(self) -> int:
        return self._value.__index__()

    def __repr__(self):
        return f"{self.__class__.__name__}({self._value})"


def main():
    import numpy as np

    test_int = WackyInt(3)
    print(test_int)
    print(str(test_int))

    print(test_int + 2)
    print(test_int - 2)
    print(test_int * 2)
    print(test_int / 2)
    print(10 + test_int)
    print(10 - test_int)
    print(10 * test_int)
    print(10 / test_int)

    test_arr = np.array([test_int], dtype=int)
    print(test_arr)
    print(test_arr.shape)
    print(test_arr.dtype)

    from wacky_envs.numbers import WackyFloat

    print(WackyFloat(1.0) - test_int)
    print(WackyFloat(1.0) * test_int)
    print(WackyFloat(1.0) / test_int)
    print(WackyFloat(1.0) + test_int)


if __name__ == '__main__':
    main()
