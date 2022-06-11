import sys
from typing import Optional, Tuple, Any


class WackyInt:

    def __init__(self, value: int):
        self.set(value)

    def step(self, *args, **kwargs):
        return self._value

    def set(self, value):
        if not isinstance(value, int):
            raise TypeError(f"Expected type int, got {type(value)} instead")
        self._value = value

    @property
    def value(self):
        return self._value

    @property
    def real(self) -> int:
        return self._value.real

    @property
    def imag(self) -> int:
        return self._value.imag

    @property
    def numerator(self) -> int:
        return self._value.numerator

    @property
    def denominator(self) -> int:
        return self._value.denominator

    def conjugate(self) -> int:
        return self._value.conjugate()

    def bit_length(self) -> int:
        return self._value.bit_length()

    if sys.version_info >= (3,):
        def to_bytes(self, *args, **kwargs) -> bytes:
            return self._value.to_bytes(*args, **kwargs)

        @classmethod
        def from_bytes(cls, *args, **kwargs) -> int: return int.from_bytes(*args, **kwargs)

    def __add__(self, x: int) -> int:
        return self._value.__add__(x)

    def __sub__(self, x: int) -> int:
        return self._value.__sub__(x)

    def __mul__(self, x: int) -> int:
        return self._value.__mul__(x)

    def __floordiv__(self, x: int) -> int:
        return self._value.__floordiv__(x)

    if sys.version_info < (3,):
        def __div__(self, x: int) -> int: return self._value.__div__(x)

    def __truediv__(self, x: int) -> float:
        return self._value.__truediv__(x)

    def __mod__(self, x: int) -> int:
        return self._value.__mod__(x)

    def __divmod__(self, x: int) -> Tuple[int, int]:
        return self._value.__divmod__(x)

    def __radd__(self, x: int) -> int:
        return self._value.__radd__(x)

    def __rsub__(self, x: int) -> int:
        return self._value.__rsub__(x)

    def __rmul__(self, x: int) -> int:
        return self._value.__rmul__(x)

    def __rfloordiv__(self, x: int) -> int:
        return self._value.__rfloordiv__(x)

    if sys.version_info < (3,):
        def __rdiv__(self, x: int) -> int: return self._value.__rdiv__(x)

    def __rtruediv__(self, x: int) -> float:
        return self._value.__rtruediv__(x)

    def __rmod__(self, x: int) -> int:
        return self._value.__rmod__(x)

    def __rdivmod__(self, x: int) -> Tuple[int, int]:
        return self._value.__rdivmod__(x)

    def __pow__(self, __x: int, __modulo: Optional[int] = ...) -> Any:
        return self._value.__pow__(__x)  # Return type can be int or float, depending on x.

    def __rpow__(self, x: int) -> Any:
        return self._value.__rpow__(x)

    def __and__(self, n: int) -> int:
        return self._value.__and__(n)

    def __or__(self, n: int) -> int:
        return self._value.__or__(n)

    def __xor__(self, n: int) -> int:
        return self._value.__xor__(n)

    def __lshift__(self, n: int) -> int:
        return self._value.__lshift__(n)

    def __rshift__(self, n: int) -> int:
        return self._value.__rshift__(n)

    def __rand__(self, n: int) -> int:
        return self._value.__rand__(n)

    def __ror__(self, n: int) -> int:
        return self._value.__ror__(n)

    def __rxor__(self, n: int) -> int:
        return self._value.__rxor__(n)

    def __rlshift__(self, n: int) -> int:
        return self._value.__rlshift__(n)

    def __rrshift__(self, n: int) -> int:
        return self._value.__rrshift__(n)

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
        return self._value.__eq__(x)

    def __ne__(self, x: object) -> bool:
        return self._value.__ne__(x)

    def __lt__(self, x: int) -> bool:
        return self._value.__lt__(x)

    def __le__(self, x: int) -> bool:
        return self._value.__le__(x)

    def __gt__(self, x: int) -> bool:
        return self._value.__gt__(x)

    def __ge__(self, x: int) -> bool:
        return self._value.__ge__(x)

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


if __name__ == '__main__':
    main()
