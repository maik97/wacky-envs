import sys
from typing import Tuple, overload


class WackyFloat:

    def __init__(self, value: float):
        self.set(value)

    def step(self, *args, **kwargs):
        return self._value

    def set(self, value):
        if not isinstance(value, float):
            raise TypeError(f"Expected type float, got {type(value)} instead")
        self._value = value

    @property
    def value(self):
        return self._value

    def as_integer_ratio(self) -> Tuple[int, int]:
        return self._value.as_integer_ratio()

    def hex(self) -> str:
        return self._value.hex()

    def is_integer(self) -> bool:
        return self._value.is_integer()

    @classmethod
    def fromhex(cls, s: str) -> float:
        return float.fromhex(s)

    @property
    def real(self) -> float:
        return self._value.real

    @property
    def imag(self) -> float:
        return self._value.imag

    def conjugate(self) -> float:
        return self._value.conjugate()

    def __add__(self, x: float) -> float:
        return self._value.__add__(x)

    def __sub__(self, x: float) -> float:
        return self._value.__sub__(x)

    def __mul__(self, x: float) -> float:
        return self._value.__mul__(x)

    def __floordiv__(self, x: float) -> float:
        return self._value.__floordiv__(x)

    if sys.version_info < (3,):
        def __div__(self, x: float) -> float: return self._value.__div__(x)

    def __truediv__(self, x: float) -> float:
        return self._value.__truediv__(x)

    def __mod__(self, x: float) -> float:
        return self._value.__mod__(x)

    def __divmod__(self, x: float) -> Tuple[float, float]:
        return self._value.__divmod__(x)

    def __pow__(self, x: float) -> float:
        return self._value.__pow__(x)  # In Python 3, returns complex if self is negative and x is not whole

    def __radd__(self, x: float) -> float:
        return self._value.__radd__(x)

    def __rsub__(self, x: float) -> float:
        return self._value.__rsub__(x)

    def __rmul__(self, x: float) -> float:
        return self._value.__rmul__(x)

    def __rfloordiv__(self, x: float) -> float:
        return self._value.__rfloordiv__(x)

    if sys.version_info < (3,):
        def __rdiv__(self, x: float) -> float: return self._value.__rdiv__(x)

    def __rtruediv__(self, x: float) -> float:
        return self._value.__rtruediv__(x)

    def __rmod__(self, x: float) -> float:
        return self._value.__rmod__(x)

    def __rdivmod__(self, x: float) -> Tuple[float, float]:
        return self._value.__rdivmod__(x)

    def __rpow__(self, x: float) -> float:
        return self._value.__rpow__(x)

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
        return self._value.__eq__(x)

    def __ne__(self, x: object) -> bool:
        return self._value.__ne__(x)

    def __lt__(self, x: float) -> bool:
        return self._value.__lt__(x)

    def __le__(self, x: float) -> bool:
        return self._value.__le__(x)

    def __gt__(self, x: float) -> bool:
        return self._value.__gt__(x)

    def __ge__(self, x: float) -> bool:
        return self._value.__ge__(x)

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


if __name__ == '__main__':
    main()
