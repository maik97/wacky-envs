from dataclasses import dataclass
from wacky_envs.numbers import FloatConstr, IntConstr, WackyMath
from wacky_envs.callables import BaseCallable

@dataclass
class ValueUpdate(BaseCallable):
    """Updates value of :attr:`to_update` with the value of :attr:`set_from`."""

    def __init__(self, to_update, set_from):
        """
        Initialize
        :param to_update:
        :param set_from:
        """
        super(ValueUpdate, self).__init__()
        self.to_update = to_update
        self.set_from = set_from

    def __call__(self):
        self.to_update.delta(self.set_from.value - self.to_update.value)
        self.to_update.accept(self.to_update.to_accept_op_x, self.to_update.to_accept_op_time)


@dataclass
class ValueTransfer(BaseCallable):
    """Transfers :attr:`delta_x` from :attr:`trans_from` to the value of :attr:`trans_to`."""

    delta_x: [IntConstr, FloatConstr, WackyMath]
    trans_from: [IntConstr, FloatConstr]
    trans_to: [IntConstr, FloatConstr]
    trans_from_func: WackyMath
    trans_to_func: WackyMath

    def __init__(
            self,
            delta_x: [IntConstr, FloatConstr, WackyMath],
            trans_from: [IntConstr, FloatConstr],
            trans_to: [IntConstr, FloatConstr],
            trans_from_func: WackyMath = None,
            trans_to_func: WackyMath = None,
    ):
        """
        Initialize
        :param delta_x:
        :param trans_from:
        :param trans_to:
        :param trans_from_func:
        :param trans_to_func:
        """
        super(ValueTransfer, self).__init__()
        self._delta_x = delta_x
        self._trans_from = self._init_contr(trans_from)
        self._trans_to = self._init_contr(trans_to)
        self._trans_from_func = self._init_trans_func(trans_from_func)
        self._trans_to_func = self._init_trans_func(trans_to_func)

        # TODO: implement dtype property for everthing
        if not isinstance(self.trans_from, type(self.trans_to)):
            raise TypeError(f'Expected same types. Got {type(trans_from)} and {type(trans_to)}.')

        if not isinstance(self.trans_from_func, type(self.trans_to_func)):
            raise TypeError(f'Expected same types. Got {type(trans_from_func)} and {type(trans_to_func)}.')

    @staticmethod
    def _init_contr(constr: [IntConstr, FloatConstr]) -> [IntConstr, FloatConstr]:
        if isinstance(constr, (IntConstr, FloatConstr)):
            return constr
        else:
            raise TypeError(f'Expected type: IntConstr, FloatConstr. Got {type(constr)} instead.')

    @staticmethod
    def _init_trans_func(trans_func: [None, FloatConstr]) -> [None, FloatConstr]:
        if trans_func is None:
            return None
        elif isinstance(trans_func, WackyMath):
            return trans_func
        else:
            raise TypeError(f'Expected type: None, WackyMath. Got {type(trans_func)} instead.')

    @property
    def delta_x(self):
        return self._delta_x

    @property
    def trans_from(self):
        return self._trans_from

    @property
    def trans_to(self):
        return self._trans_to

    @property
    def trans_from_func(self):
        return self._trans_from_func

    @property
    def trans_to_func(self):
        return self._trans_to_func

    def __call__(self) -> None:

        x_from = self.delta_x.value

        if self.trans_to_func is not None:
            x_to = abs(self.trans_to_func({'x': x_from}))
        else:
            x_to = x_from

        self.trans_from.delta(-x_from)
        self.trans_to.delta(x_to)

        self.error_signal = (self.trans_from.error_signal or self.trans_to.error_signal)

        if not self.error_signal:
            delta_t = max(
                self.trans_from.to_accept_op_time,
                self.trans_to.to_accept_op_time
            )

            x_from = abs(self.trans_from.to_accept_op_x)
            x_to = abs(self.trans_to.to_accept_op_x)

            if self.trans_from_func is not None:
                x_to = abs(self.trans_from_func({'y': x_to}))

            x_from = min(x_from, x_to)

            if self.trans_to_func is not None:
                x_to = abs(self.trans_to_func({'x': x_from}))
            else:
                x_to = x_from

            self.trans_from.accept(-x_from, delta_t)
            self.trans_to.accept(x_to, delta_t)
