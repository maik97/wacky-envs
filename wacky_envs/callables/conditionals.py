from dataclasses import dataclass

from wacky_envs import ValueEnvModule
from wacky_envs.callables import BaseCallable
from wacky_envs.numbers import WackyMath


@dataclass
class Condition(BaseCallable):
    """Calls :attr:`consequence`, if some :attr:`condition` is true."""

    condition: WackyMath
    consequence: [BaseCallable]
    consequence_value: [ValueEnvModule]

    def __init__(
            self,
            condition: WackyMath,
            consequence: [BaseCallable],
            consequence_value: [ValueEnvModule] = None
    ) -> None:
        """
        Initialize
        :param condition:
        :param consequence:
        :param consequence_value:
        """

        super(Condition, self).__init__()

        self._condition = condition
        self._consequence = consequence
        self._consequence_value = consequence_value

    def __call__(self) -> None:

        if self.condition.value:
            if self.consequence_value is not None:
                self.consequence(self.consequence_value.value)
            else:
                self.consequence()

    @property
    def condition(self):
        return self._condition

    @property
    def consequence(self):
        return self._consequence

    @property
    def consequence_value(self):
        return self._consequence_value
