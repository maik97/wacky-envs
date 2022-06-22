from dataclasses import dataclass

from wacky_envs import ValueEnvModule
from wacky_envs.callables import BaseCallable
from wacky_envs.numbers import WackyMath


@dataclass
class Condition(BaseCallable):

    condition: WackyMath
    consequence: [BaseCallable]
    consequence_value: [ValueEnvModule]

    def __init__(
            self,
            condition: WackyMath,
            consequence: [BaseCallable],
            consequence_value: [ValueEnvModule] = None
    ) -> None:

        super(Condition, self).__init__()

        self.condition = condition
        self.consequence = consequence
        self.consequence_value = consequence_value

    def __call__(self) -> None:

        if self.condition.value:
            if self.consequence_value is not None:
                self.consequence(self.consequence_value.value)
            else:
                self.consequence()
