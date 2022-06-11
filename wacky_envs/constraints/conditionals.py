from dataclasses import dataclass
from wacky_envs.constraints import (
    Allocations, ValueTransfer, FloatConstr, WackyFloat, IntConstr, WackyInt, WackyMath
)


@dataclass
class Condition:

    condition: WackyMath
    consequence: [ValueTransfer, Allocations]
    consequence_value: [FloatConstr, IntConstr, WackyFloat, WackyInt, WackyMath]

    def __init__(
            self,
            condition: WackyMath,
            consequence: [ValueTransfer, Allocations],
            consequence_value: [FloatConstr, IntConstr, WackyFloat, WackyInt, WackyMath] = None
    ) -> None:

        self.condition = condition
        self.consequence = consequence
        self.consequence_value = consequence_value

    def __call__(self) -> None:

        if self.condition.value:
            if self.consequence_value is not None:
                self.consequence(self.consequence_value.value)
            else:
                self.consequence()
