from typing import Any

from dataclasses import dataclass
from wacky_envs import EnvModule
from wacky_envs.steppers import BaseStepper

# TODO: max steps

@dataclass
class FixStepper(BaseStepper):

    def __init__(self, delta_t: Any = None,  max_t: int = None, init_t: int = 0):
        super(FixStepper, self).__init__(delta_t, max_t, init_t)
