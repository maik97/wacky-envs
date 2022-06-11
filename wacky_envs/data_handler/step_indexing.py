from dataclasses import dataclass
from wacky_envs.data_handler import ObjectGroup
from wacky_envs.step_functions import FixStepper

@dataclass
class GatherByStepIndex:

    indexable_data: ObjectGroup
    stepper: FixStepper
    step_obj: object

    def __init__(
            self,
            indexable_data: ObjectGroup,
            stepper: FixStepper,
    ):
        self.indexable_data = indexable_data
        self.stepper = stepper

    @property
    def step_obj(self) -> object:
        return self.indexable_data(idx=self.stepper.t)

