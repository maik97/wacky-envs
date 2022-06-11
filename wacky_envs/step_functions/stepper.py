from dataclasses import dataclass


@dataclass
class FixStepper:

    t: int
    delta_t: float

    def __init__(self, delta_t: float):
        self.delta_t = delta_t

    def __call__(self):
        self.t += 1

    def reset(self):
        self.t = 0
