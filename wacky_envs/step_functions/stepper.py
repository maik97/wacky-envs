from dataclasses import dataclass


@dataclass
class FixStepper:

    t: int
    delta_t: float

    def __init__(self, delta_t: float):
        self.delta_t = delta_t

    def next(self):
        self.t += 1

    def reset(self):
        self.t = 0

    @property
    def value(self):
        return self.t
