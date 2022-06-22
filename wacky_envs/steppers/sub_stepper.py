from dataclasses import dataclass


@dataclass
class SubSteps:

    substep_functions: list

    def __init__(self, substep_functions: list):
        self.substep_functions = substep_functions

    def __call__(self):
        for f in self.substep_functions:
            f()
