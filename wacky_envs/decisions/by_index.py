from dataclasses import dataclass


@dataclass
class ChoiceByIndex:

    choices: list
    cur_idx: int
    cur_choice: object
    n: int

    def __init__(self, choices: list):
        self.choices = choices

    def set(self, idx):
        self.cur_idx = idx

    def __call__(self):
        self.cur_choice()

    @property
    def cur_choice(self):
        return self.choices[self.cur_idx]

    @property
    def n(self):
        return len(self.choices)