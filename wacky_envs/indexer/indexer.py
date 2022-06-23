from wacky_envs import ValueEnvModule
from wacky_envs.indexer import BaseIndexer


class ByIndex(BaseIndexer):
    """Assigns a :attr:`cur_choice` of the iterable :attr:`choices` by calling the current value of :attr:`indexer`."""

    choices: list
    cur_idx: int
    cur_choice: object
    n: int

    def __init__(self, choices, indexer):
        super(ByIndex, self).__init__()
        self.choices = choices
        self.indexer = indexer

    def reset(self):
        pass

    def set(self, indexer):
        self.indexer = indexer

    @property
    def cur_idx(self):
        return self.indexer.value

    @property
    def value(self):
        if isinstance(self.cur_choice, ValueEnvModule):
            return self.choices[self.indexer.value].value
        else:
            return self.choices[self.indexer.value]

    @property
    def cur_choice(self):
        return self.choices[self.indexer.value]

    @property
    def n(self):
        return len(self.choices)

    def step(self, delta_t, t, episode_delta_t) -> None:
        if hasattr(self.cur_choice, 'step'):
            self.cur_choice.step(delta_t, t, episode_delta_t)
