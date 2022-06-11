from gym import spaces
import numpy as np

class BoxObs:

    def __init__(self, value_list: list):
        self.value_list = value_list

    def __call__(self):
        return np.array([v.value for v in self.value_list])

    def n_values(self):
        return len(self.value_list)

    @property
    def space(self):
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_values,)
        )
