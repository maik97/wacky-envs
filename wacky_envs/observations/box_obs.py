from gym import spaces
import numpy as np


class BoxObs:

    def __init__(self, value_list: list):
        self.value_list = value_list

    def __call__(self):
        obs = np.array([])
        for v in self.value_list:
            obs = np.append(obs, v.value)
        #print(obs)
        return obs

    @property
    def n_values(self):
        return len(self.__call__())

    @property
    def space(self):
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_values,)
        )
