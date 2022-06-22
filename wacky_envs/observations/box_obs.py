from gym import spaces
import numpy as np

from wacky_envs.observations import BaseObs


class BoxObs(BaseObs):

    def __init__(self, value_list: list):
        super(BoxObs, self).__init__()
        self.value_list = value_list

    def __call__(self):
        obs = np.array([])
        for v in self.value_list:
            obs = np.append(obs, v.value)
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
