from abc import abstractmethod

from wacky_envs import EnvModule


class BaseObs(EnvModule):
    """Base module for observations."""

    def __init__(self):
        super(BaseObs, self).__init__()

    @abstractmethod
    def __call__(self):
        pass

    @property
    @abstractmethod
    def space(self):
        pass
