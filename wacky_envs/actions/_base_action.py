from abc import abstractmethod

from wacky_envs import EnvModule

class BaseAction(EnvModule):

    def __init__(self):
        super(BaseAction, self).__init__()

    @abstractmethod
    def __call__(self, action):
        pass

    @property
    @abstractmethod
    def space(self):
        pass
