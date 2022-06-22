from abc import abstractmethod

from wacky_envs import EnvModule


class BaseCallable(EnvModule):

    def __init__(self):
        super(BaseCallable, self).__init__()

    @abstractmethod
    def __call__(self):
        pass

    def step(self, _, __, ___):
        self.__call__()

    def reset(self):
        pass
