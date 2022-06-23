from abc import abstractmethod

from wacky_envs import EnvModule


class BaseCallable(EnvModule):
    """Base module for callables."""

    def __init__(self):
        """Initialize."""
        super(BaseCallable, self).__init__()

    @abstractmethod
    def __call__(self):
        pass

    def step(self, _, __, ___):
        self.__call__()

    def reset(self):
        pass
