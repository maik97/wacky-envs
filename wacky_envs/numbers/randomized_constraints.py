import random

from wacky_envs.numbers import FloatConstr, IntConstr

# TODO: in seperate module (randomizer)


class EpisodeRandIntConstr(IntConstr):
    """Randomizes init value when resetting.

    Note:
        The attributes :attr:`lowerbound` and :attr:`upperbound` must be set.
    """

    def reset(self) -> None:
        self.set_init(random.randint(self.lowerbound, self.upperbound))
        super(EpisodeRandIntConstr, self).reset()


class OnCallRandIntConstr(IntConstr):

    """Randomizes current value when called.

    Note:
        The attributes :attr:`lowerbound` and :attr:`upperbound` must be set.
    """

    def __call__(self, *args, **kwargs):
        self.set(random.randint(self.lowerbound, self.upperbound))
        return super(OnCallRandIntConstr, self).__call__()


class EpisodeRandFloatConstr(FloatConstr):
    """Randomizes init value when resetting.

    Note:
        The attributes :attr:`lowerbound` and :attr:`upperbound` must be set.
    """

    def reset(self) -> None:
        self.set_init(random.uniform(self.lowerbound, self.upperbound))
        super(EpisodeRandFloatConstr, self).reset()


class OnCallRandFloatConstr(FloatConstr):
    """Randomizes current value when called.

    Note:
        The attributes :attr:`lowerbound` and :attr:`upperbound` must be set.
    """

    def __call__(self, *args, **kwargs):
        self.set(random.uniform(self.lowerbound, self.upperbound))
        return super(OnCallRandFloatConstr, self).__call__()
