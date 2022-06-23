from gym import spaces
import numpy as np
from wacky_envs.numbers import FloatConstr
from wacky_envs.numbers import IntConstr
from wacky_envs.actions import BaseAction


class DiscreteAction(BaseAction):
    """Discrete action that sets a value to :attr:`changeable_decision`."""

    def __init__(self, changeable_decision):
        """Test"""
        super(DiscreteAction, self).__init__()
        self.changeable_decision = changeable_decision

    def __call__(self, action):
        self.changeable_decision.set(action)

    @property
    def n(self):
        return self.changeable_decision.n

    @property
    def space(self):
        return spaces.Discrete(self.changeable_decision.n)


class AtomizedAction(BaseAction):
    """Discrete action for an atomized continuous variable."""

    def __init__(
            self,
            changeable_value: [IntConstr, FloatConstr],
            n_atoms: int = None,
            step_size: [int, float] = None,
            suppress_int_exception=False,
    ):
        super(AtomizedAction, self).__init__()

        if n_atoms is None and step_size is None:
            raise AttributeError('Both n_atoms and step_size are None.')

        if n_atoms is not None and step_size is not None:
            raise AttributeError('Either n_atoms or step_size must be None.')

        if changeable_value.lowerbound is None:
            raise AttributeError('Lower bound not set.')

        if changeable_value.upperbound is None:
            raise AttributeError('Upper bound not set.')

        start = changeable_value.lowerbound
        stop = changeable_value.upperbound

        if n_atoms is None:
            n_atoms = (stop - start) / step_size

            if not n_atoms.is_integer() and not suppress_int_exception:
                raise Exception('(stop - start) / step_size must be int')

            elif not n_atoms.is_integer():
                n_atoms = int(n_atoms)

        self.n_atoms = n_atoms
        self.changeable_value = changeable_value
        self.support = np.linspace(start, stop, num=n_atoms)

    def __call__(self, action):
        self.changeable_value.set(self.support[action])

    @property
    def n(self):
        return self.n_atoms

    @property
    def space(self):
        return spaces.Discrete(self.n_atoms)


class DiscreteSinglesToMulti(BaseAction):
    """Combines discrete actions into a multi-discrete instance."""

    def __init__(self, single_discretes: list):
        super(DiscreteSinglesToMulti, self).__init__()
        self.single_discretes = single_discretes

    def __call__(self, action):
        for act, d_space in zip(action, self.single_discretes):
            d_space(act)

    @property
    def nvec(self):
        return np.array([d_space.n for d_space in self.single_discretes])

    @property
    def space(self):
        return spaces.MultiDiscrete(self.nvec)
