from gym import spaces
from wacky_envs.actions import BaseAction

class BoxAction(BaseAction):
    """Continuous action that sets a value to :attr:`set_value_at`."""

    def __init__(self, set_value_at):
        super(BoxAction, self).__init__()
        self.set_value_at = set_value_at

    def __call__(self, action):
        self.set_value_at.set(action)

    @property
    def shape(self):
        return (1,)

    @property
    def space(self):
        return spaces.Box(
            low=self.set_value_at.lowerbound,
            high=self.set_value_at.upperbound,
            shape=self.shape
        )
