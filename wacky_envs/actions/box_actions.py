from gym import spaces


class BoxAction:

    def __init__(self, set_value_at):
        self.set_value_at = set_value_at

    def __call__(self, action):
        self.set_value_at.set(action)

    @property
    def space(self):
        return spaces.Box(
            low=self.set_value_at.lowerbound,
            high=self.set_value_at.upperbound,
            shape=(1,)
        )
