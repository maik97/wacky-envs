import gym
from gym import spaces


class WackyEnv(gym.Env):

    metadata = {
        'render.modes': ['human'],
    }

    def __init__(
            self,
            stepper,
            observation,
            action,
            reward,
            terminator,
            reset_vars: list = None,
            step_vars: list = None,
    ):
        self._stepper = stepper
        self._obs = observation
        self._action = action
        self._reward = reward
        self._terminator = terminator
        self.reset_vars = reset_vars
        self.step_vars = step_vars

    @property
    def observation_space(self):
        return self._obs.space()

    @property
    def action_space(self):
        return self._action.space()

    @property
    def observation(self):
        return self._obs()

    @property
    def reward(self):
        return self._reward()

    @property
    def done(self):
        return self._terminator()

    @property
    def info(self):
        return {}

    @property
    def delta_t(self):
        return self._stepper.delta_t

    @property
    def t(self):
        return self._stepper.t

    def step(self, action):
        self._action(action)
        if self.step_vars is not None:
            for var in self.step_vars:
                var.step(self.t, self.delta_t)
        self._terminator.step(self.t, self.delta_t)
        return self.observation, self.reward, self.done, self.info

    def reset(self):
        self._stepper.reset()
        if self.reset_vars is not None:
            for var in self.reset_vars:
                var.reset()
        return self.observation

    def render(self, mode='human'):
        pass
