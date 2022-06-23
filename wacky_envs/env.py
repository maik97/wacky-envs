import gym
import numpy as np
from gym import spaces
from typing import Any

from wacky_envs.actions import BaseAction
from wacky_envs.observations import BaseObs
from wacky_envs.steppers import BaseStepper
from wacky_envs.numbers import WackyNumber
from wacky_envs.callables import BaseCallable
from wacky_envs import EnvModule, ValueEnvModule


class WackyEnv(gym.Env):
    """
    This class is the skeleton for building environments. It follows the OpenAI Gym environments template,
    since :class:`gym.Env` is subclassed. Most Reinforcing Learning libraries are compatible with Gym environments.
    It's important to keep in mind how the methods :func:`WackyEnv.reset` and :func:`WackyEnv.step` work.

    To initialize a new episode, the agent will call the :func:`self.reset` method of the environment. Here,
    the :func:`BaseStepper.reset` method of the stepper is called first. Afterwards, all :func:`EnvModule.reset`
    methods are called according to the order of the list :attr:`self.reset_modules`. The whole call order is:

     - :func:`BaseStepper.reset`
     - List[:func:`EnvModule.reset`]

    .. code-block:: python

        def reset(self) -> np.ndarray:

            self._stepper.reset()
            if self.reset_modules is not None:
                for var in self.reset_modules:
                    var.reset()
            return self.observation

    When the Reinforcement Learning Agent decided on an action, the :func:`self.step` method of the environment is called.
    Here, the action module :attr:`self._action` will assign the action to some value of the type :class:`WackyNumber`.
    This is also the case, if the action is an index for something else (e.g., for a callable :class:`BaseCallable`).
    Next, all :func:`EnvModule.step` methods are called according to the order of the list :attr:`self.step_modules`.
    Then the stepper counts the next step and adds the step timeframe to the total episode timeframe by calling
    :func:`_stepper.next`. Finally, the returns :attr:`observation`, :attr:`reward`, :attr:`done` and :attr:`info`
    are called. Keep in mind that these attributes are properties. The whole call order is:

    - :func:`BaseAction.__call__`
    - List[:func:`EnvModule.step`]
    - :attr:`observation` -> :func:`BaseObs.__call__`
    - :attr:`rewards` -> :func:`WackyNumber.__call__` or :func:`WackyMath.__call__`
    - :attr:`done` -> :func:`WackyNumber.__call__` or :func:`WackyMath.__call__`
    - :attr:`info` -> None

    .. code-block:: python

        def step(self, action) -> tuple:

            self._action(action)

            if self.step_modules is not None:
                for var in self.step_modules:
                    try:
                        var.step(self.t, self.delta_t, self.episode_delta_t)
                    except Exception as e:
                        print(e)
                        print(var)
                        exit()

            self._stepper.next()
            self._terminator.step(self.t, self.delta_t, self.episode_delta_t)
            done = self.done
            return self.observation, self.reward, done, self.info

    """

    def __init__(
            self,
            stepper: BaseStepper,
            observation: BaseObs,
            action: BaseAction,
            reward: ValueEnvModule,
            terminator: ValueEnvModule,
            reset_modules: list = None,
            call_modules: list = None,
            step_modules: list = None,
            watch_modules: list = None,
    ):
        """
        Assigns all environment modules to their attributes.

        :param stepper:
            Module to count episode steps and total steps. Traces step and episode timeframes.
        :type stepper: :class:`wacky_envs.stepper.BaseStepper`

        :param observation:
            Module to observe the current state.
        :type observation: :class:`wacky_envs.observations.BaseObs`

        :param action:
            Module to assign the agents current actions.
        :type action: :class:`wacky_envs.actions.BaseAction`

        :param reward:
            Module to calculate the current rewards.
        :type reward: :class:`wacky_envs.ValueEnvModule`

        :param terminator:
            Module to control episode termination.
        :type terminator: :class:`wacky_envs.ValueEnvModule`

        :param reset_modules:
            List of modules that must reset for a new episode. Order of the list matters.
        :type reset_modules: List[:class:`wacky_envs.EnvModule`]

        :param call_modules:
            List of modules that are called after action is assigned and before steps are taken. Order of the list matters.
        :type call_modules: List[:class:`wacky_envs.callables.BaseCallable`]

        :param step_modules:
            List of modules that each call :func:`EnvModule.step` after an action is assigned. Order of the list matters.
        :type step_modules: List[:class:`wacky_envs.EnvModule`]

        :param watch_modules:
            List of modules flagged for debugging.
        :type watch_modules: List[:class:`wacky_envs.EnvModule`]
        """
        self._stepper = stepper
        self._obs = observation
        self._action = action
        self._reward = reward
        self._terminator = terminator
        self.reset_modules = reset_modules
        self.call_modules = call_modules
        self.step_modules = step_modules
        self.watch_modules = watch_modules

    @property
    def observation_space(self) -> spaces.Space:
        """

        :return: Gym space for the observation
        :rtype: spaces.Space
        """
        return self._obs.space

    @property
    def action_space(self) -> spaces.Space:
        """

        :return: Gym space for the action
        :rtype: spaces.Space
        """
        return self._action.space

    @property
    def observation(self) -> np.ndarray:
        """

        :return: Current observation of the environment state
        :rtype: np.ndarray
        """
        return self._obs()

    @property
    def reward(self) -> Any:
        """

        :return: Current step reward
        :rtype: Any
        """
        return self._reward()

    @property
    def done(self) -> bool:
        """

        :return: True, if the episode terminates
        :rtype: bool
        """
        return self._terminator() or self._stepper.done

    @property
    def info(self) -> dict:
        """
        Nothing implemented yet (Important later, e.g., useful for action masking).

        :return: Empty dict
        :rtype: dict
        """
        return {}

    @property
    def delta_t(self) -> float:
        """

        :return: Current step timeframe
        :rtype: float
        """
        return self._stepper.delta_t

    @property
    def episode_delta_t(self) -> float:
        """

        :return: Current episode timeframe (so far)
        :rtype: float
        """
        return self._stepper.episode_delta_t

    @property
    def t(self) -> int:
        """

        :return: Current episode step count
        :rtype: int
        """
        return self._stepper.t

    def step(self, action) -> tuple:
        """
        This is where the agents decisions are acted out. See class description above for more details.

        :param action:
        :return: Tuple of observation, reward, done and info
        :rtype: tuple
        """

        self._action(action)

        if self.call_modules is not None:
            for module in self.call_modules:
                try:
                    module()
                except Exception as e:
                    print(e)
                    print(module)
                    exit()

        if self.step_modules is not None:
            for module in self.step_modules:
                try:
                    module.step(self.t, self.delta_t, self.episode_delta_t)
                except Exception as e:
                    print(e)
                    print(module.watch_dict)
                    exit()

        if self.watch_modules is not None:
            for module in self.watch_modules:
                print(module.watch_dict)

        self._stepper.next()
        self._terminator.step(self.t, self.delta_t, self.episode_delta_t)
        #self._reward.step(self.t, self.delta_t, self.episode_delta_t)
        return self.observation, self.reward, self.done, self.info

    def reset(self) -> np.ndarray:
        """
        The environment resets if a new episode starts. See class description above for more details.

        :return: Current Observations
        :rtype: np.ndarray
        """
        self._stepper.reset()
        if self.reset_modules is not None:
            for module in self.reset_modules:
                module.reset()
        return self.observation

    def render(self, mode='human'):
        """Not implemented yet."""
        pass
