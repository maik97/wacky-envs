from abc import ABC
from typing import Any
import itertools

from dataclasses import dataclass, asdict


@dataclass
class EnvModule(ABC):
    """Parent class for all environment modules."""
    name: str
    id: int
    classname: str
    newid = itertools.count()
    """Counting the number of environment moduls for assigning unique ids."""

    def __init__(self, name: str = None):
        """Assigns name and unique id"""
        self._id = next(EnvModule.newid)
        if not hasattr(self, '_name'):
            self._name = name
        print(f"{self.classname}(id={self.id})")

    @property
    def classname(self):
        """Classname of instance"""
        return self.__class__.__name__

    @property
    def name(self):
        """Either user assigned name or None"""
        return self._name

    @property
    def id(self):
        """Unique id of instance by counting all environment modules."""
        return self._id

    @property
    def token_dict(self):
        """Infos for token construction"""
        return {
            'id': self.id,
            'module_type': self.__class__.__name__,
        }

    @property
    def watch_dict(self):
        """Watch instance for debugging"""
        watch_dict = asdict(self)
        for k, v in watch_dict.items():
            if isinstance(v, dict):
                v_keys = list(v.keys())
                if 'value' in v_keys and 'name' in v_keys and 'id' in v_keys:
                    watch_dict[k] = f"{v['classname']}(name={v['name']}, id={v['id']}, value={v['value']})"
                elif 'name' in v_keys and 'id' in v_keys:
                    watch_dict[k] = f"{v['classname']}(name={v['name']}, id={v['id']})"
        return watch_dict


@dataclass
class ValueEnvModule(EnvModule):
    """
    Base Module for values

    Note:
        Attributes :attr:`value` and :attr:`prev_value` are `None`, if :attr:`init_value` is `None`.
    """

    value: Any
    init_value: Any
    prev_value: Any
    delta_value: Any
    dtype: type

    @property
    def dtype(self) -> type:
        """Datatype for `value`, `init_value`, `prev_value`, `delta_value`, ect."""
        return Any

    def __init__(self, init_value: Any = None):
        r"""
        Sets :attr:`init_value` and calls :func:`reset`.

        Note: Attributes :attr:`prev_value` and :attr:`value` are `None`, if :attr:`init_value` is `None`.

        :param init_value: The initial value. Resets to this value at the start of each episode.
        """
        super(ValueEnvModule, self).__init__()
        if init_value is not None:
            self.set_init(init_value)
            self.reset()
        else:
            self._init_value = None
            self._prev_value = None
            self._value = None

    def __call__(self, *args, **kwargs) -> Any:
        """Alternative for getting the current value. Might be useful for some properties in
        :class:`wacky_envs.EnvModule`. (Example: See rewards for the peak-shaver)"""
        return self.value

    def set_init(self, init_value: Any) -> None:
        """
        Updates `init_value` after checking if the value is the right datatype.

        :param init_value: Set new `init_value`
        :return: None
        """
        if not isinstance(init_value, self.dtype):
            raise TypeError(f"Expected type {self.dtype}, got {type(init_value)} instead")
        self._init_value = init_value

    def set(self, value: Any) -> None:
        """
        Updates `value` and `prev_value` after checking if the value is the right datatype.

        :param value: Set new value
        :return: None
        """
        if not isinstance(value, self.dtype):
            raise TypeError(f"Expected type {self.dtype}, got {type(value)} instead")
        self._prev_value = self.value
        self._value = value

    def reset(self) -> None:
        """Assigns `init_value` to value and `prev_value`."""
        self._prev_value = self.init_value
        self._value = self.init_value

    @property
    def value(self) -> Any:
        """Current value"""
        return self._value

    @property
    def init_value(self) -> Any:
        """Used for resetting the current value to the initial value."""
        return self._init_value

    @property
    def prev_value(self) -> Any:
        """Value from before the :func:`wacky_envs.EnvModule.set()` method was called."""
        return self._prev_value

    @property
    def delta_value(self) -> Any:
        """
        Difference between new value and previous value
        after calling the :func:`wacky_envs.EnvModule.set()` method.
        """
        if self.value is None or self.prev_value is None:
            return None
        return self.value - self.prev_value

    def step(self, _, __, ___) -> None:
        """
        Placeholder method, which might be called in :func:`wacky_envs.WackyEnv.step()`

        :param _: Placeholder for the parameter `t` (current step)
        :param __: Placeholder for the parameter `delta_t` (current timeframe)
        :param ___: Placeholder for the parameter `episode_delta_t` (current episode timeframe)
        :return: None
        """
        pass

    def take_step(self, _, __, ___, ____):
        """
        Placeholder method, which might be called :func:`wacky_envs.EnvModule.step()` in another instance.

        :param _: Placeholder for the parameter `value` (current value of other instance)
        :param __: Placeholder for the parameter `t` (current step)
        :param ___: Placeholder for the parameter `delta_t` (current timeframe)
        :param ____: Placeholder for the parameter `episode_delta_t` (current episode timeframe)
        :return:
        """
        return self.value

    def act(self, input):
        """Placeholder method, which might be called by :class:`wacky_envs.actions.BaseAction`"""
        raise AttributeError(f"{self.__class__.__name__} has no method 'act'.")

    @property
    def obs(self):
        """Current value"""
        return self.value

    @property
    def token_dict(self):
        """Infos for token construction"""
        return {
            'id': self.id,
            'module_type': self.__class__.__name__,
            'dtype': self.dtype,
            'value': self.value,
            'init_value': self.init_value,
            'prev_value': self.prev_value,
            'delta_value': self.delta_value,
        }

    @property
    def space(self):
        """Placeholder attribute, which might be used by :class:`wacky_envs.actions.BaseAction`"""
        raise AttributeError(f"{self.__class__.__name__} has no property 'space'.")
