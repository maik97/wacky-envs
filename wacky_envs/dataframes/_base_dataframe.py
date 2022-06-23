from abc import abstractmethod

from wacky_envs import ValueEnvModule


class BaseDataframe(ValueEnvModule):
    """Base module for dataframes."""

    def __init__(self, df, dtype=float):
        super(ValueEnvModule, self).__init__()
        self._df = df
        self._dtype = dtype

    @property
    def dtype(self) -> type:
        return self._dtype

    @property
    def df(self):
        return self._df

    @property
    @abstractmethod
    def value(self):
        pass


class StepperDataframe(BaseDataframe):
    """Base module for dataframes indexed by step count or timeframe."""

    def __init__(self, df, dtype=float):
        super(StepperDataframe, self).__init__(df, dtype)

    @abstractmethod
    def step(self, _, __, ___):
        pass

    @abstractmethod
    def take_step(self, _, __, ___, ____):
        pass
