from wacky_envs.env_module import ValueEnvModule


class BaseIndexer(ValueEnvModule):
    """Base module for indexer."""

    def __init__(self):
        super(BaseIndexer, self).__init__()

    def reset(self):
        pass
