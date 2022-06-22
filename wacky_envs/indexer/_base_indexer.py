from wacky_envs.env_module import ValueEnvModule


class BaseIndexer(ValueEnvModule):

    def __init__(self):
        super(BaseIndexer, self).__init__()

    def reset(self):
        pass
