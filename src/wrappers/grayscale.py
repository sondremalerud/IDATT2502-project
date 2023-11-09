from gymnasium import Env

from .filteredenv import FilteredEnv
from .filters.grayscale import red_channel

class GrayscaleEnv(FilteredEnv):
    def __init__(self, env: Env):
        super().__init__(env, red_channel)
