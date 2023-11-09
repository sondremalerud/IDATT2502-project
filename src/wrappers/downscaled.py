from gymnasium import Env

from .filteredenv import FilteredEnv
from .filters.downscale import divide

class DownscaledEnv(FilteredEnv):
    def __init__(self, env: Env, scale_factor: int):
        super().__init__(env, divide, scale_factor)
