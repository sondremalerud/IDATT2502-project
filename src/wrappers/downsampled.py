from gymnasium import Env

from .filteredenv import FilteredEnv
from .filters.downsample import downsample

class DownsampledEnv(FilteredEnv):
    def __init__(self, env: Env, sample_depth: int):
        super().__init__(env, downsample, sample_depth)
