from gymnasium import Env
from gymnasium.spaces import Box

from .filteredenv import FilteredEnv
from .filters.downsampling import downsample


class DownsampledEnv(FilteredEnv):
    def __init__(self, env: Env, sample_depth: int):
        super().__init__(env, downsample, sample_depth)

        self.observation_space = Box(
            low=0,
            high=sample_depth - 1,
            shape=self.observation_space.shape,
            dtype=self.observation_space.dtype,
        )
