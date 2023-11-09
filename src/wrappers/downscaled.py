from gymnasium import Env
from gymnasium.spaces import Box

from .filteredenv import FilteredEnv
from .filters.downscale import divide

class DownscaledEnv(FilteredEnv):
    def __init__(self, env: Env, scale_factor: int):
        super().__init__(env, divide, scale_factor)

        self.observation_space = Box(
            low=self.observation_space.low,
            high=self.observation_space.high,
            shape=(
                self.observation_space.shape[0] // scale_factor,
                self.observation_space.shape[1] // scale_factor,
                1
            ),
            dtype=self.observation_space.dtype
        )
