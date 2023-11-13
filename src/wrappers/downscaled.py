from gymnasium import Env
from gymnasium.spaces import Box

from .filteredenv import FilteredEnv
from .filters.downscale import divide

class DownscaledEnv(FilteredEnv):
    def __init__(self, env: Env, scale_factor: int):
        super().__init__(env, divide, scale_factor)

        self.observation_space = Box(
            low=0,
            high=255,
            shape=(
                self.env.observation_space.shape[0] // scale_factor,
                self.env.observation_space.shape[1] // scale_factor,
                1
            ),
            dtype=self.env.observation_space.dtype
        )
