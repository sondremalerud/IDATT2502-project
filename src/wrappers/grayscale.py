from gymnasium import Env
from gymnasium.spaces import Box

from .filteredenv import FilteredEnv
from .filters import GrayscaleFilters

class GrayscaleEnv(FilteredEnv):
    def __init__(self, env: Env, filter: GrayscaleFilters):
        super().__init__(env, filter)

        self.observation_space = Box(
            low=0,
            high=255,
            shape=(
                self.observation_space.shape[0],
                self.observation_space.shape[1],
                1
            ),
            dtype=self.observation_space.dtype
        )
