from typing import Any, Callable
from gymnasium import Wrapper
from .filters.types import Image


class FilteredEnv(Wrapper):
    def __init__(self, env, filter: Callable[[Image, Any], Image], *filterargs):
        super().__init__(env)
        self.filter = filter
        self.filterargs = filterargs

    def step(self, action):
        screen, reward, self.terminated, self.truncated, info = self.env.step(action)

        return self.filter(
            screen, *self.filterargs
        ), reward, self.terminated, self.truncated, info

    def reset(self, seed=None, options=None):
        screen, info = super().reset(seed, options)
        return self.filter(screen, *self.filterargs), info
