from nes_py.wrappers import JoypadSpace
import mario_bros_env
from mario_bros_env.actions import RIGHT_ONLY
from matplotlib import pyplot as plt

from wrappers import DownsampledEnv, DownscaledEnv, GrayscaleEnv
from wrappers.filters import GrayscaleFilters

env = mario_bros_env.make("SuperMarioBros-v0", render_mode=None)
env = JoypadSpace(env, RIGHT_ONLY)
env = GrayscaleEnv(env, GrayscaleFilters.RED)
env = DownscaledEnv(env, 16)
env = DownsampledEnv(env, 3)

state, info = env.reset()

for _ in range(200):
    state, _, _, _, _ = env.step(env.action_space.sample())

fig = plt.figure()

plt.imshow(state, cmap="Greys_r")
plt.show()

env.close()
