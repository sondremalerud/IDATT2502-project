from nes_py.wrappers import JoypadSpace
import mario_bros_env
from mario_bros_env.actions import RIGHT_ONLY
from matplotlib import pyplot as plt

import wrappers.filters

env = mario_bros_env.make("SuperMarioBros-v0", render_mode=None)
env = JoypadSpace(env, RIGHT_ONLY)

state, info = env.reset()

for _ in range(500):
    state, _, _, _, _ = env.step(env.action_space.sample())

fig = plt.figure()

plt.subplot(2, 2, 1)
plt.title("Original frame")
plt.axis('off')
plt.imshow(state, cmap="Greys_r")

state = wrappers.filters.red_channel(state)

plt.subplot(2, 2, 2)
plt.title("Red channel only")
plt.axis('off')
plt.imshow(state, cmap="Greys_r")

state = wrappers.filters.downscale.divide(state, 12)

plt.subplot(2, 2, 3)
plt.title("Downscaled")
plt.axis('off')
plt.imshow(state, cmap="Greys_r")

state = wrappers.filters.downsample(state, 6)

plt.subplot(2, 2, 4)
plt.title("Downsampled")
plt.axis('off')
plt.imshow(state, cmap="Greys_r")

plt.show()

env.close()
