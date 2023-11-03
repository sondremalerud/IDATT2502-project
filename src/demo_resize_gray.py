from nes_py.wrappers import JoypadSpace
import mario_bros_env
from mario_bros_env.actions import RIGHT_ONLY
from matplotlib import pyplot as plt

from filters import grayscale, downscale, downsample

env = mario_bros_env.make(
    'SuperMarioBros-v0',
    render_mode=None
)
env = JoypadSpace(env, RIGHT_ONLY)

state, info = env.reset()

for _ in range(200):
    state, _, _, _, _ = env.step(env.action_space.sample())

gray_state = grayscale.red_channel(state)
rescaled_state = downscale.divide(gray_state, 8)
downsampled_state = downsample.downsample(rescaled_state, 4)

fig = plt.figure()

fig.add_subplot(2, 2, 1)
plt.imshow(state)

fig.add_subplot(2, 2, 2)
plt.imshow(gray_state, cmap='Greys_r')

fig.add_subplot(2, 2, 3)
plt.imshow(rescaled_state, cmap="Greys_r")

fig.add_subplot(2, 2, 4)
plt.imshow(downsampled_state, cmap="Greys_r")
plt.show()

env.close()
