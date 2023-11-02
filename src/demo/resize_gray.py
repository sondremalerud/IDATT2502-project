from nes_py.wrappers import JoypadSpace
import mario_bros_env
from mario_bros_env.actions import RIGHT_ONLY
from matplotlib import pyplot as plt

from filters import grayscale, downscale

env = mario_bros_env.make(
    'SuperMarioBros-v0',
    render_mode=None
)
env = JoypadSpace(env, RIGHT_ONLY)

state, info = env.reset()

for _ in range(150):
    state, _, _, _, _ = env.step(env.action_space.sample())

gray_state = grayscale.vxycc601(state)

rescaled_state = downscale.divide(gray_state, 9)
fig = plt.figure()

fig.add_subplot(1, 3, 1)
plt.imshow(state)

fig.add_subplot(1, 3, 2)
plt.imshow(gray_state, cmap='Greys_r')

fig.add_subplot(1, 3, 3)
plt.imshow(rescaled_state, cmap="Greys_r")
plt.show()

env.close()
