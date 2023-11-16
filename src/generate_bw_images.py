from nes_py.wrappers import JoypadSpace
import mario_bros_env
from mario_bros_env.actions import RIGHT_ONLY
from matplotlib import pyplot as plt

from wrappers.filters import GrayscaleFilters

env = mario_bros_env.make("SuperMarioBros-v0", render_mode=None)
env = JoypadSpace(env, RIGHT_ONLY)

state, info = env.reset()

for _ in range(150):
    state, _, _, _, _ = env.step(env.action_space.sample())

fig = plt.figure()

plt.axis('off')
plt.imshow(state)
plt.savefig("./report/images/bw_comparison/original.png", bbox_inches="tight")

gray_state = GrayscaleFilters.AVERAGE(state)
plt.imshow(gray_state, cmap="Greys_r")
plt.savefig("./report/images/bw_comparison/avg.png", bbox_inches="tight")

gray_state = GrayscaleFilters.YCC601(state)
plt.imshow(gray_state, cmap="Greys_r")
plt.savefig("./report/images/bw_comparison/ycc601.png", bbox_inches="tight")

gray_state = GrayscaleFilters.YCC709(state)
plt.imshow(gray_state, cmap="Greys_r")
plt.savefig("./report/images/bw_comparison/ycc709.png", bbox_inches="tight")

gray_state = GrayscaleFilters.RED(state)
plt.imshow(gray_state, cmap="Greys_r")
plt.savefig("./report/images/bw_comparison/red_channel.png", bbox_inches="tight")

gray_state = GrayscaleFilters.GREEN(state)
plt.imshow(gray_state, cmap="Greys_r")
plt.savefig("./report/images/bw_comparison/green_channel.png", bbox_inches="tight")

gray_state = GrayscaleFilters.BLUE(state)
plt.imshow(gray_state, cmap="Greys_r")
plt.savefig("./report/images/bw_comparison/blue_channel.png", bbox_inches="tight")

env.close()
