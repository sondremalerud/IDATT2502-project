from nes_py.wrappers import JoypadSpace
import mario_bros_env
from mario_bros_env.actions import RIGHT_ONLY
env = mario_bros_env.make(
    'SuperMarioBros-v0',
    render_mode="human"
)
env = JoypadSpace(env, RIGHT_ONLY)


done = True
for step in range(5000):
    if done:
        state = env.reset()
    state, reward, done, _, info = env.step(env.action_space.sample())
    env.render()

env.close()
