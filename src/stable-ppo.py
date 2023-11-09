import mario_bros_env
from nes_py.wrappers import JoypadSpace
from mario_bros_env.actions import SIMPLE_MOVEMENT
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
import wrappers
# tjuvlånt fra https://github.com/BJEnrik/reinforcement-learning-super-mario

env = mario_bros_env.make("SuperMarioBros-v0", render_mode=None)
env = JoypadSpace(env, SIMPLE_MOVEMENT)

env = wrappers.GrayscaleEnv(env)
env = wrappers.DownscaledEnv(env, 16)
env = wrappers.DownsampledEnv(env, 8)

env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, 4, channels_order="last")

model = PPO('CnnPolicy', env)
model.learn(total_timesteps=5)

state = env.reset()

while True:
     action, _ = model.predict(state)
     state, reward, done, info = env.step(action)

