import mario_bros_env
from nes_py.wrappers import JoypadSpace
from mario_bros_env.actions import SIMPLE_MOVEMENT
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from wrappers import GrayscaleEnv, DownsampledEnv, DownscaledEnv
from wrappers.filters import GrayscaleFilters
# tjuvlånt fra https://github.com/BJEnrik/reinforcement-learning-super-mario

env = mario_bros_env.make("SuperMarioBros-v0", render_mode="human")
env = JoypadSpace(env, SIMPLE_MOVEMENT)

env = GrayscaleEnv(env, GrayscaleFilters.RED)
env = DownscaledEnv(env, 12)
env = DownsampledEnv(env, 4)

env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, 8, channels_order="last")

model: PPO

try:
     print("Trying to load stored pre-trained model...")
     model = PPO.load("models/PPO.zip", env)
     print("Loaded! :)")
except BaseException:
     print("Failed to load existing model, creating new")
     model = PPO("MlpPolicy", env, verbose=True)

try:
     model = model.learn(total_timesteps=1024 * 1024, progress_bar=True)
except KeyboardInterrupt:
     print("Saving and quitting...")
     model.save("models/PPO.zip")
     print("Saved :)")
     exit(1)

state = env.reset()

while True:
    action, _ = model.predict(state)
    state, reward, done, info = env.step(action)
