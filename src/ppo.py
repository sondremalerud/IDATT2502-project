#! python3
import mario_bros_env
from nes_py.wrappers import JoypadSpace
from mario_bros_env.actions import SIMPLE_MOVEMENT
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from wrappers import GrayscaleEnv, DownsampledEnv, DownscaledEnv
from wrappers.filters import GrayscaleFilters
import sys
from datetime import datetime
import os


# Use "python ppo.py train" or "python ppo.py run".
# Ctrl + C to save and quit.

env = mario_bros_env.make("SuperMarioBros-v0", render_mode="human")
env = JoypadSpace(env, SIMPLE_MOVEMENT)

# Process frames
env = GrayscaleEnv(env, GrayscaleFilters.RED)
env = DownscaledEnv(env, 12)
env = DownsampledEnv(env, 4)

# Frame stacking
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, 8, channels_order="last")

model: PPO

# Save model to zip file
def saveModel(model):
     print("Saving and quitting...")
     model.save("models/PPO.zip")
     print("Saved :)")
     exit(1)


# Load existing model or create new one if there is none
try:
     print("Trying to load stored pre-trained model...")
     model = PPO.load("models/PPO.zip", env)
     print("Loaded! :)")
     exit()
except BaseException:
     print("Failed to load existing model, creating new")
     model = PPO("MlpPolicy", env, verbose=True)


# Read command line arguments
try:
     command = sys.argv[1].lower()
except IndexError:
     print("Use command \"train\" or \"run\".")
     exit(1)

# Train
if command == "train":
     try:
          model = model.learn(total_timesteps=1024 * 1024, progress_bar=True)
          saveModel(model)
     except KeyboardInterrupt:
          saveModel(model)
# Run
elif command == "run":
     state = env.reset()
     while True:
          action, _ = model.predict(state)
          state, reward, done, info = env.step(action)
     



