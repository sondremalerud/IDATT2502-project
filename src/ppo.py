#! python3
import mario_bros_env
from nes_py.wrappers import JoypadSpace
from mario_bros_env.actions import SIMPLE_MOVEMENT, RIGHT_ONLY
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import BaseCallback
from wrappers import GrayscaleEnv, DownsampledEnv, DownscaledEnv
from wrappers.filters import GrayscaleFilters
import sys
import matplotlib.pyplot as plt
import numpy as np


# Use "python ppo.py train" or "python ppo.py run".
# Ctrl + C to save and quit.

env = mario_bros_env.make("SuperMarioBros-v0", render_mode="human")
env = JoypadSpace(env, RIGHT_ONLY)

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

class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.cumulative_reward = 0
        self.rewards = []

    def on_rollout_end(self) -> None:
        self.rewards.append(self.cumulative_reward)
        #print(self.rewards)

    def _on_step(self) -> bool:
        self.cumulative_reward += self.locals["rewards"][0]
        
        if self.locals["dones"][0]:
            print("RIP 💀")
            self.rewards.append(self.cumulative_reward)
            self.cumulative_reward = 0

        return True
    
    def plot_rewards(self):
        """ Plots mean rewards """
        mean_rewards = []

        for t in range(len(self.rewards)):
            mean_rewards.append(np.mean(self.rewards[max(0, t-100):(t+1)]))
        plt.plot(mean_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Mean Reward')
        plt.savefig('rewards.png')



#Load existing model or create new one if there is none
try:
    print("Trying to load stored pre-trained model...")
    model = PPO.load("models/PPO.zip", env)
    print("Loaded! :)")
except BaseException:
    print("Failed to load existing model, creating new")
    model = PPO("MlpPolicy", env, n_steps=256, verbose=True)


# Read command line arguments
try:
    command = sys.argv[1].lower()
except IndexError:
    print('Use command "train" or "run".')
    exit(1)

# Train
if command == "train":
    rewardlogger = RewardLoggerCallback()
    try:
        model = model.learn(total_timesteps=10000 * 100, progress_bar=True, callback=rewardlogger)
        saveModel(model)
    except KeyboardInterrupt:
        saveModel(model)

    rewardlogger.plot_rewards()
    
# Run
elif command == "run":
    state = env.reset()
    while True:
        action, _ = model.predict(state)
        state, reward, done, info = env.step(action)
