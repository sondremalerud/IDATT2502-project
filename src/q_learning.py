from mario_bros_env import actions
import numpy as np
from nes_py.wrappers import JoypadSpace
import mario_bros_env
from mario_bros_env.actions import RIGHT_ONLY
from filters import grayscale, downscale, downsample
import signal
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple, deque
import random

# A named tuple representing a single transition in our environment. 
# It maps (state, action) pairs to their (next_state, reward) result
#https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

#https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class Model:
    def __init__(self, env):
        super(Model, self).__init__()
        # 4 stacked frames as input (which will be one state)
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        self.layer1 = nn.Linear(64*5*5,512)
        self.layer2 = nn.Linear(512, self.env.action_space.n)

        self.optimizer = torch.optim.Adam()

        self.image_dimentions = (30,32)
        self.num_colors = 4
        
        self.env = env
        self.actions = actions.RIGHT_ONLY

        self.lr = 0.2
        self.exp_rate = 1.0
        self.decay_rate = 0.995
        self.min_exp_rate = 0.1
        self.discount = 0.99
        self.observation_space = self.env.observation_space.shape
        
        

    def forward(self):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.layer1(x))
        return self.layer2(x)

    def action(self, state):
        exploration_rate_threshold = np.random.random() # tilfeldig float fra 0-1
        exploration_rate = self.exp_rate
        if (exploration_rate_threshold <= exploration_rate):
            action = self.env.action_space.sample()
        else:
            action = np.argmax('''#todo''')
        return action
    
    def get_exploration_rate(self):
        return max(self.exp_rate * self.decay_rate, self.min_exp_rate)
    
    def updated_q_value(self, state, action, reward, new_state):
        return (self.lr * (reward + self.discount * np.max(self.Q_Values[new_state]) - self.Q_Values[state][action]))
    
    def processFrame(self, state):
        gray_state = grayscale.vxycc709(state)
        rescaled_state = downscale.divide(gray_state, 8)
        downsampled_state = downsample.downsample(rescaled_state, self.num_colors)
        return downsampled_state

    def train(self, episodes=100, steps=1000):
        rewards = []
        episode = 0
        while True:
            self.exp_rate = self.get_exploration_rate()
            observation, info = self.env.reset()
            state = self.processFrame(observation)

            #done = False
            reward_current_ep = 0
            for _ in range(steps):
                self.env.render()
                action = self.action(state)
                new_state, reward, done, truncated, info = self.env.step(action)
                new_state = self.processFrame(new_state)


                #self.Q_Values[new_state][action] += self.updated_q_value(state, action, reward, new_state)
                #TODO :)
               
                state = new_state
                reward_current_ep += reward
                if done or truncated:
                    break
            rewards.append(reward_current_ep)
            print(f"Score for episode {episode+1}: {rewards[episode]}")
            print(f"Exploration probability: {self.exp_rate}")
            episode +=1

    def save_q_values(self):
        torch.save(self.state_dict(),"dqn.pth")
    
    def run(self):
    

        done = False
        truncated = False
        state, info = self.env.reset()
        state = self.processFrame(state)
        while not (done or truncated):
            self.env.render()
            action = self.action(state)
            new_state, reward, done, truncated, info = self.env.step(action)
            new_state = self.processFrame(new_state)
            state = new_state
        
        self.env.close()


def custom_interrupt_handler(signum, frame):
    # This function will be called when Ctrl+C is pressed
    print("\nCustom interrupt handler activated.")
    model.save_q_values()
    print("Q_values saved")
    exit()


training = True

batch_size = 100


if training:
    env = mario_bros_env.make(
        'SuperMarioBros-v0',
        render_mode=None
    )
    env = JoypadSpace(env, RIGHT_ONLY)

    model = Model(env)
    target_model = Model(env)
    target_model.load_state_dict(model.state_dict)

    # Register the custom interrupt handler for Ctrl+C (SIGINT)
    signal.signal(signal.SIGINT, custom_interrupt_handler)
    model.train()
    model.run()

else:
    env = mario_bros_env.make(
        'SuperMarioBros-v0',
        render_mode="human"
    )
    env = JoypadSpace(env, RIGHT_ONLY)
    model = Model(env)
    model.load_state_dict(torch.load("dqn.pth"))
    model.run()
