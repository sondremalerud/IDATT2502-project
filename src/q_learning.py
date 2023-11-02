from mario_bros_env import actions
import numpy as np
from nes_py.wrappers import JoypadSpace
import mario_bros_env
from mario_bros_env.actions import RIGHT_ONLY
from filters import grayscale, downscale
import signal
import sys

class Model:
    def __init__(self, env):
        self.states = []
        self.actions = actions.RIGHT_ONLY
        self.lr = 0.2
        self.exp_rate = 1.0
        self.env = env
        self.discount = 0.99
        self.observation_space = self.env.observation_space.shape
        self.decay_rate = 0.995
        self.Q_Values = np.zeros((256, 240, self.env.action_space.n))

    def action(self, state):
        exploration_rate_threshold = np.random.random() # tilfeldig float fra 0-1
        exploration_rate = self.exp_rate
        if (exploration_rate_threshold <= exploration_rate):
            action = self.env.action_space.sample()
        else:
            action = np.argmax(self.Q_Values[state])
        return action
    
    def get_exploration_rate(self):
        return self.exp_rate * self.decay_rate
    
    def updated_q_value(self, state, action, reward, new_state):
        return (self.lr * (reward + self.discount * np.max(self.Q_Values[new_state]) - self.Q_Values[state][action]))
    
    def processFrame(self, state):
        gray_state = grayscale.vxycc709(state)
        rescaled_state = downscale.divide(gray_state, 9)
        return rescaled_state

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
                self.Q_Values[new_state][action] += self.updated_q_value(state, action, reward, new_state)
                state = new_state
                reward_current_ep += reward

                if done or truncated:
                    break
            rewards.append(reward_current_ep)
            print(f"Score for episode {episode+1}: {rewards[episode]}")
            print(f"Exploration probability: {self.exp_rate}")
            episode +=1

    def save_q_values(self):
        np.save("q_values.npy", self.Q_Values)
    
    def run(self):
        self.Q_Values = np.load("q_values.npy")

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

if training:
    env = mario_bros_env.make(
        'SuperMarioBros-v0',
        render_mode=None
    )
    env = JoypadSpace(env, RIGHT_ONLY)
    model = Model(env)
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
    model.run()
