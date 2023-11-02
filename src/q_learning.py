from mario_bros_env import actions
import numpy as np
from nes_py.wrappers import JoypadSpace
import mario_bros_env
from mario_bros_env.actions import RIGHT_ONLY

class Model:
    def __init__(self, env):
        self.states = []
        self.actions = actions.RIGHT_ONLY
        self.lr = 0.2
        self.exp_rate = 0
        self.env = env
        self.discount = 0.99
        self.observation_space = self.env.observation_space.shape
        #self.Q_Values = np.zeros(self.observation_space + (self.env.action_space.n,))
        self.Q_Values = np.zeros((256, 240, 3, self.env.action_space.n))

    def action(self, state):
        exploration_rate_threshold = np.random.random() # tilfeldig float fra 0-1
        exploration_rate = self.exp_rate
        if (exploration_rate_threshold <= exploration_rate):
            action = self.env.action_space.sample()
        else:
            action = np.argmax(self.Q_Values[state])
        return action
    
    def updated_q_value(self, state, action, reward, new_state):
        return (self.lr * (reward + self.discount * np.max(self.Q_Values[new_state]) - self.Q_Values[state][action]))
    
    def train(self, rounds=1):
        print("Hallo")
        rewards = []
        for episode in range(rounds):
            state, info = self.env.reset()
            done = False
            reward_current_ep = 0 
            print("Episode "+str(episode))
            while not done:
                print("While loop")
                action = self.action(state)
                new_state, reward, done, truncated, info = self.env.step(action)
                self.Q_Values[new_state][action] += self.updated_q_value(state, action, reward, new_state)
                state = new_state
                reward_current_ep += 1
            rewards.append(reward_current_ep)
            print(f"Score for episode {episode+1}: {rewards[episode]}")

        print('Finished')
        return rewards

    def run(self):
        done = False
        state, info = self.env.reset()
        while not done:
            self.env.render()
            action = self.action(current_state)
            state, reward, done, truncated, info = self.env.step(action)
            current_state = state
    


env = mario_bros_env.make(
    'SuperMarioBros-v0',
    render_mode=None
)
env = JoypadSpace(env, RIGHT_ONLY)

model = Model(env)
model.train()
model.run()
