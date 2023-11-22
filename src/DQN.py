import numpy as np
from nes_py.wrappers import JoypadSpace
import mario_bros_env
from mario_bros_env.actions import *
from wrappers import GrayscaleEnv, DownscaledEnv
from wrappers.filters import GrayscaleFilters
import signal
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple, deque
import random
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))
class ReplayBuffer(object):
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class Model(nn.Module):
    def __init__(self,observation_num, num_action ):
        super(Model, self).__init__()
        self.observation_num = observation_num
        self.num_action = num_action
       
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)

        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(1600, 128)
        self.layer2 = nn.Linear(128, self.num_action)


    def forward(self,x):
        x = self.conv1(x) 
        x = nn.functional.relu(x) 
        x = self.conv2(x)
        x = nn.functional.relu(x) 

        x = self.flatten(x)
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return x


class Agent:
    def __init__(self,
                env,
                observation_space_n,
                action_space_n,
                memory_capacity=100_000,
                discount=0.95,
                learning_rate=0.0005,
                exp_rate=1.0,
                min_exp_rate=0.15,
                exp_decay=0.99,
                num_stacked_frames=4,
                 ):

        self.device = device
        self.env = env

        self.action_space_n = action_space_n
        self.observation_space_n = observation_space_n

        # Define the number of stacked frames
        self.num_stacked_frames = num_stacked_frames

        # parameters
        self.memory_capacity = memory_capacity
        self.discount = discount
        self.learning_rate = learning_rate
        self.exp_rate = exp_rate
        self.min_exp_rate = min_exp_rate
        self.exp_decay = exp_decay

        # networks
        self.model = Model(self.observation_space_n, self.action_space_n).to(device)
        self.target_model = Model(self.observation_space_n, self.action_space_n).to(device)

        # agent memory
        self.replay_memory = ReplayBuffer(self.memory_capacity)

        # rewards
        self.rewards = []

        # optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # self.target_model.eval()
        self.target_model.load_state_dict(self.model.state_dict())

        # Initialize the deque for frame stacking
        self.stacked_frames = np.zeros((num_stacked_frames, 30, 30))

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def get_exploration_rate(self):
        """ returns exploration rate """

        return max(self.exp_rate, self.min_exp_rate)

    def update_exploration_rate(self):
        """ Updates exploration rate """

        self.exp_rate = self.exp_rate * self.exp_decay
        return self.exp_rate

    def save_model(self):
        """ Saves model to the .pth file format """

        torch.save(self.model.state_dict(), "dqn.pth")
        print("saved model to 'dqn.pth'")

    def load_model(self):
        """ Loads network from .pth file """

        self.model.load_state_dict(torch.load("dqn.pth"))
        self.model.eval()

    def action(self, state):
        """ Selects an action using the epsilon-greedy strategy """

        state = state.reshape(1, 4, 30, 30) # needs extra dim for batch of size 1
        exploration_rate_threshold = np.random.random()  # random float between 0-1
        exploration_rate = self.get_exploration_rate()
        

        if exploration_rate_threshold <= exploration_rate:  # do random action
            action = random.randrange(0, self.action_space_n)
            action_t = torch.tensor([[action]], device=device, dtype=torch.int64)
        else:
            action_argmax = self.model(torch.tensor(state, device=device, dtype=torch.float32)).argmax() 
            action = action_argmax
            action_t = action.reshape(1, 1)
        return action_t
    

    def processFrame(self, state):
        """ Greysscales, downscales, and downsample image. Returns 30x30 image """

        state = np.squeeze(state)
        cropped_state = []
        for i in range(len(state)):
            cropped_state.append(torch.tensor(state[i-1][:-2]))

        # Normalize pixel values to the range [0, 1]
        result = [element / 255 for element in cropped_state]
        return result
    
    
    def stack_frames(self, stacked_frames, new_frame, is_new_episode):
        """ Function to stack frames. stacked_frames is a deque. """
        frame = self.processFrame(new_frame)

        if is_new_episode:
            stacked_frames = np.zeros((self.num_stacked_frames, 30, 30))

        stacked_frames = np.roll(stacked_frames, shift=-1, axis=0)
        stacked_frames[-1, :] = frame

        return stacked_frames


    def optimize(self, batch_size):
        """" Samples from replay_memory.
             Does optimizer step and updates model """

        if len(self.replay_memory) < batch_size:
            return

        transitions = self.replay_memory.sample(batch_size)
        batch = Transition(*zip(*transitions))

        state_b = torch.cat(batch.state) # shape: [32, 4, 30, 30]
        next_state_b = torch.cat(batch.next_state) # shape: [32, 4, 30, 30]
        action_b = torch.cat(batch.action) # shape: [32, 1]
        done_t = torch.cat(batch.done).unsqueeze(1) # shape: [32, 1]
   
        target_q = self.target_model(next_state_b) # shape: [32, 5]
   
        max_target_q = torch.max(target_q, dim=1, keepdim=True)[0] # shape: [32, 1]
       
        r = torch.cat(batch.reward)  # shape: [32]
        r.unsqueeze_(1) # shape: [32, 1] 

        # Q(s, a) = r + γ * max(Q(s', a')) ||
        # Q(s, a) = r                      || if state is done
        Q_sa = r + self.discount * max_target_q * (1 - done_t)  # if done = 1 => Q_result = r
        Q_sa = Q_sa.reshape(-1, 1)
      
        predicted = torch.gather(input=self.model(state_b), dim=1, index=action_b) # shape: [32,1]
        loss = F.mse_loss(predicted, Q_sa)
        self.optimizer.zero_grad()
        loss.backward()

        # Uncomment for clipping:
        #
        #CLIP_NORM = 0.6
        #torch.nn.utils.clip_grad_norm_(self.model.parameters(),CLIP_NORM)

        self.optimizer.step()


    def plot_rewards(self):
        """ Plots mean rewards in a line diagram """
        mean_rewards = []

        for t in range(len(self.rewards)):
            mean_rewards.append(np.mean(self.rewards[max(0, t-100):(t+1)]))
        plt.plot(mean_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Mean Reward')
        plt.savefig('rewards.png')

    def train(self, episodes=100, steps=10_000):
        """ Trains model for n episodes (does not save the model) """
        self.rewards = []
        initial_state, _ = env.reset()
        stacked_frames = np.roll(self.stacked_frames, shift=-1, axis=0)
        stacked_frames[-1, :] = self.processFrame(initial_state)
        total_steps = 0

        for episode in range(episodes):
            ep_reward = 0
            state, _ = env.reset()
            stacked_frames = self.stack_frames(stacked_frames, state, True)

            for s in range(steps):
                action = agent.action(stacked_frames)
                observation, reward, terminated, truncated, _ = env.step(action.item())

                ep_reward += reward
                reward = torch.tensor([reward], device=device)

                done = terminated or truncated
                done_t = torch.tensor(done, dtype=torch.float32, device=device).unsqueeze(0)

                next_stacked_frames = self.stack_frames(stacked_frames, observation, False)
             
                stacked_frames_t = torch.tensor(stacked_frames,dtype=torch.float32, device=device).unsqueeze(0)
                next_stacked_frames_t = torch.tensor(next_stacked_frames,dtype=torch.float32, device=device).unsqueeze(0)

                # store
                agent.replay_memory.push(stacked_frames_t, action, next_stacked_frames_t, reward, done_t)

                stacked_frames = next_stacked_frames

                # optimize
                agent.optimize(batch_size)
        
                if total_steps % update_frequency == 0:
                    agent.update_target_model()

                total_steps+=1

                if done:
                    break

            self.rewards.append(ep_reward)
            print("episode: " + str(episode) + " reward: " + str(ep_reward))
            if episode % 100 == 0:
                print(f'exp_rate: {agent.get_exploration_rate()}')
            self.update_exploration_rate()


    def custom_interrupt_handler(self,signum, frame):
        """ This function will be called when Ctrl+C is pressed """

        print("\nCustom interrupt handler activated.")
        self.save_model()
        print("Q_values saved")
        self.plot_rewards()

        exit()

batch_size = 32 
update_frequency = 1000

# Training: creates a new model. No render
# Not training: loads model (from dqn.pth). With render
training = True

observation_n =30*30

if training:
    env = mario_bros_env.make(
        'SuperMarioBros-v0',
        render_mode=None
    )
    env = JoypadSpace(env, RIGHT_ONLY)

    # Process frames
    env = GrayscaleEnv(env, GrayscaleFilters.RED)
    env = DownscaledEnv(env, 8)

    env_action_num = env.action_space.n
    state, info = env.reset()
    n_observations = observation_n

    agent = Agent(env, n_observations, env_action_num, exp_rate=0.7)
    
    state, _ = env.reset()
    state = torch.tensor(state.copy(), dtype=torch.float32, device=device).unsqueeze(0)

    # Register the custom interrupt handler for Ctrl+C (SIGINT)
    signal.signal(signal.SIGINT, agent.custom_interrupt_handler)
   
    agent.train(episodes=50_000)
    agent.save_model()
    agent.plot_rewards()
    

else:
    env = mario_bros_env.make(
        'SuperMarioBros-v0',
        render_mode="human"
    )
    env = JoypadSpace(env, RIGHT_ONLY)

    # Process frames
    env = GrayscaleEnv(env, GrayscaleFilters.RED)
    env = DownscaledEnv(env, 8)


    env_action_num = env.action_space.n
    state, info = env.reset()
    n_observations = observation_n

    agent = Agent(env, n_observations, env_action_num, 
    exp_rate=0.1,
    )

    # Loads trained model
    agent.load_model()

    state, _ = env.reset()
    state = torch.tensor(state.copy(), dtype=torch.float32, device=device).unsqueeze(0)

    # Register the custom interrupt handler for Ctrl+C (SIGINT)
    signal.signal(signal.SIGINT, agent.custom_interrupt_handler)
    agent.train(episodes=10_000)
    agent.save_model()
    agent.plot_rewards()