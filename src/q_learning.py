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
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# A named tuple representing a single transition in our environment. 
# It maps (state, action) pairs to their (next_state, reward) result
#https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))
#https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
class ReplayBuffer(object):

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

        self.layer1 = nn.Linear(self.observation_num, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, self.num_action)
        # 4 stacked frames as input (which will be one state)
        #self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1) # 1 frame as state
        #self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1,padding=1)
        #self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=1)

        #self.layer1 = nn.Linear(32*30*28,512)
        #self.layer2 = nn.Linear(512, num_action)

  #      self.image_dimentions = (30,32)
   #     self.num_colors = 4
 
        

    def forward(self,x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x
        """x = F.relu(self.conv1(x))
        print(x.size())

        x = F.relu(self.conv2(x))
        print(x.size())

        #x = F.relu(self.conv3(x))

        x = torch.flatten(x, start_dim=0)

        print(x.size())


        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer2(x)
        return x
    #    x = F.relu(self.conv1(x))
     #   x = F.relu(self.conv2(x))
      #  x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.layer1(x))
        #return self.layer2(x)"""

class Agent:
    def __init__(self,
                 env,
                 observation_space_n,
                 action_space_n,
                 memory_capacity=100000,
                 discount=0.99,
                 learning_rate=0.001,
                 exp_rate=0.9,
                 min_exp_rate=0.1,
                 exp_decay=0.9990, #FIXME prøv 0.99991 når du trener hjemme
                 ):

        self.device = device
        self.env = env

        self.action_space_n = action_space_n
        self.observation_space_n = observation_space_n

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

    def update_target_model(self):
       
        self.target_model.load_state_dict(self.model.state_dict())

    def get_exploration_rate(self):
        """returns exploration rate"""
        return max(self.exp_rate, self.min_exp_rate)

    def update_exploration_rate(self):
        """Updates exploration rate"""

        self.exp_rate = self.exp_rate * self.exp_decay
        return self.exp_rate

    def save_model(self):
        """Saves model to the .pth file format"""

        torch.save(self.model.state_dict(), "dqn.pth")
        print("saved model to 'dqn.pth'")

    def load_model(self):
        """Loads network from .pth file """

        self.model.load_state_dict(torch.load("dqn.pth"))
        self.model.eval()

    def action(self, state):
        """Selects an action using the epsilon-greedy strategy"""

        exploration_rate_threshold = np.random.random()  # random float between 0-1
        exploration_rate = self.get_exploration_rate()
        

        if exploration_rate_threshold <= exploration_rate:  # do random action
            action = random.randrange(0, self.action_space_n)
            action_t = torch.tensor([[action]], device=device, dtype=torch.long)
        else:
            # action_t = self.model(state).max(dim=1)[1].reshape(1, 1)
            action_t = self.model(state).argmax().reshape(1, 1)
        return action_t

    def processFrame(self, state):
        gray_state = grayscale.vxycc709(state)
        rescaled_state = downscale.divide(gray_state, 8)
        downsampled_state = downsample.downsample(rescaled_state, 4)
        #return downsampled_state
        return torch.cat(tuple(torch.tensor(downsampled_state)))
    
    def optimize(self, batch_size):
        if len(self.replay_memory) < batch_size:
            return

        transitions = self.replay_memory.sample(batch_size)
        batch = Transition(*zip(*transitions))

        state_b = torch.cat(batch.state)
        next_state_b = torch.cat(batch.next_state)
        action_b = torch.cat(batch.action)
        done_t = torch.cat(batch.done).unsqueeze(1)

        target_q = self.target_model(next_state_b)
        max_target_q = target_q.argmax()

        r = torch.cat(batch.reward)  # dim [n]
        r = r.unsqueeze(1)  # dim [x,1]

        # Q(s, a) = r + γ * max(Q(s', a')) ||
        # Q(s, a) = r                      || if state is done
        Q_sa = r + self.discount * max_target_q * (1 - done_t)  # if done = 1 => Q_result = r
        Q_sa = Q_sa.reshape(-1, 1)

        predicted = torch.gather(input=self.model(state_b), dim=1, index=action_b)

        loss = F.mse_loss(predicted, Q_sa)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def plot_rewards(self):
        # Plotting the rewards
        plt.plot(self.rewards, marker='o')
        plt.title('Rewards per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.show()

    def train(self, episodes=100, steps=4000):
        self.rewards = []
        for episode in range(num_episodes):
            ep_reward = 0
            state, info = env.reset()
            state = agent.processFrame(state)
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            
            for s in range(steps):
                action = agent.action(state)
                observation, reward, terminated, truncated, info = env.step(action.item())
                observation = agent.processFrame(observation)
                ep_reward += reward
                
                reward = torch.tensor([reward], device=device)

                done = terminated or truncated
                done_t = torch.tensor(done, dtype=torch.float32, device=device).unsqueeze(0)

                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

                # store
                agent.replay_memory.push(state, action, next_state, reward, done_t)

                state = next_state

                # optimize
                agent.optimize(batch_size)

                if s % update_frequency == 0:
                    agent.update_target_model()

                if done:
                    break
            self.rewards.append(ep_reward)


            print("episode: " + str(episode) + " reward: " + str(ep_reward))
            if episode % 100 == 0:
                print(f'exp_rate: {agent.get_exploration_rate()}')
            self.update_exploration_rate()

  
    
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


    def custom_interrupt_handler(self,signum, frame):
        # This function will be called when Ctrl+C is pressed
        print("\nCustom interrupt handler activated.")
        self.save_model()
        print("Q_values saved")
        self.plot_rewards()

        exit()


num_episodes = 2000
batch_size = 128
update_frequency = 10
training = True


if training:
    env = mario_bros_env.make(
        'SuperMarioBros-v0',
        render_mode=None
    )
    env = JoypadSpace(env, RIGHT_ONLY)

    env_action_num = env.action_space.n
    state, info = env.reset()
    n_observations = 32*30
   

    agent = Agent(env, n_observations, env_action_num)
    #state= agent.processFrame(state)
    #print(state.shape)
    
    state, _ = env.reset()
    state = torch.tensor(state.copy(), dtype=torch.float32, device=device).unsqueeze(0)
   # model = Model(env)
    #target_model = Model(env)
    #target_model.load_state_dict(model.state_dict)

    # Register the custom interrupt handler for Ctrl+C (SIGINT)
    signal.signal(signal.SIGINT, agent.custom_interrupt_handler)
    agent.train(1000)
    agent.save_model()
    agent.plot_rewards()

    #model.run()

else:
    env = mario_bros_env.make(
        'SuperMarioBros-v0',
        render_mode="human"
    )
    env = JoypadSpace(env, RIGHT_ONLY)

    env_action_num = env.action_space.n
    state, info = env.reset()
    n_observations = 30*32

    agent = Agent(env, n_observations, env_action_num, 
    exp_rate=0.1,
    )
    agent.load_model()

    state, _ = env.reset()
    state = torch.tensor(state.copy(), dtype=torch.float32, device=device).unsqueeze(0)
   # model = Model(env)
    #target_model = Model(env)
    #target_model.load_state_dict(model.state_dict)

    # Register the custom interrupt handler for Ctrl+C (SIGINT)
    signal.signal(signal.SIGINT, agent.custom_interrupt_handler)
    agent.train(1000)
    agent.plot_rewards()
   # model.run()
