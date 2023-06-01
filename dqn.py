"""
Double DQN for solving atari games. 

We use two Q networks as training and updating one is highly unstable. So we slowly keep updating the target
network to train a local network.

Change the ENV_NAME and REWARD_MAX to train the agent
"""

import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
import random
from collections import deque, namedtuple
import matplotlib.pyplot as plt 

ENV_NAME = "Acrobot-v1"
REWARD_MAX = -100

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("mps" if torch.has_mps else "cpu")

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        # self.conv1 = nn.Conv2d(4, 16)
        # self.conv2 = nn.Conv2d(16, 32)
        # self.maxpool = nn.MaxPool2d(2, 2)
        self.layer1 = nn.Linear(state_size, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, action_size)

    def forward(self, x):
        # x = self.maxpool(self.conv1(x))
        # x = self.maxpool(self.conv2(x))
        # x = torch.flatten(x)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    
    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    
class Agent():
    def __init__(self, action_size, state_size) -> None:
        self.action_size = action_size
        self.state_size = state_size

        # Q networks
        self.qnetwork_target = QNetwork(state_size = self.state_size, action_size=self.action_size).to(device)
        self.qnetwork_local = QNetwork(state_size = self.state_size, action_size=self.action_size).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        self.t_step = 0
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        self.t_step = (self.t_step+1) % UPDATE_EVERY
        if self.t_step == 0:
            if(len(self.memory) > BATCH_SIZE):
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)
    
    def learn(self, experiences, gamma): 
        state, action, reward, next_state, done = experiences

        Q_target_next = self.qnetwork_target(next_state).detach().max(1)[0].unsqueeze(1) # get q value of s+1 state
        Q_targets = reward + (gamma * Q_target_next * (1 - done)) # r + gamma * Q(s+1, a)
        Q_expected = self.qnetwork_local(state).gather(1, action) # get the Q(s,a) using local network ie to train
        ## gather() helps to get the q value corresponding to the chosen aciton.

        loss = F.mse_loss(Q_expected, Q_targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)
    
    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
        
    def act(self, state, eps=0.8):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
        

def dqn(agent, env, n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        state, _ = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _, info = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score)       # to check if env solved or not
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'dqn.pth')
        if np.mean(scores_window) >= REWARD_MAX: # solved 
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'dqn.pth')
            break
    return scores

if __name__ == "__main__":

    env = gym.make(ENV_NAME)
    agent = Agent(state_size=env.observation_space.shape[0], action_size=env.action_space.n)
    scores = dqn(agent, env)

    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()


    # load the weights from file
    agent.qnetwork_local.load_state_dict(torch.load('dqn.pth'))

    env = gym.make(ENV_NAME, render_mode="human")
    for i in range(3):
        state, _ = env.reset()
        for j in range(200):
            action = agent.act(state)
            env.render()
            state, reward, done, _, info = env.step(action)
            if done:
                break 
                
    env.close()