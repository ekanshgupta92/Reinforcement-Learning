"""
    A(s, a) = Q(s,a) - V(s) 
therefore, Q(s, a) = V(s) + A(s,a) but this seems problematic so author suggests:
    Q(s, a) = V(s) + A(s,a) - A.mean()
Duelling DQN is an improvement of DQN where the model predicts both the A(s,a) and the V(s) for an input
state. We then calculate the Q(s,a) values and choose the action. Rest remains similar to dual DQN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import cv2
import gym
import ale_py
from collections import namedtuple, deque
import matplotlib.pyplot as plt

# Hyperparameters
LR = 5e-4
GAMMA = 0.99
TAU = 1e-3
BUFFER_SIZE = int(1e5)
BATCH_SIZE = 64
UPDATE_EVERY = 4
device = torch.device("mps" if torch.has_mps else "cpu")

# Experience Replay
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""
    def __init__(self, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.
        Params
        ======
        buffer_size (int): maximum size of buffer
        batch_size (int): size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        # self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = [state[None, :], action, reward, next_state[None, :], done]   # to create small batches with 4 frames for each
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        states, action, reward, next_states, done = zip(*random.sample(self.memory, k=self.batch_size))
        # Concat batches in one array 
        states = np.concatenate(states)
        next_states = np.concatenate(next_states)
        # Convert them to tensors, can't stack the images together so create batches
        states = torch.tensor(states, dtype=torch.float, device=device)
        next_states = torch.tensor(next_states, dtype=torch.float, device=device)
        action = torch.tensor(action, dtype=torch.long, device=device)
        reward = torch.tensor(reward, dtype=torch.float, device=device)
        done = torch.tensor(done, dtype=torch.float, device=device)

        return (states, action, reward, next_states, done)
    
    def __len__(self):
        """Return the current size of internal memory."""   
        return len(self.memory)
    

# Duel DQN model

class duelDQN(nn.Module):
    def __init__(self, h, w, action_size) -> None:
        super(duelDQN, self).__init__()
        self.h = h
        self.w = w
        self.action_size = action_size
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.h, self.w = self.get_size(self.h, self.w, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.h, self.w = self.get_size(self.h, self.w, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.h, self.w = self.get_size(self.h, self.w, kernel_size=3, stride=1)

        linear_output = self.h * self.w*64

        # for action value, A(s,a)
        self.Afc1 = nn.Linear(linear_output, 128)
        self.afc2 = nn.Linear(128, self.action_size)

        #for state value, V(s)
        self.Vfc1 = nn.Linear(linear_output, 128)
        self.Vfc2 = nn.Linear(128, 1)

    def get_size(self, h, w, kernel_size, stride):
        new_h = (h - kernel_size)//stride + 1
        new_w = (w - kernel_size)//stride + 1
        return new_h, new_w
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = x.view(x.size(0), -1) # flatten according to batch size. (torch.flatten gives just 1d tensor)

        Ax = F.leaky_relu(self.Afc1(x))
        Ax = self.afc2(Ax)

        Vx = F.leaky_relu(self.Vfc1(x))
        Vx = self.Vfc2(Vx)

        q = Vx + (Ax - Ax.mean())

        return q
    
class Agent():
    def __init__(self, state_size, action_size) -> None:
        self.action_size = action_size
        self.state_size_h = state_size[0]
        self.state_size_w = state_size[1]
        self.state_size_c = state_size[2]

        self.target_h = 80  # Height after process
        self.target_w = 64  # Widht after process

        self.crop_dim = [20, self.state_size_h, 0, self.state_size_w]  # Cut 20 px from top to get rid of the score table


        # Q networks
        self.qnetwork_target = duelDQN(h=self.target_h, w=self.target_w, action_size=self.action_size).to(device)
        self.qnetwork_local = duelDQN(h=self.target_h, w=self.target_w, action_size=self.action_size).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        self.t_step = 0
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)

    def preProcess(self, image):
        """
        Process image crop resize, grayscale and normalize the images
        """
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # To grayscale
        frame = frame[self.crop_dim[0]:self.crop_dim[1], self.crop_dim[2]:self.crop_dim[3]]  # Cut 20 px from top
        frame = cv2.resize(frame, (self.target_w, self.target_h))  # Resize
        frame = frame.reshape(self.target_w, self.target_h) / 255  # Normalize

        return frame

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        self.t_step = (self.t_step+1) % UPDATE_EVERY
        if self.t_step == 0:
            if(len(self.memory) > BATCH_SIZE):
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)
    
    def learn(self, experiences, gamma): 
        state, action, reward, next_state, done = experiences
        state_q_values = self.qnetwork_local(state)
        next_states_q_values = self.qnetwork_local(next_state)
        next_states_target_q_values = self.qnetwork_target(next_state)

        # Find selected action's q_value
        selected_q_value = state_q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        # Get indice of the max value of next_states_q_values
        # Use that indice to get a q_value from next_states_target_q_values
        # We use greedy for policy So it called off-policy
        next_states_target_q_value = next_states_target_q_values.gather(1, next_states_q_values.max(1)[1].unsqueeze(1)).squeeze(1)
        # Use Bellman function to find expected q value
        expected_q_value = reward + gamma * next_states_target_q_value * (1 - done)

        # Calc loss with expected_q_value and q_value
        loss = (selected_q_value - expected_q_value.detach()).pow(2).mean()

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
        state, _ = env.reset()         # new gym returns state, info_dict on reset
        state = agent.preProcess(state) 
        state = np.stack((state, state, state, state))
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _, _ = env.step(action)
            next_state = agent.preProcess(next_state)
            next_state = np.stack((next_state, state[0], state[1], state[2]))
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
        if np.mean(scores_window) >= 15.0: # solved 
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'dqn.pth')
            break
    return scores

if __name__ == "__main__":
    """
    Change the env name, works for all envs with a discrete action space. Also change the reward in dqn function
    along with the 
    """
    env = gym.make("ALE/Pong-v5")
    agent = Agent(state_size=env.observation_space.shape, action_size=env.action_space.n)
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

    for i in range(3):
        state = env.reset()
        for j in range(200):
            action = agent.act(state)
            env.render()
            state, reward, done, _,  _ = env.step(action)
            if done:
                break 
                
    env.close()