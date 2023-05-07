from env import ThreeByThreeGameEnv
import gymnasium as gym
import numpy as np
import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions
import numpy as np


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout = 0.1):
        super().__init__()        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.PReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        x = self.net(x)
        return x
    
class Actor(nn.Module):
    def __init__(self, env: gym.Env, alpha=0.1, gamma=1.0, epsilon=0.1):
        super().__init__()
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.net = MLP(3*3, 128, 2*3*3)
        self.apply(init_weights)
    
    def forward(self, state):
        return self.net(state)

class Critic(nn.Module):
    def __init__(self, env: gym.Env, alpha=0.1, gamma=1.0, epsilon=0.1):
        super().__init__()
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.net = MLP(3*3, 128, 1)
        self.apply(init_weights)

    def forward(self, state):
        return self.net(state)
    
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0)


if __name__ == "__main__":
    env = ThreeByThreeGameEnv()
    actor = Actor(env)
    critic = Critic(env)


    observation, _ = env.reset()
    observation = torch.from_numpy(observation.flatten()).to(dtype=torch.float)
    
    print(actor(observation))
    print(critic(observation))