import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions

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
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.PReLU(),
            
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        x = self.net(x)
        return x
    
class ConvNet(nn.Module):
    def __init__(self, hidden_dim=128, kernal_size=(1,3), dropout = 0.1):
        super().__init__()        
        self.net = nn.Sequential(
            nn.Conv2d(1, hidden_dim, kernel_size=kernal_size),
            nn.Dropout(dropout),
            nn.ReLU(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.Dropout(dropout),
            # nn.PReLU(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.Dropout(dropout),
            # nn.PReLU(),
            
            # nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        x = self.net(x)
        return x

class ConvActor(nn.Module):
    def __init__(self):
        super().__init__()
        hidden_dim = 128
        self.conv1 = ConvNet(hidden_dim, kernal_size=(1,3))
        self.conv2 = ConvNet(hidden_dim, kernal_size=(3,1))
        self.conv3 = ConvNet(hidden_dim, kernal_size=(3,3))
        self.net = MLP(7*hidden_dim, hidden_dim, 3*3*2)
        self.apply(init_weights)
    
    def forward(self, state):
        out = [self.conv1(state), self.conv2(state), self.conv3(state)]
        out[1] = torch.swapaxes(out[1], 1,2)
        out = torch.cat(out , dim=1).flatten()

        return self.net(out)

class ConvCritic(nn.Module):
    def __init__(self):
        super().__init__()
        hidden_dim = 256
        self.conv1 = ConvNet(hidden_dim, kernal_size=(1,3))
        self.conv2 = ConvNet(hidden_dim, kernal_size=(3,1))
        self.conv3 = ConvNet(hidden_dim, kernal_size=(3,3))
        self.net = MLP(7*hidden_dim, hidden_dim, 1)
        self.apply(init_weights)
    
    def forward(self, state):
        out = [self.conv1(state), self.conv2(state), self.conv3(state)]
        out[1] = torch.swapaxes(out[1], 1,2)
        out = torch.cat(out , dim=1).flatten()

        return self.net(out)
    
class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = MLP(3*3, 256, 2*3*3)
        self.apply(init_weights)
    
    def forward(self, state):
        return self.net(state.reshape(state.shape[0], -1))

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = MLP(3*3, 256, 1)
        self.apply(init_weights)

    def forward(self, state):
        return self.net(state.reshape(state.shape[0], -1))
    
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
    
    action_logit = actor(observation)
    action_prob = F.softmax(action_logit, dim=-1)
    dist = distributions.Categorical(action_prob)
    print(action_prob)

    action = dist.sample()
    print(action)
    action = action.detach().item()
    for action in range(18):
        print(action)
        print(action//9, action%9//3, action%3)
    # print(critic(observation))