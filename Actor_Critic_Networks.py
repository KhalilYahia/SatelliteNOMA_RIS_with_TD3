import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(state_dim, 400)
        self.layer2 = nn.Linear(400, 300)
        self.layer3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = torch.relu(self.layer1(state))
        x = torch.relu(self.layer2(x))
        return self.max_action * torch.tanh(self.layer3(x))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Q1 network
        self.layer1 = nn.Linear(state_dim + action_dim, 400)
        self.layer2 = nn.Linear(400, 300)
        self.layer3 = nn.Linear(300, 1)
        # Q2 network
        self.layer4 = nn.Linear(state_dim + action_dim, 400)
        self.layer5 = nn.Linear(400, 300)
        self.layer6 = nn.Linear(300, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        q1 = torch.relu(self.layer1(sa))
        q1 = torch.relu(self.layer2(q1))
        q1 = self.layer3(q1)

        q2 = torch.relu(self.layer4(sa))
        q2 = torch.relu(self.layer5(q2))
        q2 = self.layer6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], dim=1)
        q1 = torch.relu(self.layer1(sa))
        q1 = torch.relu(self.layer2(q1))
        return self.layer3(q1)
