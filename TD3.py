import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from Actor_Critic_Networks import Critic,Actor


class ReplayBuffer:
    def __init__(self, max_size=1000000):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))
        return states, actions, rewards, next_states, dones

    def size(self):
        return len(self.buffer)

class TD3:
    def __init__(self, state_dim, action_dim, max_action, gamma=0.99, tau=0.005, noise=0.2, noise_clip=0.5, policy_delay=2):
        self.actor = Actor(state_dim, action_dim, max_action).to("cpu")
        self.actor_target = Actor(state_dim, action_dim, max_action).to("cpu")
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to("cpu")
        self.critic_target = Critic(state_dim, action_dim).to("cpu")
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.noise = noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to("cpu")
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=100):
        if replay_buffer.size() < batch_size:
            return

        self.total_it += 1
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        states = torch.FloatTensor(states).to("cpu")
        actions = torch.FloatTensor(actions).to("cpu")
        rewards = torch.FloatTensor(rewards).to("cpu")
        next_states = torch.FloatTensor(next_states).to("cpu")
        dones = torch.FloatTensor(dones).to("cpu")

        # Select action according to policy and add clipped noise
        noise = torch.normal(0, self.noise, size=actions.size()).to("cpu").clamp(-self.noise_clip, self.noise_clip)
        next_action = (self.actor_target(next_states) + noise).clamp(-self.max_action, self.max_action)

        # Compute the target Q value
        target_Q1, target_Q2 = self.critic_target(next_states, next_action)
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = rewards + ((1 - dones) * self.gamma * target_Q).detach()

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(states, actions)

        # Compute critic loss
        critic_loss = nn.MSELoss()(current_Q1, target_Q) + nn.MSELoss()(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_delay == 0:
            # Compute actor loss
            actor_loss = -self.critic.Q1(states, self.actor(states)).mean()
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)



