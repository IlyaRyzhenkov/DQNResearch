import numpy as np
from torch import nn
import torch
import random
import network


class DQNAgent(nn.Module):
    def __init__(self, state_dim, action_n):
        super().__init__()
        self.state_dim = state_dim
        self.action_n = action_n

        self.gamma = 0.95
        self.epsilon = 1
        self.memory_size = 10000
        self.memory = []
        self.batch_size = 64
        self.learinig_rate = 1e-2

        self.q = network.Network(self.state_dim, self.action_n)
        self.optimazer = torch.optim.Adam(self.q.parameters(), lr=self.learinig_rate)

    def get_action(self, state):
        state = torch.FloatTensor(state)
        argmax_action = torch.argmax(self.q(state))
        probs = np.ones(self.action_n) * self.epsilon / self.action_n
        probs[argmax_action] += 1 - self.epsilon
        actions = np.arange(self.action_n)
        action = np.random.choice(actions, p=probs)
        return action

    def fit(self, state, action, reward, pre_done, done, next_state):
        self.memory.append([state, action, reward, pre_done, done, next_state])
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)

        if len(self.memory) > self.batch_size:
            batch = random.sample(self.memory, self.batch_size)

            states, actions, rewards, pre_dones, dones, next_states = list(zip(*batch))
            states = torch.FloatTensor(states)
            q_values = self.q(states)
            next_states = torch.FloatTensor(next_states)
            next_q_values = self.q(next_states)
            targets = self.calculate_targets(q_values, next_q_values, actions, rewards, pre_dones, dones)

            loss = torch.mean((targets.detach() - q_values) ** 2)

            loss.backward()
            self.optimazer.step()
            self.optimazer.zero_grad()

            if self.epsilon > 0.01:
                self.epsilon *= 0.999

    def calculate_targets(self, q_values, next_q_values, actions, rewards, pre_dones, dones):
        targets = q_values.clone()
        for i in range(self.batch_size):
            if not dones[i]:
                targets[i][actions[i]] = rewards[i] + self.gamma * max(next_q_values[i])
            else:
                targets[i][actions[i]] = rewards[i]
        return targets
