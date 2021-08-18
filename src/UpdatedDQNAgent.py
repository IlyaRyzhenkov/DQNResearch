import torch

import DQNAgent


class UpdatedDQNAgent(DQNAgent.DQNAgent):
    def __init__(self, state_dim, action_n, env):
        super().__init__(state_dim, action_n)
        self.env = env

    def calculate_targets(self, q_values, next_q_values, actions, rewards, pre_dones, dones):
        targets = q_values.clone()
        for i in range(self.batch_size):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            elif pre_dones[i]:
                targets[i][actions[i]] = rewards[i] + self.gamma * max(next_q_values[i])
            else:
                max_val = -10000000
                for action in actions:
                    state, rew = self.env.virtual_step(action)
                    val = rewards[i] + rew + self.gamma * max(self.q(torch.FloatTensor([state]))[0])
                    if val > max_val:
                        max_val = val
                targets[i][actions[i]] = rewards[i] + self.gamma * max_val
        return targets
