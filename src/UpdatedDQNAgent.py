import torch
from src import DQNAgent


class UpdatedDQNAgent(DQNAgent.DQNAgent):
    def __init__(self, state_dim, action_n, env):
        super().__init__(state_dim, action_n)
        self.env = env

    def calculate_targets(self, q_values, next_states, actions, rewards, dones):
        targets = q_values.clone()
        next_q_values = self.q(torch.FloatTensor(next_states))
        for i in range(self.batch_size):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                max_val = float("-inf")
                for next_action in actions:
                    next_state, next_reward, next_done = self.env.virtual_step(next_states[i], next_action)
                    if not next_done:
                        val = rewards[i] + self.gamma * next_reward + self.gamma ** 2 * max(
                            self.q(torch.FloatTensor([next_state]))[0])
                    else:
                        val = rewards[i] + self.gamma * max(next_q_values[i])
                    if val > max_val:
                        max_val = val
                targets[i][actions[i]] = rewards[i] + self.gamma * max_val
        return targets
