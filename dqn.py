from src import SimpleControlProblem, DQNAgent, UpdatedDQNAgent

if __name__ == "__main__":
    env = SimpleControlProblem.SimpleControlProblemDiscrete()
    state_dim = env.state_dim
    action_n = env.action_n
    agent = DQNAgent.DQNAgent(state_dim, action_n)
    agent = UpdatedDQNAgent.UpdatedDQNAgent(state_dim, action_n, env)

    episode_n = 100
    for episode in range(episode_n):
        state = env.reset()
        total_reward = 0
        for t in range(500):
            action = agent.get_action(state)
            next_state, reward, pre_done, done, _ = env.step(action)
            agent.fit(state, action, reward, pre_done, done, next_state)
            state = next_state
            total_reward += reward
            if done:
                break
        print(total_reward)
