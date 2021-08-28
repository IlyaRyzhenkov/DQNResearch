from src import SimpleControlProblem, DQNAgent, UpdatedDQNAgent, CSVWriter


def run_dqn(env, agent, episode_n=100):
    episode_stat = []
    for episode in range(episode_n):
        state = env.reset()
        total_reward = 0
        for t in range(500):
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.fit(state, action, reward, done, next_state)
            state = next_state
            total_reward += reward
            if done:
                break
        episode_stat.append(total_reward)
        print(total_reward)
    return episode_stat


if __name__ == "__main__":
    env = SimpleControlProblem.SimpleControlProblemDiscrete()
    state_dim = env.state_dim
    action_n = env.action_n
    agent = DQNAgent.DQNAgent(state_dim, action_n)
    print("Basic DQN test:")
    basic_dqn_stat = run_dqn(env, agent)

    env = SimpleControlProblem.SimpleControlProblemDiscrete()
    agent = UpdatedDQNAgent.UpdatedDQNAgent(state_dim, action_n, env)
    print("Updated DQN test:")
    updated_dqn_stat = run_dqn(env, agent)

    CSVWriter.CSVWriter.write_dqn_stat_scv("result.csv", basic_dqn_stat, updated_dqn_stat)
