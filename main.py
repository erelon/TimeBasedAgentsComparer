from collections import defaultdict

from env import StatelessEnv
from learners import RandomAgent, QLearningAgent, RLAgent, ContinuesMAB, OracleAgent, MAB


def train_single_agent(agent, env, episodes=200, eval_steps=20, seed=42):
    """
    Train the agent in the given environment for a specified number of episodes.
    """
    env.set_seed(seed)
    env.reset()
    state = env.get_state()  # In a stateless environment, state is not used
    for episode in range(episodes):
        action = agent.act(state)
        time, reward = env.get_reward(agent, action)
        new_state = env.get_state()
        agent.learn(state, action, reward, new_state, time)
        state = new_state

    rewards = []
    for _ in range(eval_steps):
        action = agent.eval(state)
        time, reward = env.get_reward(agent, action)
        state = env.get_state()
        rewards.append(reward)

    return sum(rewards) / len(rewards)


if __name__ == '__main__':
    env = StatelessEnv("Stateless Environment")

    orcale = OracleAgent(name="Oracle Agent", action_space=env.get_action_space(), env_secret=env.secret())

    q_agent = QLearningAgent(name="QLearning Agent", action_space=env.get_action_space())
    random_agent = RandomAgent(name="Random Agent", action_space=env.get_action_space())
    r_agent_with_trick = RLAgent(name="RL Agent with trick", action_space=env.get_action_space())
    r_agent_without_trick = RLAgent(name="RL Agent without trick", action_space=env.get_action_space(),
                                    with_rho_trick=False)
    mab = MAB(name="MAB", action_space=env.get_action_space())
    c_mab = ContinuesMAB(name="Continues MAB", action_space=env.get_action_space())

    agents = [orcale, random_agent, q_agent, r_agent_with_trick, r_agent_without_trick, c_mab, mab]
    episodes = 2000
    eval_steps = 100
    for agent in agents:
        # Check if that the agent learning is consistent - for each state what is the chosen action?
        # do all learning agents agree?
        best_action_per_state = defaultdict(list)
        avg_rewards = []
        for i in range(1, 100):
            agent.reset()
            avg_reward = train_single_agent(agent, env, episodes=episodes, eval_steps=eval_steps, seed=i)
            avg_rewards.append(avg_reward)
            for state in agent.q_table:
                actions = agent.q_table[state]
                best_action = max(actions, key=actions.get)
                best_action_per_state[state].append(best_action)
        # Print the best action ratio for each state
        print(f"Best action ratio for {agent.name}:")
        for state, actions in best_action_per_state.items():
            best_action_ratio = len([i for i in actions if i == orcale.act(state)]) / len(actions)
            print(f"State {state}: Best Action Ratio: {best_action_ratio}")
        print(f"{agent.name}: Average Reward over {eval_steps} steps: {sum(avg_rewards) / len(avg_rewards)}")
        print("-" * 50)
        print()
