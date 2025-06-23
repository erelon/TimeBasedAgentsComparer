from collections import defaultdict

import pandas as pd

from env import *
from learners import *


def train_single_agent(agent, env, episodes=200, eval_steps=20, seed=42):
    """
    Train the agent in the given environment for a specified number of episodes.
    """
    random.seed(seed)
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


def experiment_runner(env, name="Experiment"):
    print(f"Running experiment with env: {name}")
    orcale = OracleAgent(
        name="Oracle Agent", action_space=env.get_action_space(), env_secret=env.secret())

    q_agent = QLearningAgent(name="QLearning",
                             action_space=env.get_action_space())
    continuousQ_agent = ContinuousQLearningAgent(name="Cont. QLearning",
                                                 action_space=env.get_action_space())
    random_agent = RandomAgent(
        name="Random Agent", action_space=env.get_action_space())
    r_agent_with_trick = RLAgent(
        name="RLearning (update on policy)", action_space=env.get_action_space())
    r_agent_without_trick = RLAgent(name="RLearning (update always)", action_space=env.get_action_space(),
                                    with_rho_trick=False)
    continuous_r_agent_with_trick = ContinuousRLAgent(name="Cont. RLearning (update on-policy)",
                                                      action_space=env.get_action_space())
    continuous_r_agent_without_trick = ContinuousRLAgent(name="Cont. RLearning (update always)",
                                                         action_space=env.get_action_space(),
                                                         with_rho_trick=False)
    mab = MAB(name="MAB", action_space=env.get_action_space())
    c_mab = ContinuesMAB(name="Continues MAB",
                         action_space=env.get_action_space())

    ucb = UCB(name="UCB", action_space=env.get_action_space())
    continuosUCB = ContinuosUCB(
        name="Continuous UCB", action_space=env.get_action_space())

    agents = [orcale, random_agent,
             ucb, continuosUCB,
             q_agent, continuousQ_agent,
             r_agent_with_trick, continuous_r_agent_with_trick,
             r_agent_without_trick, continuous_r_agent_without_trick,]

    episodes = 2000
    eval_steps = 200
    epochs = 400 
    results = defaultdict(dict)
    for agent in agents:
        # Check if that the agent learning is consistent - for each state what is the chosen action?
        # do all learning agents agree?
        best_action_per_state = defaultdict(list)
        avg_rewards = []
        for i in range(1, epochs+1):
            agent.reset()
            avg_reward = train_single_agent(agent, env, episodes=episodes, eval_steps=eval_steps, seed=i)
            avg_rewards.append(avg_reward)
            for state in agent.q_table:
                actions = agent.q_table[state]
                best_action = max(actions, key=actions.get)
                best_action_per_state[state].append(best_action)
        # Print the best action ratio for each state
        results[agent.name] = \
            {f"Average Reward over {eval_steps} steps": sum(avg_rewards) / len(avg_rewards)}
        print(f"Best action ratio for {agent.name}:")
        for state, actions in best_action_per_state.items():
            best_action_ratio = len([i for i in actions if i == orcale.act(state)]) / len(actions)
            print(f"State {state}: Best Action Ratio: {best_action_ratio}")
            results[agent.name][f"State {state} Best Action Ratio"] = best_action_ratio
        print(f"{agent.name}: Average Reward over {eval_steps} steps: {sum(avg_rewards) / len(avg_rewards)}")
        print("-" * 50)
        print()

    # Print the results
    df = pd.DataFrame(results,)
    print(df.to_string())
    # Save the results to a CSV file with float values up to 3 decimal places
    df.to_csv(f"{name}_results.csv", float_format='%.3f')


if __name__ == '__main__':
    stateless_env = StatelessEnv("Stateless Environment")
    two_state_ed_env = TwoStatesEvenDistEnv("Two States Even Distribution Environment")
    two_state_ued_env = TwoStatesUnevenDistEnv("Two States Uneven Distribution Environment")

    # Run experiments for each environment
    # experiment_runner(stateless_env, name="Stateless Environment Experiment")

    # experiment_runner(two_state_ed_env, name="Two States Even Distribution Environment Experiment")
    experiment_runner(two_state_ued_env, name="Two States Uneven Distribution Environment Experiment")
