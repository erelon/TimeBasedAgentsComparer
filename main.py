import sys
from collections import defaultdict

import pandas as pd

from env import *
from learners import *


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

    env.set_seed(seed+1)
    env.reset()
    rewards = []
    for _ in range(eval_steps):
        state = env.get_state()
        action = agent.eval(state)
        time, reward = env.get_reward(agent, action)
        rewards.append(reward)

    return sum(rewards) / len(rewards)


def experiment_runner(env, name="Experiment"):
    print(f"Running experiment with env: {name}")

    oracle = OracleAgent(name="Oracle Agent",  action_space=env.get_action_space(), env_secret=env.secret(),   )

    q_agent = QLearningAgent(name="QLearning", action_space=env.get_action_space())
    continuousQ_agent = ContinuousQLearningAgent(
        name="Cont. QLearning", action_space=env.get_action_space()
    )
    random_agent = RandomAgent(name="Random Agent", action_space=env.get_action_space())
    r_agent_with_trick = RLAgent(
        name="RLearning (update on policy)", action_space=env.get_action_space()
    )
    r_agent_without_trick = RLAgent(
        name="RLearning (update always)",
        action_space=env.get_action_space(),
        with_rho_trick=False,
    )
    continuous_r_agent_with_trick = ContinuousRLAgent(
        name="Cont. RLearning (update on-policy)", action_space=env.get_action_space()
    )
    continuous_r_agent_without_trick = ContinuousRLAgent(
        name="Cont. RLearning (update always)",
        action_space=env.get_action_space(),
        with_rho_trick=False,
    )

    smart_r_agent_with_trick = SMARTRLAgent(name="SMART (update on policy)",
                                            action_space=env.get_action_space())
    smart_r_agent_without_trick = SMARTRLAgent(name="SMART (update always)",
                                            action_space=env.get_action_space(), with_rho_trick=False)

    mab = MAB(name="MAB", action_space=env.get_action_space())
    c_mab = ContinuesMAB(name="Continues MAB", action_space=env.get_action_space())

    ucb = UCB(name="UCB", action_space=env.get_action_space())
    continuosUCB = ContinuosUCB(
        name="Continuous UCB", action_space=env.get_action_space()
    )

    harmonic_agent_with_trick = HarmonicRLAgent(name="harmonic (update on policy)",
                                            action_space=env.get_action_space())
    harmonic_agent_without_trick = HarmonicRLAgent(name="harmonic (update always)",
                                            action_space=env.get_action_space(), with_rho_trick=False)

    harmonicq_agent = HarmonicQAgent(name="harmonic Q", action_space=env.get_action_space())

    myopic_agent_without =MyopicRLearn(name="myopic R", action_space=env.get_action_space(), with_rho_trick=False)
    myopic_agent_with =MyopicRLearn(name="myopic R (upd on policy)", action_space=env.get_action_space())

    statesmart_agent_with = StateSMARTRLAgent(name="State SMART (update on policy)", action_space=env.get_action_space())
    statesmart_agent_without = StateSMARTRLAgent(name="State SMART (update always)", action_space=env.get_action_space(), with_rho_trick=False)

    harmonic2_with = HarmonicRLAgent2(name="Harmonic2 (update on policy)", action_space=env.get_action_space())
    harmonic2_without = HarmonicRLAgent2(name="Harmonic2 (update always)", action_space=env.get_action_space(), with_rho_trick=False)

    print("Got here")
    agents = [
        oracle,
        random_agent,
        # ucb,
        # continuosUCB,
         # q_agent,
         # continuousQ_agent,
         harmonicq_agent,
        # r_agent_with_trick,
        # continuous_r_agent_with_trick,
        # r_agent_without_trick,
        # continuous_r_agent_without_trick,
        smart_r_agent_with_trick,
        # smart_r_agent_without_trick,
        harmonic_agent_with_trick,
        # harmonic_agent_without_trick,
        # myopic_agent_with,
        # myopic_agent_without,
        # statesmart_agent_with,
        # statesmart_agent_without,
        harmonic2_with,
        harmonic2_without,
    ]

    episodes = 5000
    eval_steps = 200
    epochs =50 
    results = defaultdict(dict)
    for agent in agents:
        print(f"Agent: {agent.name}", file=sys.stderr)
        # Check if that the agent learning is consistent - for each state what is the chosen action?
        # do all learning agents agree?
        best_action_per_state = defaultdict(list)
        avg_rewards = []
        for i in range(1, epochs + 1):
            agent.reset()
            avg_reward = train_single_agent(
                agent, env, episodes=episodes, eval_steps=eval_steps, seed=i
            )
            avg_rewards.append(avg_reward)
            for state in agent.q_table:
                actions = agent.q_table[state]
                best_action = max(actions, key=actions.get)
                best_action_per_state[state].append(best_action)
        # Print the best action ratio for each state
        results[agent.name] = {
            f"Average Reward over {eval_steps} steps": sum(avg_rewards)
            / len(avg_rewards)
        }
        print(f"Best action ratio for {agent.name}:")
        for state, actions in best_action_per_state.items():
            best_action_ratio = len(
                [i for i in actions if i == oracle.act(state)]
            ) / len(actions)
            print(f"State {state}: Best Action Ratio: {best_action_ratio}")
            results[agent.name][f"State {state} Best Action Ratio"] = best_action_ratio
        print(
            f"{agent.name}: Average Reward over {eval_steps} steps: {sum(avg_rewards) / len(avg_rewards)}"
        )
        print("-" * 50)
        print()

    # Print the results
    df = pd.DataFrame(
        results,
    )
    print(df.to_string())
    # Save the results to a CSV file with float values up to 3 decimal places
    df.to_csv(f"{name}_results.csv", float_format="%.3f")


if __name__ == "__main__":
    stateless_env = StatelessEnv("Stateless Environment")
    two_state_ed_env = TwoStatesEvenDistEnv("Two States Even Distribution Environment")
    two_state_ued_wide = Uneven_wide("Two States Uneven Distribution (wide range)")
    two_state_ued_narrow = Uneven_narrow("Two States Uneven Distribution (narrow range)")
    two_state_ued_sin = TwoStatesUnevenDistSinEnv("Two States Uneven Distribution Sin")

    # Run experiments for each environment
    # experiment_runner(stateless_env, name="Stateless Environment Experiment")

    # experiment_runner(two_state_ed_env, name="Two States Even Distribution Environment Experiment")
    # env =  stateless_env
    # env =   two_state_ed_env
    env =   two_state_ued_wide
    # env =   two_state_ued_narrow
    env = two_state_ued_sin

    experiment_runner(env, name=env.name+" Experiment")

