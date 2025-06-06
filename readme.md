# Project Overview

This project implements and evaluates various reinforcement learning agents in a stateless environment. The agents are
trained to maximize rewards by interacting with the environment, and their performance is compared against an oracle
agent.

## How to Run

1. **Clone the Repository**:
   Clone the repository to your local machine.

   ```bash
   git clone https://github.com/erelon/TimeBasedAgentsComparer
   cd TimeBasedAgentsComparer
   ```

2. **Install Dependencies**:
   Ensure you have Python 3.8+ installed. No dependencies at this stage.

3. **Run the Main Script**:
   Execute the `main.py` file to train and evaluate the agents.

   ```bash
   python main.py
   ```

   The script will train each agent, evaluate their performance, and print the results, including the best action ratios
   and average rewards.
   "Best action ratio" refers to the proportion of times the agent converged on the optimal action as determined by the
   oracle (for learning algorithms only).
   "Average reward" is the mean reward received by the agent over the evaluation period.

---

## Learners

The project includes the following types of agents:

1. **`QLearningAgent`**:
    - Implements the Q-learning algorithm.
    - Uses an epsilon-greedy strategy for exploration and exploitation.
    - Updates Q-values based on the Bellman equation.

2. **`RandomAgent`**:
    - Chooses actions randomly from the action space.
    - Does not learn or update its knowledge.

3. **`OracleAgent`**:
    - Acts as a benchmark by always selecting the optimal action for a given state.
    - Requires access to the environment's secret to determine the best action.

4. **`RLAgent`**:
    - A reinforcement learning agent based
      on [Schwartz's algorithm](https://www.researchgate.net/profile/Anton-Schwartz/publication/221346025_A_Reinforcement_Learning_Method_for_Maximizing_Undiscounted_Rewards/links/5e72421aa6fdcc37caf4cf4b/A-Reinforcement-Learning-Method-for-Maximizing-Undiscounted-Rewards.pdf).
    - Includes an optional disabling of the "rho trick" so that the rho will update every time.

5. **`ContinuesMAB`**:
    - A continuous multi-armed bandit agent.
    - Learns from continuous rewards and updates Q-values based on the average reward per action according to time.

6. **`MAB`**:
    - A multi-armed bandit agent.
    - Learns from discrete rewards and updates Q-values based on the average reward per action according to steps.

---

## Stateless Environment

The environment (`StatelessEnv`) is a simple, stateless environment where:

- **State**: The state is always `0`, meaning the environment does not maintain any history.
- **Action Space**: The agent can choose between two actions: `0` and `1`.
- **Rewards**:
    - Action `0`: Generates a reward based on a normal distribution with a mean of `5`.
    - Action `1`: Generates a reward based on a normal distribution with a mean of `6`.
- **Oracle**: The oracle always selects action `0` as the optimal action.

This environment is designed to test the agents' ability to learn and exploit the reward structure without relying on
state transitions.

## Current output
```
Best action ratio for Oracle Agent:
Oracle Agent: Average Reward over 100 steps: 3.00937080713335
--------------------------------------------------

Best action ratio for Random Agent:
Random Agent: Average Reward over 100 steps: 2.5074142216498756
--------------------------------------------------

Best action ratio for QLearning Agent:
State 0: Best Action Ratio: 0.898989898989899
QLearning Agent: Average Reward over 100 steps: 2.909443733202572
--------------------------------------------------

Best action ratio for RL Agent with trick:
State 0: Best Action Ratio: 0.9696969696969697
RL Agent with trick: Average Reward over 100 steps: 2.9796628519887505
--------------------------------------------------

Best action ratio for RL Agent without trick:
State 0: Best Action Ratio: 1.0
RL Agent without trick: Average Reward over 100 steps: 3.00937080713335
--------------------------------------------------

Best action ratio for Continues MAB:
State 0: Best Action Ratio: 0.5353535353535354
Continues MAB: Average Reward over 100 steps: 2.548829239704895
--------------------------------------------------

Best action ratio for MAB:
State 0: Best Action Ratio: 1.0
MAB: Average Reward over 100 steps: 3.00937080713335
--------------------------------------------------
```
