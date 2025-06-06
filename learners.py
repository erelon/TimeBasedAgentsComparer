import random


class Agent:
    """
    Base class for all rl agents in the system.
    """

    def __init__(self, name: str, action_space=None):
        self.name = name
        self.q_table = {}
        self.action_space = action_space if action_space is not None else []

    def __repr__(self):
        return f"Agent(name={self.name})"

    def reset(self):
        """
        Reset the agent's knowledge and state.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def act(self, state):
        """
        Perform an action based on the current state.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def eval(self, state):
        """
        Only exploit the knowledge of the agent.
        """
        pass

    def learn(self, state, action, reward, next_state, time):
        """
        Update the agent's knowledge based on the action taken and the reward received.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")


class QLearningAgent(Agent):
    """
    Q-learning agent that learns from the environment.
    """

    def __init__(self, name: str, action_space=None, learning_rate=0.1, discount_factor=0.99, exploration_rate=0.1):
        super().__init__(name, action_space)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate

    def act(self, state):
        """
        Choose an action based on the current state using an epsilon-greedy strategy.
        """
        if state not in self.q_table:
            self.q_table[state] = {action: 0 for action in self.action_space}

        if random.random() < self.exploration_rate:
            return random.choice(self.action_space)  # Explore
        else:
            return max(self.q_table[state], key=self.q_table[state].get)  # Exploit

    def reset(self):
        """
        Reset the agent's knowledge and state.
        """
        self.q_table = {}

    def eval(self, state):
        """
        Only exploit the knowledge of the agent.
        """
        if state not in self.q_table:
            self.q_table[state] = {action: 0 for action in self.action_space}
        return max(self.q_table[state], key=self.q_table[state].get)

    def learn(self, state, action, reward, next_state, time):
        """
        Update the Q-value based on the action taken and the reward received.
        """
        if next_state not in self.q_table:
            self.q_table[next_state] = {action: 0 for action in self.action_space}

        best_next_action = max(self.q_table[next_state], key=self.q_table[next_state].get)
        td_target = reward + self.discount_factor * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]

        # Update Q-value
        self.q_table[state][action] += self.learning_rate * td_error


class RandomAgent(Agent):
    """
    Random agent that chooses actions randomly.
    """

    def __init__(self, name: str, action_space=None):
        super().__init__(name, action_space)

    def reset(self):
        pass

    def act(self, state):
        """
        Choose a random action from the action space.
        """
        return random.choice(self.action_space)

    def eval(self, state):
        """
        Random agent does not evaluate states.
        """
        return random.choice(self.action_space)

    def learn(self, state, action, reward, next_state, time):
        """
        Random agent does not learn from the environment.
        """
        pass  # No learning mechanism for random agent


class OracleAgent(Agent):
    """
    Oracle agent that provides the optimal action for a given state.
    This is typically used for benchmarking purposes.
    """

    def __init__(self, name: str, action_space=None, env_secret=None):
        super().__init__(name, action_space)
        if env_secret is None:
            raise ValueError("OracleAgent requires an environment secret to provide optimal actions.")
        self.env_secret = env_secret

    def reset(self):
        pass

    def act(self, state):
        """
        Return the optimal action for the given state.
        This method should be overridden by subclasses to provide the actual oracle logic.
        """
        return self.env_secret(state)

    def eval(self, state):
        return self.act(state)  # For oracle, eval is the same as act

    def learn(self, state, action, reward, next_state, time):
        pass  # No learning mechanism for oracle agent


class RLAgent(QLearningAgent):
    """
    Reinforcement Learning Agent based on Schwartz's algorithm.
    https://www.researchgate.net/profile/Anton-Schwartz/publication/221346025_A_Reinforcement_Learning_Method_for_Maximizing_Undiscounted_Rewards/links/5e72421aa6fdcc37caf4cf4b/A-Reinforcement-Learning-Method-for-Maximizing-Undiscounted-Rewards.pdf
    """

    def __init__(self, name: str, action_space=None, learning_rate=0.1, exploration_rate=0.1, with_rho_trick=True):
        super().__init__(name, action_space, learning_rate, exploration_rate=exploration_rate)
        self.rho = 0
        self.with_rho_trick = with_rho_trick

    def reset(self):
        """
        Reset the agent's knowledge and state.
        """
        super().reset()
        self.rho = 0  # Reset rho for RLAgent

    def act(self, state):
        """
        Choose an action based on the current state using an epsilon-greedy strategy.
        """
        return super().act(state)  # Inherits behavior from QLearningAgent

    def learn(self, state, action, reward, next_state, time):
        """
        Update the agent's knowledge based on the action taken and the reward received.
        """
        if next_state not in self.q_table:
            self.q_table[next_state] = {action: 0 for action in self.action_space}

        best_next_action = max(self.q_table[next_state], key=self.q_table[next_state].get)
        self.q_table[state][action] = \
            self.q_table[state][action] + self.learning_rate * (
                    reward - self.rho + self.q_table[next_state][best_next_action] - self.q_table[state][action])

        best_current_action = max(self.q_table[state], key=self.q_table[state].get)

        if self.with_rho_trick and self.q_table[state][action] == self.q_table[state][best_current_action]:
            self.rho = self.rho + self.learning_rate * (
                    reward + self.q_table[next_state][best_next_action] - self.q_table[next_state][
                best_current_action] - self.rho)
        elif not self.with_rho_trick:
            self.rho = self.rho + self.learning_rate * (
                    reward + self.q_table[next_state][best_next_action] - self.q_table[next_state][
                best_current_action] - self.rho)


class ContinuesMAB(Agent):
    """
    Continues Multi-Armed Bandit agent that learns from continuous rewards.
    """

    def __init__(self, name: str, action_space=None, learning_rate=0.1, exploration_rate=0.1):
        super().__init__(name, action_space)
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.total_time = {}
        self.total_reward = {}

    def reset(self):
        """
        Reset the agent's knowledge and state.
        """
        self.q_table = {}
        self.total_time = {}
        self.total_reward = {}

    def act(self, state):
        """
        Choose an action based on the current state using an epsilon-greedy strategy.
        """
        if state not in self.q_table:
            self.q_table[state] = {action: 0 for action in self.action_space}

        if random.random() < self.exploration_rate:
            return random.choice(self.action_space)  # Explore
        else:
            return max(self.q_table[state], key=self.q_table[state].get)  # Exploit

    def eval(self, state):
        """
        Choose an action based on the current state using an epsilon-greedy strategy.
        """
        if state not in self.q_table:
            self.q_table[state] = {action: 0 for action in self.action_space}

        return max(self.q_table[state], key=self.q_table[state].get)  # Exploit

    def learn(self, state, action, reward, next_state, time):
        """
        Update the Q-value based on the action taken and the reward received.
        """
        if state not in self.q_table:
            self.q_table[state] = {action: 0 for action in self.action_space}
        if state not in self.total_time:
            self.total_time[state] = {action: 0 for action in self.action_space}
            self.total_reward[state] = {action: 0 for action in self.action_space}

        self.total_time[state][action] += time
        self.total_reward[state][action] += reward
        if self.total_time[state][action] == 0:
            return
        self.q_table[state][action] = self.total_reward[state][action] / self.total_time[state][action]


class MAB(Agent):
    """
    Multi-Armed Bandit agent that learns from the rewards.
    """

    def __init__(self, name: str, action_space=None, learning_rate=0.1, exploration_rate=0.1):
        super().__init__(name, action_space)
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.total_steps = {}
        self.total_reward = {}

    def reset(self):
        """
        Reset the agent's knowledge and state.
        """
        self.q_table = {}
        self.total_steps = {}
        self.total_reward = {}

    def act(self, state):
        """
        Choose an action based on the current state using an epsilon-greedy strategy.
        """
        if state not in self.q_table:
            self.q_table[state] = {action: 0 for action in self.action_space}

        if random.random() < self.exploration_rate:
            return random.choice(self.action_space)  # Explore
        else:
            return max(self.q_table[state], key=self.q_table[state].get)  # Exploit

    def eval(self, state):
        """
        Choose an action based on the current state using an epsilon-greedy strategy.
        """
        if state not in self.q_table:
            self.q_table[state] = {action: 0 for action in self.action_space}

        return max(self.q_table[state], key=self.q_table[state].get)  # Exploit

    def learn(self, state, action, reward, next_state, time):
        """
        Update the Q-value based on the action taken and the reward received.
        """
        if state not in self.q_table:
            self.q_table[state] = {action: 0 for action in self.action_space}
        if state not in self.total_steps:
            self.total_steps[state] = {action: 0 for action in self.action_space}
            self.total_reward[state] = {action: 0 for action in self.action_space}

        self.total_steps[state][action] += 1
        self.total_reward[state][action] += reward
        if self.total_steps[state][action] == 0:
            return
        self.q_table[state][action] = self.total_reward[state][action] / self.total_steps[state][action]
