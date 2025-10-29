import math
from .base import Agent, MAX_REWARDS

class QLearningAgent(Agent):
    def __init__(self, name: str, action_space=None, learning_rate=0.1, discount_factor=0.99, exploration_rate=0.1):
        super().__init__(name, action_space)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.q_table = {}
        self.policy_changed = False

    def act(self, state):
        if state not in self.q_table:
            self.q_table[state] = {action: MAX_REWARDS for action in self.action_space}
        if self.rng.random() < self.exploration_rate:
            return self.rng.choice(self.action_space)
        return max(self.q_table[state], key=self.q_table[state].get)

    def reset(self):
        self.q_table = {}

    def eval(self, state):
        if state not in self.q_table:
            self.q_table[state] = {action: MAX_REWARDS for action in self.action_space}
        return max(self.q_table[state], key=self.q_table[state].get)

    def learn(self, state, action, reward, next_state, time):
        if next_state not in self.q_table:
            self.q_table[next_state] = {action: MAX_REWARDS for action in self.action_space}
        best_next_action = max(self.q_table[next_state], key=self.q_table[next_state].get)
        td_target = reward + self.discount_factor * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self._check_convergence(state, action, self.learning_rate * td_error)
        self.q_table[state][action] += self.learning_rate * td_error

class ContinuousQLearningAgent(QLearningAgent):
    def __init__(self, name: str, action_space=None, learning_rate=0.1, discount_factor=0.99, exploration_rate=0.1, _lambda=0.01):
        super().__init__(name, action_space, learning_rate, discount_factor, exploration_rate)
        self._lambda = _lambda

    def learn(self, state, action, reward, next_state, time):
        if next_state not in self.q_table:
            self.q_table[next_state] = {action: MAX_REWARDS for action in self.action_space}
        best_next_action = max(self.q_table[next_state], key=self.q_table[next_state].get)
        df = math.exp(-self._lambda * time * self.discount_factor)
        td_target = reward + df * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self._check_convergence(state, action, self.learning_rate * td_error)
        self.q_table[state][action] += self.learning_rate * td_error

class HarmonicQAgent(QLearningAgent):
    def __init__(self, name: str, action_space=None, learning_rate=0.1, discount_factor=0.99, exploration_rate=0.1):
        super().__init__(name, action_space)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.rq_table = {}

    def act(self, state):
        if state not in self.q_table:
            self.q_table[state] = {action: MAX_REWARDS for action in self.action_space}
            self.rq_table[state] = {action: 1.0 for action in self.action_space}
        if self.rng.random() < self.exploration_rate:
            return self.rng.choice(self.action_space)
        return max(self.q_table[state], key=self.q_table[state].get)

    def reset(self):
        super().reset()
        self.rq_table = {}

    def learn(self, state, action, reward, next_state, time):
        if next_state not in self.q_table:
            self.q_table[next_state] = {action: MAX_REWARDS for action in self.action_space}
            self.rq_table[next_state] = {action: 1.0 for action in self.action_space}
        best_next_action = max(self.q_table[next_state], key=self.q_table[next_state].get)
        td_target = time / reward + (self.discount_factor * 1 / (self.rq_table[next_state][best_next_action]))
        td_error = td_target - self.rq_table[state][action]
        self.rq_table[state][action] += self.learning_rate * td_error
        self._check_convergence(state, action, 1 / self.rq_table[state][action], True)
        self.q_table[state][action] = 1 / self.rq_table[state][action]

