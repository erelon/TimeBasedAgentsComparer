from .q_learning import QLearningAgent

class RLAgent(QLearningAgent):
    def __init__(self, name: str, action_space=None, learning_rate=0.2, exploration_rate=0.1, with_rho_trick=True, rho_learning_rate=0.03):
        super().__init__(name, action_space, learning_rate, exploration_rate=exploration_rate)
        self.rho = 0
        self.with_rho_trick = with_rho_trick
        self.rho_learning_rate = rho_learning_rate

    def reset(self):
        super().reset()
        self.rho = 0

    def learn(self, state, action, reward, next_state, time):
        if next_state not in self.q_table:
            self.q_table[next_state] = {action: 0 for action in self.action_space}
        best_current_action = max(self.q_table[state], key=self.q_table[state].get)
        best_next_action = max(self.q_table[next_state], key=self.q_table[next_state].get)
        delta = reward - self.rho + self.q_table[next_state][best_next_action] - self.q_table[state][action]
        self._check_convergence(state, action, self.learning_rate * delta)
        self.q_table[state][action] += self.learning_rate * delta
        if not self.with_rho_trick or (self.with_rho_trick and action == best_current_action):
            self.rho += self.rho_learning_rate * delta

class ContinuousRLAgent(RLAgent):
    def __init__(self, name: str, action_space=None, learning_rate=0.2, exploration_rate=0.1, with_rho_trick=True, rho_learning_rate=0.03):
        super().__init__(name, action_space, learning_rate, exploration_rate, with_rho_trick, rho_learning_rate)
        self.total_time = 0
        self.total_reward = 0

    def learn(self, state, action, reward, next_state, time):
        if next_state not in self.q_table:
            self.q_table[next_state] = {action: 0 for action in self.action_space}
        self.total_time += time
        self.total_reward += reward
        best_next_action = max(self.q_table[next_state], key=self.q_table[next_state].get)
        best_current_action = max(self.q_table[state], key=self.q_table[state].get)
        delta = reward - self.rho * time + self.q_table[next_state][best_next_action] - self.q_table[state][action]
        self._check_convergence(state, action, self.learning_rate * delta)
        self.q_table[state][action] += self.learning_rate * delta
        if not self.with_rho_trick or (self.with_rho_trick and action == best_current_action):
            self.rho += self.rho_learning_rate * (delta / time)

class MyopicRLearn(RLAgent):
    def __init__(self, name: str, action_space=None, learning_rate=0.1, exploration_rate=0.1, with_rho_trick=True, rho_learning_rate=0.3):
        super().__init__(name, action_space, learning_rate, exploration_rate, with_rho_trick, rho_learning_rate)

    def learn(self, state, action, reward, next_state, time):
        if next_state not in self.q_table:
            self.q_table[next_state] = {action: 0 for action in self.action_space}
        best_current_action = max(self.q_table[state], key=self.q_table[state].get)
        delta = reward - self.rho
        self._check_convergence(state, action, self.learning_rate * delta + (1 - self.learning_rate) * self.q_table[state][action], True)
        self.q_table[state][action] = self.learning_rate * delta + (1 - self.learning_rate) * self.q_table[state][action]
        if not self.with_rho_trick or (self.with_rho_trick and action == best_current_action):
            self.rho = (1 - self.rho_learning_rate) * self.rho + self.rho_learning_rate * reward

