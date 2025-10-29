from .r_learning import RLAgent

class SMARTRLAgent(RLAgent):
    def __init__(self, name: str, action_space=None, learning_rate=0.1, exploration_rate=0.1, with_rho_trick=True, rho_learning_rate=0.3):
        super().__init__(name, action_space, learning_rate, exploration_rate, with_rho_trick, rho_learning_rate)
        self.total_time = 0
        self.total_reward = 0

    def learn(self, state, action, reward, next_state, time):
        if next_state not in self.q_table:
            self.q_table[next_state] = {action: 0 for action in self.action_space}
        if state not in self.q_table:
            self.q_table[state] = {action: 0 for action in self.action_space}
        best_next_action = max(self.q_table[next_state], key=self.q_table[next_state].get)
        best_current_action = max(self.q_table[state], key=self.q_table[state].get)
        deltarho = reward - self.rho * time
        delta = deltarho + self.q_table[next_state][best_next_action] - self.q_table[state][action]
        self._check_convergence(state, action, self.learning_rate * delta)
        self.q_table[state][action] += self.learning_rate * delta
        if not self.with_rho_trick or (self.with_rho_trick and action == best_current_action):
            self.total_time += time
            self.total_reward += reward
            self.rho = self.total_reward / self.total_time

class StateSMARTRLAgent(RLAgent):
    def __init__(self, name: str, action_space=None, learning_rate=0.1, exploration_rate=0.1, with_rho_trick=True, rho_learning_rate=0.03):
        super().__init__(name, action_space, learning_rate, exploration_rate, with_rho_trick, rho_learning_rate)
        self.total_time = 0
        self.total_reward = 0
        self.time = {}
        self.reward = {}

    def learn(self, state, action, reward, next_state, time):
        if next_state not in self.q_table:
            self.q_table[next_state] = {action: reward for action in self.action_space}
        if next_state not in self.time:
            self.time[next_state] = 0
            self.reward[next_state] = 0
        if state not in self.q_table:
            self.q_table[state] = {action: reward for action in self.action_space}
        if state not in self.time:
            self.time[state] = 0
            self.reward[state] = 0
        best_next_action = max(self.q_table[next_state], key=self.q_table[next_state].get)
        best_current_action = max(self.q_table[state], key=self.q_table[state].get)
        rho = (self.reward[state] / self.time[state]) if self.time[state] != 0 else 1
        delta = reward - rho * time
        new_q_state = delta + self.q_table[next_state][best_next_action]
        self._check_convergence(state, action, self.learning_rate * (new_q_state - self.q_table[state][action]))
        self.q_table[state][action] += self.learning_rate * (new_q_state - self.q_table[state][action])
        if not self.with_rho_trick or (self.with_rho_trick and action == best_current_action):
            self.reward[state] += reward
            self.time[state] += time
            self.total_time += time
            self.total_reward += reward
            self.rho = self.total_reward / self.total_time

class AdaptiveSMARTRLAgent(RLAgent):
    def __init__(self, name: str, action_space=None, learning_rate=0.1, exploration_rate=0.1, with_rho_trick=True, rho_learning_rate=0.3):
        super().__init__(name, action_space, learning_rate, exploration_rate, with_rho_trick, rho_learning_rate)
        self.total_time = 0
        self.total_reward = 0

    def learn(self, state, action, reward, next_state, time):
        if next_state not in self.q_table:
            self.q_table[next_state] = {action: 0 for action in self.action_space}
        if state not in self.q_table:
            self.q_table[state] = {action: 0 for action in self.action_space}
        best_next_action = max(self.q_table[next_state], key=self.q_table[next_state].get)
        best_current_action = max(self.q_table[state], key=self.q_table[state].get)
        deltarho = reward - self.rho * time
        delta = deltarho + self.q_table[next_state][best_next_action] - self.q_table[state][action]
        alpha = 1.00001 - (self.rho * time) / ((abs(deltarho) + self.rho * time))
        self._check_convergence(state, action, alpha * delta)
        self.q_table[state][action] += alpha * delta
        if not self.with_rho_trick or (self.with_rho_trick and action == best_current_action):
            self.total_time += time
            self.total_reward += reward
            self.rho = self.total_reward / self.total_time

