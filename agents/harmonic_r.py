import sys
from .r_learning import RLAgent


class HarmonicRLAgent(RLAgent):
    def __init__(self, name: str, action_space=None, learning_rate=0.1, exploration_rate=0.1, with_rho_trick=True,
                 rho_learning_rate=0.3):
        super().__init__(name, action_space, learning_rate, exploration_rate, with_rho_trick, rho_learning_rate)
        self.reciprocal_rho = 1.0
        self.total_time = 0
        self.total_reward = 0

    def reset(self):
        super().reset()
        self.pos_reciprocal_rho = 0.0
        self.neg_reciprocal_rho = 0.0
        self.neg_w = 1
        self.pos_w = 1
        self.zero_w = 1
        self.total_time = 0
        self.total_reward = 0

    def learn(self, state, action, reward, next_state, time):
        if next_state not in self.q_table:
            self.q_table[next_state] = {action: 0 for action in self.action_space}
        best_next_action = max(self.q_table[next_state], key=self.q_table[next_state].get)
        best_current_action = max(self.q_table[state], key=self.q_table[state].get)
        deltarho = reward - self.rho * time
        delta = deltarho + self.q_table[next_state][best_next_action] - self.q_table[state][action]
        self._check_convergence(state, action, self.learning_rate * delta)
        self.q_table[state][action] += self.learning_rate * delta
        if not self.with_rho_trick or (self.with_rho_trick and action == best_current_action):
            pos = 1 if reward > 0 else 0
            neg = 1 if reward < 0 else 0
            zero = 1 if reward == 0 else 0

            # rho = rho + alpha * (time - reward * rho)
            # equivalent to: rho = (1 - (reward * alpha)) * rho + alpha * time

            #reciprocal_rate = 0 if zero == 1 else time / reward
            self.pos_reciprocal_rho += self.rho_learning_rate * (time - reward * self.pos_reciprocal_rho) * pos
            self.pos_w = (1 - self.rho_learning_rate) * self.pos_w + self.rho_learning_rate * pos
            H_pos = 0 if self.pos_reciprocal_rho == 0 else self.pos_w / self.pos_reciprocal_rho

            self.neg_reciprocal_rho += self.rho_learning_rate * (time - reward * self.neg_reciprocal_rho) * neg
            self.neg_w = (1 - self.rho_learning_rate) * self.neg_w + self.rho_learning_rate * neg
            H_neg = 0 if self.neg_reciprocal_rho == 0 else self.neg_w / self.neg_reciprocal_rho

            self.zero_w = (1 - self.rho_learning_rate) * self.zero_w + self.rho_learning_rate * zero

            self.rho = (H_pos * self.pos_w + H_neg * self.neg_w) / (self.pos_w + self.neg_w + self.zero_w)


class HarmonicROLAgent(RLAgent):
    def __init__(self, name: str, action_space=None, learning_rate=0.1, exploration_rate=0.1, with_rho_trick=True,
                 rho_learning_rate=0.3):
        super().__init__(name, action_space, learning_rate, exploration_rate, with_rho_trick, rho_learning_rate)
        self.reciprocal_rho = 1.0
        self.total_time = 0
        self.total_reward = 0

    def reset(self):
        super().reset()
        self.pos_reciprocal_rho = 0.0
        self.neg_reciprocal_rho = 0.0
        self.neg_w = 1
        self.pos_w = 1
        self.zero_w = 1
        self.total_time = 0
        self.total_reward = 0

    def learn(self, state, action, reward, next_state, time):
        if next_state not in self.q_table:
            self.q_table[next_state] = {action: 0 for action in self.action_space}
        best_next_action = max(self.q_table[next_state], key=self.q_table[next_state].get)
        best_current_action = max(self.q_table[state], key=self.q_table[state].get)
        deltarho = reward - self.rho * time
        delta = deltarho + self.q_table[next_state][best_next_action] - self.q_table[state][action]
        self._check_convergence(state, action, self.learning_rate * delta)
        self.q_table[state][action] += self.learning_rate * delta
        if not self.with_rho_trick or (self.with_rho_trick and action == best_current_action):
            pos = 1 if reward > 0 else 0
            neg = 1 if reward < 0 else 0
            zero = 1 if reward == 0 else 0
            reciprocal_rate = 0 if zero == 1 else time / reward
            self.pos_reciprocal_rho = (
                                                  1 - self.rho_learning_rate) * self.pos_reciprocal_rho + self.rho_learning_rate * reciprocal_rate * pos
            self.pos_w = (1 - self.rho_learning_rate) * self.pos_w + self.rho_learning_rate * pos
            H_pos = 0 if self.pos_reciprocal_rho == 0 else self.pos_w / self.pos_reciprocal_rho
            self.neg_reciprocal_rho = (
                                                  1 - self.rho_learning_rate) * self.neg_reciprocal_rho + self.rho_learning_rate * reciprocal_rate * neg
            self.neg_w = (1 - self.rho_learning_rate) * self.neg_w + self.rho_learning_rate * neg
            H_neg = 0 if self.neg_reciprocal_rho == 0 else self.neg_w / self.neg_reciprocal_rho
            self.zero_w = (1 - self.rho_learning_rate) * self.zero_w + self.rho_learning_rate * zero
            self.rho = (H_pos * self.pos_w + H_neg * self.neg_w) / (self.pos_w + self.neg_w + self.zero_w)


class HarmonicRLAgent2(RLAgent):
    def __init__(self, name: str, action_space=None, learning_rate=0.1, exploration_rate=0.1, with_rho_trick=True,
                 rho_learning_rate=0.3):
        super().__init__(name, action_space, learning_rate, exploration_rate, with_rho_trick, rho_learning_rate)
        self.rq_table = {}
        self.reciprocal_rho = 1.0
        self.total_time = 0
        self.total_reward = 0

    def reset(self):
        super().reset()
        self.rq_table = {}
        self.reciprocal_rho = 1.0
        self.total_time = 0
        self.total_reward = 0

    def learn(self, state, action, reward, next_state, time):
        if next_state not in self.q_table or next_state not in self.rq_table:
            self.q_table[next_state] = {action: 0 for action in self.action_space}
            self.rq_table[next_state] = {action: 0 for action in self.action_space}
        if state not in self.q_table or state not in self.rq_table:
            self.q_table[state] = {action: 0 for action in self.action_space}
            self.rq_table[state] = {action: 0 for action in self.action_space}
        best_next_action = max(self.q_table[next_state], key=self.q_table[next_state].get)
        best_current_action = max(self.q_table[state], key=self.q_table[state].get)
        recip_td_target = (time / reward) - self.reciprocal_rho + (self.rq_table[next_state][best_next_action])
        recip_td_error = recip_td_target - self.rq_table[state][action]
        self.rq_table[state][action] += self.learning_rate * recip_td_error
        self._check_convergence(state, action, 1 / self.rq_table[state][action], True)
        self.q_table[state][action] = 1 / self.rq_table[state][action]
        if not self.with_rho_trick or (self.with_rho_trick and action == best_current_action):
            self.reciprocal_rho = (1 - self.rho_learning_rate) * self.reciprocal_rho + self.rho_learning_rate * (
                recip_td_error)
            self.rho = 1 / self.reciprocal_rho
            self.total_time += time
            self.total_reward += reward


class AdaptiveHarmonicRLAgent2(RLAgent):
    def __init__(self, name: str, action_space=None, learning_rate=0.1, exploration_rate=0.1, with_rho_trick=True,
                 rho_learning_rate=0.3):
        super().__init__(name, action_space, learning_rate, exploration_rate, with_rho_trick, rho_learning_rate)
        self.rq_table = {}
        self.reciprocal_rho = 1.0
        self.error_scale = 1.0
        self.error_scale_learning_rate = 0.1
        self.total_time = 0
        self.total_reward = 0

    def reset(self):
        super().reset()
        self.rq_table = {}
        self.reciprocal_rho = 1.0
        self.error_scale = 1.0
        self.total_time = 0
        self.total_reward = 0

    def learn(self, state, action, reward, next_state, time):
        if next_state not in self.q_table or next_state not in self.rq_table:
            self.q_table[next_state] = {action: 0 for action in self.action_space}
            self.rq_table[next_state] = {action: 0 for action in self.action_space}
        if state not in self.q_table or state not in self.rq_table:
            self.q_table[state] = {action: 0 for action in self.action_space}
            self.rq_table[state] = {action: 0 for action in self.action_space}
        best_next_action = max(self.q_table[next_state], key=self.q_table[next_state].get)
        best_current_action = max(self.q_table[state], key=self.q_table[state].get)
        recip_td_target = (time / reward) - self.reciprocal_rho + (self.rq_table[next_state][best_next_action])
        recip_td_error = recip_td_target - self.rq_table[state][action]
        rdeltarho = abs(time / reward - self.reciprocal_rho)
        z = 1.0 - ((self.reciprocal_rho) / ((rdeltarho + self.reciprocal_rho)))
        alpha_min, alpha_max = 0.001, 0.8
        alpha = alpha_min + (alpha_max - alpha_min) * (z / (z + 0.1))
        self.rq_table[state][action] += alpha * recip_td_error
        self._check_convergence(state, action, 1 / self.rq_table[state][action], True)
        self.q_table[state][action] = 1 / self.rq_table[state][action]
        if not self.with_rho_trick or (self.with_rho_trick and action == best_current_action):
            self.reciprocal_rho = (1 - self.rho_learning_rate) * self.reciprocal_rho + self.rho_learning_rate * (
                recip_td_error)
            self.rho = 1 / self.reciprocal_rho
            self.total_time += time
            self.total_reward += reward


class AdaptiveHarmonicRLAgent(RLAgent):
    def __init__(self, name: str, action_space=None, learning_rate=0.1, exploration_rate=0.1, with_rho_trick=True,
                 rho_learning_rate=0.3):
        super().__init__(name, action_space, learning_rate, exploration_rate, with_rho_trick, rho_learning_rate)
        self.reciprocal_rho = 1.0
        self.total_time = 0
        self.total_reward = 0

    def reset(self):
        super().reset()
        self.reciprocal_rho = 1.0
        self.total_time = 0
        self.total_reward = 0

    def learn(self, state, action, reward, next_state, time):
        if next_state not in self.q_table:
            self.q_table[next_state] = {action: 0 for action in self.action_space}
        best_next_action = max(self.q_table[next_state], key=self.q_table[next_state].get)
        best_current_action = max(self.q_table[state], key=self.q_table[state].get)
        deltarho = reward - self.rho * time
        delta = deltarho + self.q_table[next_state][best_next_action] - self.q_table[state][action]
        alpha = 1.00001 - reward / ((abs(deltarho) + reward))
        self._check_convergence(state, action, alpha * delta)
        self.q_table[state][action] += alpha * delta
        if not self.with_rho_trick or (self.with_rho_trick and action == best_current_action):
            self.reciprocal_rho = (1 - self.rho_learning_rate) * self.reciprocal_rho + self.rho_learning_rate * (
                        time / (reward))
            self.rho = 1 / self.reciprocal_rho
            self.total_time += time
            self.total_reward += reward
