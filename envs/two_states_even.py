from .base import AbstractEnvironment

class TwoStatesEvenDistEnv(AbstractEnvironment):
    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        self.action_space = [0, 1]

    def update_state(self, action):
        self.state = self.rng.choice([0, 1])

    def get_reward(self, agent, action):
        T = self.rng.uniform(self.interval_min_len, self.interval_max_len)
        if self.state == 0:
            reward = 1 + self.interval_min_len if action == 0 else self.interval_min_len
        else:
            reward = self.interval_min_len if action == 0 else 1 + self.interval_min_len
        reward = max(self.interval_min_len, reward)
        reward = min(T, reward)
        self.update_state(action)
        return T, reward

    def secret(self):
        return lambda state: 0 if state == 0 else 1

