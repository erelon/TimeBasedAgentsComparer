from .base import AbstractEnvironment

class StatelessEnv(AbstractEnvironment):
    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        self.action_space = [0, 1]

    def get_reward(self, agent, action):
        T = self.rng.uniform(self.interval_min_len, self.interval_max_len)
        if action == 1:
            reward = self.rng.normalvariate(T * 0.55, 100)
        elif action == 0:
            reward = self.rng.normalvariate(T * 0.6, 100)
        else:
            raise ValueError("Invalid action")
        reward = max(self.interval_min_len, reward)
        reward = min(T, reward)
        self.update_state()
        return T, reward

    def update_state(self, action=None):
        self.state = 0

    def secret(self):
        return lambda state: 0

