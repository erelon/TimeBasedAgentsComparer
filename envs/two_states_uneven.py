from .base import AbstractEnvironment


class TwoStatesUnevenDistEnv(AbstractEnvironment):
    def __init__(self, name: str, _maxp=0.8, _maxv=5, **kwargs):
        super().__init__(name, **kwargs)
        self.action_space = [0, 1]
        self.maxp = _maxp
        self.maxv = _maxv

    def update_state(self, action):
        if self.state == 0:
            self.state = 1 if self.rng.random() < 0.2 else 0
        elif self.state == 1 and action == 0:
            self.state = 0 if self.rng.random() < 0.2 else 1
        elif self.state == 1 and action == 1:
            self.state = 0 if self.rng.random() < 0.8 else 1

    def get_reward(self, agent, action):
        T = self.rng.uniform(self.interval_min_len, self.interval_max_len)
        if self.state == 0:
            if action == 0:
                reward = self.rng.normalvariate(T * self.maxp, self.maxv)
            else:
                reward = self.rng.normalvariate(T * (self.maxp / 2.0), self.maxv)
        else:
            if action == 0:
                reward = self.rng.normalvariate(T * (self.maxp / 1.5), self.maxv)
            else:
                reward = self.rng.normalvariate(T * (self.maxp / 3.0), self.maxv)
        reward = max(self.interval_min_len, reward)
        reward = min(T, reward)
        self.update_state(action)
        return T, reward

    def secret(self):
        return lambda state: 0 if state == 0 else 1


class Uneven_wide(TwoStatesUnevenDistEnv):
    def __init__(self, name: str, **kwargs):
        super().__init__(name, _maxp=0.6, **kwargs)


class Uneven_narrow(TwoStatesUnevenDistEnv):
    def __init__(self, name: str, **kwargs):
        super().__init__(name, _maxp=0.2, **kwargs)
