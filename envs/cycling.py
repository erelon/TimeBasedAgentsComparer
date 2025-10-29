from .two_states_uneven import TwoStatesUnevenDistEnv

class UnevenCycling(TwoStatesUnevenDistEnv):
    def __init__(self, name: str, _maxp=0.8, _maxv=5, _cycle=5, **kwargs):
        super().__init__(name, _maxp=_maxp, _maxv=_maxv, **kwargs)
        self.cycle = _cycle

    def reset(self):
        super().reset()
        self.clock = 0
        self.cycle_state = 0

    def update_state(self, action):
        super().update_state(action)
        self.clock += 1
        self.cycle_state = (self.clock // self.cycle) % 2

    def get_reward(self, agent, action):
        T = self.rng.uniform(self.interval_min_len, self.interval_max_len)
        if self.cycle_state == 0:
            if self.state == 0:
                reward = self.rng.normalvariate(T * (self.maxp / (1.0 if action == 0 else 9.0)), self.maxv)
            else:
                reward = self.rng.normalvariate(T * (self.maxp / (3.0 if action == 0 else 1.5)), self.maxv)
        else:
            if self.state == 0:
                reward = self.rng.normalvariate(T * (self.maxp / (9.0 if action == 0 else 1.0)), self.maxv)
            else:
                reward = self.rng.normalvariate(T * (self.maxp / (3.0 if action == 0 else 1.5)), self.maxv)
        reward = max(self.interval_min_len, reward)
        reward = min(T, reward)
        self.update_state(action)
        return T, reward

    def secret(self):
        def secret(state):
            if self.cycle_state == 0:
                return 0 if state == 0 else 1
            else:
                return 1
        return secret

class UnevenLatentCycling(TwoStatesUnevenDistEnv):
    def __init__(self, name: str, _maxp=0.8, _maxv=5, _cycle=50, **kwargs):
        super().__init__(name, _maxp=_maxp, _maxv=_maxv, **kwargs)
        self.cycle = _cycle

    def reset(self):
        super().reset()
        self.clock = 0
        self.cycle_state = 0

    def update_state(self, action):
        super().update_state(action)
        self.clock += 1
        self.cycle_state = (self.clock // self.cycle) % 2

    def get_reward(self, agent, action):
        T = self.rng.uniform(self.interval_min_len, self.interval_max_len)
        if self.cycle_state == 0:
            reward = self.rng.normalvariate(T * (self.maxp / (1.0 if action == 0 else 2.0)), self.maxv)
        else:
            reward = self.rng.normalvariate(T * (self.maxp / (3.0 if action == 0 else 1.5)), self.maxv)
        reward = max(self.interval_min_len, reward)
        reward = min(T, reward)
        self.update_state(action)
        return T, reward

    def secret(self):
        return lambda state: 0 if self.cycle_state == 0 else 1

