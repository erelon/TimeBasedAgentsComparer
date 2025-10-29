import random
from typing import Any, Callable, List, Optional

class AbstractEnvironment:
    """Base class for SMDP-style environments with variable time per transition."""
    def __init__(self, name: str, seed: int = 42, interval_min_len: int = 1, interval_max_len: int = 500):
        self.name = name
        self.action_space: Optional[List[int]] = None
        self.seed = seed
        self.interval_min_len = interval_min_len
        self.interval_max_len = interval_max_len
        self.set_seed(seed)
        self.reset()

    def reset(self) -> Any:
        self.rng = random.Random(self.seed)
        self.rng.seed(self.seed)
        self.state = 0
        return self.get_state()

    def set_seed(self, seed: int):
        self.seed = seed

    def get_name(self) -> str:
        return self.name

    def get_state(self, agent=None) -> Any:
        return self.state

    def get_action_space(self):
        return self.action_space

    def update_state(self, action=None):
        raise NotImplementedError

    def get_reward(self, agent, action, state=None):  # returns (duration, reward)
        raise NotImplementedError

    def secret(self) -> Callable[[Any], int]:
        raise NotImplementedError

    def __str__(self) -> str:
        return f"Environment(name={self.name})"

    def __repr__(self) -> str:
        return f"AbstractEnvironment(name={self.name})"

