try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # optional dependency guard
    raise ImportError("gymnasium is required for SMDPEnvWrapper. Install via 'pip install gymnasium'.")

try:
    from envs.base import AbstractEnvironment
except ImportError:
    from .envs.base import AbstractEnvironment

class SMDPEnvWrapper(gym.Env):
    """Wraps an AbstractEnvironment to Gym API, exposing durations via info."""
    metadata = {"render_modes": []}

    def __init__(self, env: AbstractEnvironment):
        super().__init__()
        self.env = env
        # Assume discrete action space
        self.action_space = spaces.Discrete(len(self.env.get_action_space()))
        # Observation: integer state -> use Discrete
        self.observation_space = spaces.Discrete(1000)  # arbitrary upper bound

    def reset(self, *, seed=None, options=None):
        if seed is not None and hasattr(self.env, 'set_seed'):
            self.env.set_seed(seed)
        state = self.env.reset()
        info = {}
        return state, info

    def step(self, action: int):
        state = self.env.get_state()
        duration, reward = self.env.get_reward(None, action)
        next_state = self.env.get_state()
        terminated = False  # episodic termination not defined
        truncated = False
        info = {"duration": duration, "raw_reward": reward}
        return next_state, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass
