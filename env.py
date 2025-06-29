import random

# All common env parameters are set in AbstractEnvironment __init__
# in particular:
#  interval min and max durations


class AbstractEnvironment:
    def __init__(self, name: str, seed: int = None):
        self.name = name
        self.action_space = None
        # we want to make sure the random seed is set for reproducibility
        self.seed = seed if seed is not None else 42
        self.rng = random.Random(
            seed if seed is not None else 42
        )  # Initialize the random seed

        self.interval_min_len = 1
        self.interval_max_len = 500
        self.reset()

    def reset(self):
        """
        Reset the environment to its initial state.
        """
        # we need to reset the random seed for reproducibility
        self.rng.seed(self.seed)

    def set_seed(self, seed: int):
        """
        Set the random seed for the environment.
        """
        self.seed = seed
        self.reset()

    def get_name(self) -> str:
        return self.name

    def get_state(self, agent=None):
        """
        Get the current state of the environment.
        In a stateless environment, the state is always 0.
        """
        return self.state

    def get_action_space(self):
        return self.action_space

    def update_state(self, action=None):
        """
        Update the state of the environment.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def get_reward(self, agent, action, state):
        """
        Get the reward for the given action.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def __str__(self) -> str:
        return f"Environment(name={self.name})"

    def __repr__(self) -> str:
        return f"AbstractEnvironment(name={self.name})"

    def secret(self):
        """
        A helper method to load an oracle for the environment.
        :return:
        """
        raise NotImplementedError("This method should be overridden by subclasses.")


class StatelessEnv(AbstractEnvironment):
    """
    Stateless environment where the agent does not need to maintain a state.
    """

    def __init__(self, name: str):
        super().__init__(name)

    def get_reward(self, agent, action):
        """
        Get the interval duration and reward for the given action.
        action 0 is supposed to be better (higher reward)
        """
        # Roll for interval duration
        T = self.rng.uniform(self.interval_min_len, self.interval_max_len)

        if action == 1:
            reward = self.rng.normalvariate(T * 0.6, 2)
        elif action == 0:
            reward = self.rng.normalvariate(T * 0.4, 2)
        else:
            raise

        # make sure reward is positive
        reward = max(self.interval_min_len, reward)
        reward = min(T, reward)

        self.update_state()
        return T, reward


class TwoStatesEvenDistEnv(AbstractEnvironment):
    """
    Two states environment with even distribution.
    """

    def __init__(self, name: str):
        super().__init__(name)
        self.action_space = [0, 1]
        self.state = 0

    def update_state(self, action):
        """
        Update the state of the environment.
        In this environment, we can switch between two states.
        """
        self.state = self.rng.choice([0, 1])

    def get_reward(self, agent, action):
        """
        Get the interval duration and reward for the given action.
        action 0 is supposed to be better (higher reward) for state 0
        action 1 is supposed to be better (higher reward) for state 1
        """
        T = self.rng.uniform(self.interval_min_len, self.interval_max_len)
        if self.state == 0:
            if action == 0:
                reward = self.rng.normalvariate(T * 0.6, 2)
            elif action == 1:
                reward = self.rng.normalvariate(T * 0.55, 2)
        else:  # state 1
            if action == 0:
                reward = self.rng.normalvariate(T * 0.45, 2)
            elif action == 1:
                reward = self.rng.normalvariate(T * 0.5, 2)

        # make sure reward is positive
        reward = max(self.interval_min_len, reward)
        reward = min(T, reward)

        self.update_state(action)
        return T, reward

    def secret(self):
        """
        A helper method to load an oracle for the environment.
        In this environment, we can return a fixed oracle that always returns action 0.
        :return:
        """
        return lambda state: 0 if state == 0 else 1


class TwoStatesUnevenDistEnv(AbstractEnvironment):
    def __init__(self, name: str):
        super().__init__(name)
        self.action_space = [0, 1]
        self.state = 0

    def update_state(self, action):
        """
        Update the state of the environment.
        In this environment, we can switch between two states.
        """
        if self.state == 0:
            self.state = 1 if self.rng.random() < 0.2 else 0

        elif self.state == 1 and action == 0:
            self.state = 0 if self.rng.random() < 0.2 else 1

        elif self.state == 1 and action == 1:
            self.state = 0 if self.rng.random() < 0.8 else 1

    def get_reward(self, agent, action):
        """
        Get the interval duration and reward for the given action.
        action 0 is supposed to be better (higher reward) for state 0
        action 1 is supposed to be better (higher reward) for state 1
        """
        T = self.rng.uniform(self.interval_min_len, self.interval_max_len)

        if self.state == 0:
            if action == 0:
                reward = self.rng.normalvariate(T * 0.8, 2)
            elif action == 1:
                reward = self.rng.normalvariate(T * 0.5, 2)
        else:  # state 1
            if action == 0:
                reward = self.rng.normalvariate(T * 0.3, 2)
            elif action == 1:
                reward = self.rng.normalvariate(T * 0.1, 2)

        # make sure reward is positive
        reward = max(self.interval_min_len, reward)
        reward = min(T, reward)

        self.update_state(action)
        return T, reward

    def secret(self):
        """
        A helper method to load an oracle for the environment.
        In this environment, we can return a fixed oracle that always returns action 0.
        :return:
        """
        return lambda state: 0 if state == 0 else 1
