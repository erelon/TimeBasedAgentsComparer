import random


class AbstractEnvironment:
    def __init__(self, name: str, seed: int = None):
        self.name = name
        self.action_space = None
        # we want to make sure the random seed is set for reproducibility
        self.seed = seed if seed is not None else 42
        self.rng = random.Random(
            seed if seed is not None else 42
        )  # Initialize the random seed
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
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def get_action_space(self):
        return self.action_space

    def get_reward(self, agent, action):
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
        self.action_space = [0, 1]

    def get_state(self, agent=None):
        """
        Get the current state of the environment.
        In a stateless environment, the state is always 0.
        """
        return 0

    def get_reward(self, agent, action):
        """
        Get the interval duration and reward for the given action.
        action 0 is supposed to be better (higher reward)
        """
        # Roll for time
        if action == 0:
            time = self.rng.gauss(6, 2)  # TODO: replace with constants
        elif action == 1:
            time = self.rng.gauss(
                4, 2
            )  # TODO replace with same constant as before, set mean here to be as above minus some constant. This way we can play with separation between the arms

        # make sure time is positive
        time = max(0, time)
        # roll between 0 and time with mean of time/2:
        reward = self.rng.gauss(time / 2, time / 4)
        return time, max(0, reward)

    def secret(self):
        """
        A helper method to load an oracle for the environment.
        In a stateless environment, we can return a fixed oracle.
        :return:
        """
        return lambda state: 0  # Always return action 0 as the best action
