from .two_states_uneven import TwoStatesUnevenDistEnv
import math


class ShiftingUnevenTwoStates(TwoStatesUnevenDistEnv):
    def __init__(self, name: str, _maxp=0.8, _maxv=5, _shiftsteps=50, _shift_min=0.1, _shift_max=1.0, **kwargs):
        # _shift_min/_shift_max define the uniform range for random shift_constant updates
        super().__init__(name, _maxp=_maxp, _maxv=_maxv, **kwargs)
        self.shiftsteps = _shiftsteps
        self.shift_min = _shift_min
        self.shift_max = _shift_max

    def reset(self):
        super().reset()
        self.clock = 0
        self.shift_constant = 1.0

    def update_state(self, action):
        super().update_state(action)
        self.clock += 1
        if self.clock % self.shiftsteps == 0:
            # Random shift within configured bounds
            self.shift_constant = self.rng.uniform(self.shift_min, self.shift_max)

    def get_reward(self, agent, action):
        T = self.rng.uniform(self.interval_min_len, self.interval_max_len)
        if self.state == 0:
            if action == 0:
                reward = self.rng.normalvariate(T * (self.maxp / 1.0) * self.shift_constant, self.maxv)
            else:
                reward = self.rng.normalvariate(T * (self.maxp / 2.0) * self.shift_constant, self.maxv)
        else:
            if action == 0:
                reward = self.rng.normalvariate(T * (self.maxp / 3.0) * self.shift_constant, self.maxv)
            else:
                reward = self.rng.normalvariate(T * (self.maxp / 1.5) * self.shift_constant, self.maxv)
        reward = max(self.interval_min_len, reward)
        reward = min(T, reward)
        self.update_state(action)
        return T, reward

    def secret(self):
        return lambda state: 0 if self.state == 0 else 1


class SlopeShiftingUnevenTwoStates(TwoStatesUnevenDistEnv):
    """Deterministic shifting environment with configurable scaling mode.

    Modes (``_scale_mode``):
    - 'exp' (default): multiplicative decay/growth -> shift_constant *= _slope
      Typical range: _slope in (0,1) for decay, >1 for growth.
    - 'linear': additive -> shift_constant += _slope
      Use negative _slope for linear decay.
    - 'toward_target': move fractionally toward a target value ->
        shift_constant += (_target - shift_constant) * _slope
      Here _slope is a learning-rate in (0,1]; _target must be provided.
    - 'logistic': logistic growth toward target ->
        shift_constant += _slope * shift_constant * (1 - shift_constant / _target)
      Requires _target > 0.
    - 'sinusoidal': oscillatory overwrite ->
        shift_constant = _sinusoidal_base + _sinusoidal_amplitude * sin(2π * clock / _sinusoidal_period)
      Requires _sinusoidal_period > 0 and _sinusoidal_amplitude != 0.

    Optional clamping with ``_min_shift_constant`` / ``_max_shift_constant`` after update.
    You can choose to apply updates every step (``_apply_every_step``) or only
    at multiples of ``_shiftsteps``.
    """

    def __init__(
        self,
        name: str,
        _maxp=0.8,
        _maxv=5,
        _shiftsteps=50,
        _slope=0.99,
        _apply_every_step=False,
        _scale_mode='exp',
        _min_shift_constant=None,
        _max_shift_constant=None,
        _target=None,
        _sinusoidal_period=None,
        _sinusoidal_amplitude=0.0,
        _sinusoidal_base=1.0,
        **kwargs
    ):
        super().__init__(name, _maxp=_maxp, _maxv=_maxv, **kwargs)
        self.shiftsteps = _shiftsteps
        self.slope = _slope  # parameter whose semantics depend on scale mode
        self.apply_every_step = _apply_every_step
        self.scale_mode = _scale_mode
        self.min_shift_constant = _min_shift_constant
        self.max_shift_constant = _max_shift_constant
        self.target = _target
        self.sinusoidal_period = _sinusoidal_period
        self.sinusoidal_amplitude = _sinusoidal_amplitude
        self.sinusoidal_base = _sinusoidal_base

        # Validate mode requirements
        if self.scale_mode == 'toward_target' and self.target is None:
            raise ValueError("_target must be provided when _scale_mode='toward_target'")
        if self.scale_mode == 'logistic':
            if self.target is None:
                raise ValueError("_target must be provided when _scale_mode='logistic'")
            if self.target <= 0:
                raise ValueError("_target must be > 0 for logistic mode")
        if self.scale_mode == 'sinusoidal':
            if self.sinusoidal_period is None or self.sinusoidal_period <= 0:
                raise ValueError("_sinusoidal_period must be > 0 when _scale_mode='sinusoidal'")
            if self.sinusoidal_amplitude == 0.0:
                raise ValueError("_sinusoidal_amplitude must be non-zero when _scale_mode='sinusoidal'")

    def reset(self):
        super().reset()
        self.clock = 0
        self.shift_constant = 1.0

    def _apply_clamp(self):
        if self.min_shift_constant is not None:
            self.shift_constant = max(self.min_shift_constant, self.shift_constant)
        if self.max_shift_constant is not None:
            self.shift_constant = min(self.max_shift_constant, self.shift_constant)

    def update_shift(self):
        if self.scale_mode == 'exp':
            self.shift_constant *= self.slope
        elif self.scale_mode == 'linear':
            self.shift_constant += self.slope
        elif self.scale_mode == 'toward_target':
            self.shift_constant += (self.target - self.shift_constant) * self.slope
        elif self.scale_mode == 'logistic':
            self.shift_constant += self.slope * self.shift_constant * (1 - self.shift_constant / self.target)
        elif self.scale_mode == 'sinusoidal':
            # overwrite based on clock (deterministic cycle)
            self.shift_constant = self.sinusoidal_base + self.sinusoidal_amplitude * math.sin(2 * math.pi * self.clock / self.sinusoidal_period)
        else:
            raise ValueError(f"Unknown scale mode: {self.scale_mode}")
        self._apply_clamp()

    def update_state(self, action):
        super().update_state(action)
        self.clock += 1
        if self.apply_every_step or (self.clock % self.shiftsteps == 0):
            self.update_shift()

    def get_reward(self, agent, action):
        T = self.rng.uniform(self.interval_min_len, self.interval_max_len)
        if self.state == 0:
            if action == 0:
                reward = self.rng.normalvariate(T * (self.maxp / 1.0) * self.shift_constant, self.maxv)
            else:
                reward = self.rng.normalvariate(T * (self.maxp / 2.0) * self.shift_constant, self.maxv)
        else:
            if action == 0:
                reward = self.rng.normalvariate(T * (self.maxp / 3.0) * self.shift_constant, self.maxv)
            else:
                reward = self.rng.normalvariate(T * (self.maxp / 1.5) * self.shift_constant, self.maxv)
        reward = max(self.interval_min_len, reward)
        reward = min(T, reward)
        self.update_state(action)
        return T, reward

    def secret(self):
        return lambda state: 0 if self.state == 0 else 1
