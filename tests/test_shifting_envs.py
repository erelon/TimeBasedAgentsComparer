import math
from ..envs.shifting import ShiftingUnevenTwoStates, SlopeShiftingUnevenTwoStates
import pytest


def test_random_shifting_range():
    env = ShiftingUnevenTwoStates("RandShift", _shiftsteps=2, _shift_min=0.3, _shift_max=0.4)
    env.reset()
    initial = env.shift_constant
    # Trigger shift; after two steps shift_constant should be in range
    for _ in range(2):
        env.update_state(0)
    assert 0.3 <= env.shift_constant <= 0.4
    assert env.shift_constant != initial


def test_slope_shifting_interval():
    env = SlopeShiftingUnevenTwoStates("SlopeShift", _shiftsteps=2, _slope=0.9, _apply_every_step=False)
    env.reset()
    # After first step (not multiple of shiftsteps) shift_constant unchanged
    env.update_state(0)
    assert math.isclose(env.shift_constant, 1.0)
    # After second step shift applies
    env.update_state(0)
    assert math.isclose(env.shift_constant, 0.9, rel_tol=1e-6)


def test_slope_shifting_every_step():
    env = SlopeShiftingUnevenTwoStates("SlopeShiftEvery", _shiftsteps=10, _slope=0.95, _apply_every_step=True)
    env.reset()
    for i in range(5):
        env.update_state(0)
    # shift_constant should be 0.95^5
    expected = 0.95 ** 5
    assert math.isclose(env.shift_constant, expected, rel_tol=1e-6)


def test_linear_shifting_interval():
    env = SlopeShiftingUnevenTwoStates("LinearShift", _shiftsteps=3, _slope=-0.1, _apply_every_step=False, _scale_mode='linear')
    env.reset()
    # First two steps: no change
    env.update_state(0)
    assert math.isclose(env.shift_constant, 1.0)
    env.update_state(0)
    assert math.isclose(env.shift_constant, 1.0)
    # Third step triggers linear addition (decay)
    env.update_state(0)
    assert math.isclose(env.shift_constant, 0.9, rel_tol=1e-6)


def test_linear_shifting_every_step():
    env = SlopeShiftingUnevenTwoStates("LinearShiftEvery", _shiftsteps=5, _slope=0.05, _apply_every_step=True, _scale_mode='linear')
    env.reset()
    for _ in range(4):
        env.update_state(0)
    # shift_constant = 1.0 + 4*0.05 = 1.2
    assert math.isclose(env.shift_constant, 1.2, rel_tol=1e-6)


def test_toward_target():
    env = SlopeShiftingUnevenTwoStates("TowardTarget", _shiftsteps=1, _slope=0.5, _apply_every_step=True, _scale_mode='toward_target', _target=2.0)
    env.reset()
    # After first update: 1 + (2-1)*0.5 = 1.5
    env.update_state(0)
    assert math.isclose(env.shift_constant, 1.5, rel_tol=1e-6)
    # Next: 1.5 + (2-1.5)*0.5 = 1.75
    env.update_state(0)
    assert math.isclose(env.shift_constant, 1.75, rel_tol=1e-6)


def test_toward_target_requires_target():
    with pytest.raises(ValueError):
        SlopeShiftingUnevenTwoStates("TowardTargetMissing", _shiftsteps=1, _slope=0.5, _apply_every_step=True, _scale_mode='toward_target')


def test_logistic_growth_every_step():
    env = SlopeShiftingUnevenTwoStates(
        "LogisticGrowth", _shiftsteps=10, _slope=0.1, _apply_every_step=True, _scale_mode='logistic', _target=2.0
    )
    env.reset()
    env.update_state(0)  # clock=1
    expected1 = 1.0 + 0.1 * 1.0 * (1 - 1.0 / 2.0)  # 1 + 0.05 = 1.05
    assert math.isclose(env.shift_constant, expected1, rel_tol=1e-6)
    env.update_state(0)  # clock=2
    expected2 = expected1 + 0.1 * expected1 * (1 - expected1 / 2.0)
    assert math.isclose(env.shift_constant, expected2, rel_tol=1e-6)


def test_logistic_requires_target():
    with pytest.raises(ValueError):
        SlopeShiftingUnevenTwoStates("LogisticMissingTarget", _scale_mode='logistic', _slope=0.1)


def test_logistic_target_positive():
    with pytest.raises(ValueError):
        SlopeShiftingUnevenTwoStates("LogisticBadTarget", _scale_mode='logistic', _slope=0.1, _target=0)


def test_sinusoidal_every_step():
    env = SlopeShiftingUnevenTwoStates(
        "Sinusoidal", _scale_mode='sinusoidal', _apply_every_step=True, _sinusoidal_period=4, _sinusoidal_amplitude=0.5, _sinusoidal_base=1.0
    )
    env.reset()
    env.update_state(0)  # clock=1 -> sin(pi/2)=1 -> 1+0.5=1.5
    assert math.isclose(env.shift_constant, 1.5, rel_tol=1e-6)
    env.update_state(0)  # clock=2 -> sin(pi)=0 -> 1.0
    assert math.isclose(env.shift_constant, 1.0, rel_tol=1e-6)
    env.update_state(0)  # clock=3 -> sin(3pi/2)=-1 -> 0.5
    assert math.isclose(env.shift_constant, 0.5, rel_tol=1e-6)
    env.update_state(0)  # clock=4 -> sin(2pi)=0 -> 1.0
    assert math.isclose(env.shift_constant, 1.0, rel_tol=1e-6)


def test_sinusoidal_requires_period():
    with pytest.raises(ValueError):
        SlopeShiftingUnevenTwoStates("SinBadPeriod", _scale_mode='sinusoidal', _sinusoidal_amplitude=0.5)


def test_sinusoidal_requires_amplitude():
    with pytest.raises(ValueError):
        SlopeShiftingUnevenTwoStates("SinBadAmp", _scale_mode='sinusoidal', _sinusoidal_period=4)
