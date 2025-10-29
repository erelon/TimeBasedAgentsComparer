# Time-Based Agents Comparer

This project implements and evaluates a collection of reinforcement learning (RL) and bandit agents in semi-Markov decision process (SMDP) environments where each state-action produces both a reward and a *duration* (time).

Key features:
- Modular environment & agent packages (`envs/`, `agents/`)
- YAML-based configuration in `configs/` (multiple experiment configs)
- Epsilon (exploration rate) decay scheduling (linear / exponential) via `epsilon_schedule`
- Gymnasium wrapper (`SMDPEnvWrapper`) for SMDP environments
- Timestamped, per-run results directories under `results/`
- CLI with override flags for quick experimentation
- Rich time‑varying environment dynamics (random shifts, slope-based deterministic scaling modes)

## Quick Start
```bash
# Clone and install
git clone https://github.com/erelon/TimeBasedAgentsComparer
cd TimeBasedAgentsComparer/TimeBasedAgentsComparer
pip install -r requirements.txt

# Run a small example experiment
python run_experiment.py configs/config_small.yaml

# Or via CLI with overrides
python cli.py configs/config_small.yaml --episodes 100 --output custom.csv
```

## Configuration Files
All configuration files reside in `configs/`. Example (`configs/config.yaml`):
```yaml
experiment:
  name: shifting
  episodes: 5000
  eval_steps: 1000
  epochs: 10
  gym_mode: false
  results_root: results
  epsilon_schedule:
    start: 0.1
    end: 0.01
    decay_type: linear
    decay_episodes: 4000
```
`epsilon_schedule` applies to all agents that have an `exploration_rate` attribute.

## Results
Each run creates a directory: `results/<experiment_name>_<YYYYMMDD_HHMMSS>/` containing:
- `results.csv` (or custom name)
- `config_used.yaml` (for reproducibility)

Multiple runs never overwrite prior outputs.

## Epsilon Decay
Two schedule types supported:
- Linear: interpolates from `start` to `end` over `decay_episodes`.
- Exponential: set `decay_type: exponential` (approaches `end`).
If omitted, exploration rate remains constant.

## Slope-Based Shifting Environment (`SlopeShiftingUnevenTwoStates`)
`SlopeShiftingUnevenTwoStates` is a deterministic variant of the uneven two‑state environment whose reward scale factor (`shift_constant`) evolves over time according to a selected mode. Choose the mode with `_scale_mode`.

Supported `_scale_mode` values:
- `exp` (default): multiplicative update -> `shift_constant *= _slope` (decay if 0 < `_slope` < 1, growth if > 1)
- `linear`: additive update -> `shift_constant += _slope` (use negative slope for linear decay)
- `toward_target`: fractional move toward `_target` -> `shift_constant += (_target - shift_constant) * _slope` (where `_slope` ∈ (0,1])
- `logistic`: logistic growth/decay toward `_target` -> `shift_constant += _slope * shift_constant * (1 - shift_constant / _target)` (requires `_target > 0`)
- `sinusoidal`: overwrite with cyclic value -> `shift_constant = _sinusoidal_base + _sinusoidal_amplitude * sin(2π * clock / _sinusoidal_period)` (requires `_sinusoidal_period > 0` and non‑zero `_sinusoidal_amplitude`)

Additional optional parameters:
- `_apply_every_step` (bool): if true, apply the update each step; otherwise only when `clock % _shiftsteps == 0`.
- `_min_shift_constant`, `_max_shift_constant`: clamp after each update.
- `_target`: required for `toward_target` and `logistic` modes.
- `_sinusoidal_period`, `_sinusoidal_amplitude`, `_sinusoidal_base`: required/used for `sinusoidal` mode.

Example instantiations:
```python
from envs.shifting import SlopeShiftingUnevenTwoStates

# Exponential decay every 20 steps
env_exp = SlopeShiftingUnevenTwoStates("ExpDecay", _shiftsteps=20, _slope=0.97, _scale_mode='exp')

# Linear growth applied every step with clamping
env_lin = SlopeShiftingUnevenTwoStates("LinearGrow", _apply_every_step=True, _slope=0.02, _scale_mode='linear', _max_shift_constant=2.0)

# Approach target with learning-rate style slope
env_target = SlopeShiftingUnevenTwoStates("Toward2", _apply_every_step=True, _slope=0.3, _scale_mode='toward_target', _target=2.0)

# Logistic growth toward target
env_logistic = SlopeShiftingUnevenTwoStates("Logistic", _apply_every_step=True, _slope=0.15, _scale_mode='logistic', _target=3.0)

# Sinusoidal oscillation (period 50, amplitude 0.5 around base 1.0)
env_sin = SlopeShiftingUnevenTwoStates("SinOsc", _apply_every_step=True, _scale_mode='sinusoidal', _sinusoidal_period=50, _sinusoidal_amplitude=0.5, _sinusoidal_base=1.0)
```

YAML override snippet example:
```yaml
environment:
  class: envs.shifting.SlopeShiftingUnevenTwoStates
  params:
    _scale_mode: logistic
    _slope: 0.1
    _target: 2.0
    _apply_every_step: true
    _min_shift_constant: 0.1
    _max_shift_constant: 3.0
```

## Agents & Environments
(See earlier detailed sections; unchanged except for epsilon scheduling integration and new scaling modes.)

## Extending
Add a new YAML under `configs/`, then run it. No code changes required for most additions.

## Tests
Run all tests:
```bash
pytest -q
```
Tests now cover: random shifting, exponential / linear / target / logistic / sinusoidal scaling modes and validation errors.

## License
MIT – see `LICENSE`.

## Multi-Run Slope Scan
You can run a parameter sweep over `_slope` (works best with `exp`, `linear`, `toward_target`, or `logistic` modes) and automatically generate plots:
```bash
python cli.py configs/config_slope.yaml --slope-scan --slope-start 0.1 --slope-end 2.0 --slope-step 0.1
```
Outputs:
- Aggregated directory: `results/slope_scan_<timestamp>/`
  - `slope_scan_summary.csv`: per-agent metrics across slope values
  - `avg_reward_vs_slope.png`
  - `avg_last_policy_change_vs_slope.png`
  - Individual run subdirectories + manifests

Adjust `--slope-start`, `--slope-end`, and `--slope-step` for finer or coarser sweeps. For non-monotonic (sinusoidal) behavior, consider period sweeps instead.
