import os
import glob
import pandas as pd
from ..run_experiment import run_from_config
from ..gym_wrapper import SMDPEnvWrapper
from ..envs.stateless import StatelessEnv


def test_small_config_runs(tmp_path):
    config_path = os.path.join(os.getcwd(), 'configs', 'config_small.yaml')
    run_from_config(config_path)
    # find latest results directory for 'small'
    result_dirs = sorted(glob.glob(os.path.join('results', 'small_*')))
    assert result_dirs, 'No results directory created for small experiment'
    latest = result_dirs[-1]
    csv_path = os.path.join(latest, 'small_results.csv')
    assert os.path.exists(csv_path), 'Results CSV not created in run dir'
    df = pd.read_csv(csv_path)
    assert not df.empty


def test_gym_wrapper_step():
    env = StatelessEnv('Stateless')
    wrapped = SMDPEnvWrapper(env)
    obs, info = wrapped.reset()
    assert isinstance(obs, int)
    action = wrapped.action_space.sample()
    obs2, reward, terminated, truncated, info2 = wrapped.step(action)
    assert isinstance(obs2, int)
    assert isinstance(reward, (int, float))
    assert 'duration' in info2
