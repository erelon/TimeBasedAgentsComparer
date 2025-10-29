import os
import pandas as pd
from ..plots.multirun_shift_range import run_shift_range_scan


def test_multirun_shift_range_small(tmp_path):
    cfg_path = os.path.join(os.getcwd(), 'configs', 'config_shift_range.yaml')
    agg_dir, summary_df = run_shift_range_scan(cfg_path, mode='vary_min', start=0.1, end=0.2, step=0.1, fixed_value=1.0)
    assert os.path.isdir(agg_dir)
    summary_csv = os.path.join(agg_dir, 'shift_range_summary.csv')
    assert os.path.exists(summary_csv)
    assert not summary_df.empty
    assert os.path.exists(os.path.join(agg_dir, 'avg_reward_vs_var_value.png'))
    assert os.path.exists(os.path.join(agg_dir, 'avg_last_policy_change_vs_var_value.png'))
    df_loaded = pd.read_csv(summary_csv)
    assert 'var_value' in df_loaded.columns
    assert 'avg_reward' in df_loaded.columns

