import os
import pandas as pd
from ..plots.multirun_slope import run_slope_scan


def test_multirun_slope_small(tmp_path):
    cfg_path = os.path.join(os.getcwd(), 'configs', 'config_slope.yaml')
    agg_dir, summary_df = run_slope_scan(cfg_path, slope_start=0.9, slope_end=1.0, slope_step=0.1)
    assert os.path.isdir(agg_dir)
    summary_csv = os.path.join(agg_dir, 'slope_scan_summary.csv')
    assert os.path.exists(summary_csv)
    assert not summary_df.empty
    assert os.path.exists(os.path.join(agg_dir, 'avg_reward_vs_slope.png'))
    assert os.path.exists(os.path.join(agg_dir, 'avg_last_policy_change_vs_slope.png'))
    df_loaded = pd.read_csv(summary_csv)
    assert 'slope' in df_loaded.columns
    assert 'avg_reward' in df_loaded.columns
