import os
import sys
from datetime import datetime
import yaml
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Ensure root path on sys.path when run as a script
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from run_experiment import run_from_config  # noqa


VARIED_MIN = 'vary_min'
VARIED_MAX = 'vary_max'


def _validate_mode(mode: str):
    mode = mode.lower()
    if mode not in (VARIED_MIN, VARIED_MAX):
        raise ValueError(f"mode must be one of '{VARIED_MIN}','{VARIED_MAX}' (got {mode})")
    return mode


def _frange(start: float, end: float, step: float):
    vals = []
    v = start
    # Add small tolerance for float addition finishing on boundary
    while v <= end + 1e-9:
        vals.append(round(v, 6))
        v += step
    return vals


def _plot(summary_df: pd.DataFrame, output_dir: str, x_col: str, x_label: str):
    if summary_df.empty:
        print('Empty summary_df, skipping plots.')
        return
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style='whitegrid')
    # Average reward static
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=summary_df, x=x_col, y='avg_reward', hue='agent', marker='o')
    plt.title(f'Average Reward vs {x_label}')
    plt.xlabel(x_label)
    plt.ylabel('Average Reward')
    plt.legend(title='Agent', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'avg_reward_vs_{x_col}.png'))
    plt.close()
    # Interactive
    fig_r = px.line(summary_df, x=x_col, y='avg_reward', color='agent', markers=True,
                    title=f'Average Reward vs {x_label}')
    fig_r.update_layout(xaxis_title=x_label, yaxis_title='Average Reward', legend_title_text='Agent')
    fig_r.write_html(os.path.join(output_dir, f'avg_reward_vs_{x_col}.html'))

    # Policy change static
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=summary_df, x=x_col, y='avg_last_policy_change', hue='agent', marker='o')
    plt.title(f'Avg Last Policy Change vs {x_label}')
    plt.xlabel(x_label)
    plt.ylabel('Avg Last Policy Change')
    plt.legend(title='Agent', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'avg_last_policy_change_vs_{x_col}.png'))
    plt.close()
    # Interactive
    fig_p = px.line(summary_df, x=x_col, y='avg_last_policy_change', color='agent', markers=True,
                    title=f'Avg Last Policy Change vs {x_label}')
    fig_p.update_layout(xaxis_title=x_label, yaxis_title='Avg Last Policy Change', legend_title_text='Agent')
    fig_p.write_html(os.path.join(output_dir, f'avg_last_policy_change_vs_{x_col}.html'))


def run_shift_range_scan(base_config_path: str,
                         mode: str = VARIED_MIN,
                         start: float = 0.1,
                         end: float = 1.0,
                         step: float = 0.1,
                         fixed_value: float | None = None,
                         interval_max_len_override: int | None = None):
    """Run repeated experiments across a range of (shift_min, shift_max) values.

    Parameters
    ----------
    base_config_path : str
        Path to base YAML config file.
    mode : str
        'vary_min' to vary _shift_min with fixed _shift_max (default), or
        'vary_max' to vary _shift_max with fixed _shift_min.
    start, end, step : float
        Range specification for the varied parameter.
    fixed_value : float | None
        If provided, overrides the static counterpart. If None, a sensible
        default is used: when varying min, fixed max defaults to 1.0; when
        varying max, fixed min defaults to 1.0.
    interval_max_len_override : int | None
        If provided, overrides environment.params.interval_max_len for all runs.
    Returns
    -------
    (aggregate_dir, summary_df)
    """
    mode = _validate_mode(mode)
    with open(base_config_path, 'r') as f:
        base_cfg = yaml.safe_load(f)
    if base_cfg is None:
        raise ValueError(f"Config file '{base_config_path}' is empty or invalid YAML.")

    # Force environment to ShiftingUnevenTwoStates
    env_cfg = base_cfg.get('environment', {})
    if env_cfg.get('class') != 'envs.shifting.ShiftingUnevenTwoStates':
        env_cfg['class'] = 'envs.shifting.ShiftingUnevenTwoStates'
        env_cfg.setdefault('params', {})
        env_cfg['params'].setdefault('_maxp', 0.8)
        env_cfg['params'].setdefault('_maxv', 5)
        env_cfg['params'].setdefault('_shiftsteps', 50)
        env_cfg['params'].setdefault('_shift_min', 0.1)
        env_cfg['params'].setdefault('_shift_max', 1.0)
        base_cfg['environment'] = env_cfg

    # Optional override of interval_max_len
    if interval_max_len_override is not None:
        env_cfg.setdefault('params', {})
        env_cfg['params']['interval_max_len'] = interval_max_len_override

    exp_cfg = base_cfg.get('experiment', {})
    eval_steps = exp_cfg.get('eval_steps', 1000)
    epochs = exp_cfg.get('epochs', 100)
    results_root = exp_cfg.get('results_root', 'results')

    varied_param = '_shift_min' if mode == VARIED_MIN else '_shift_max'
    static_param = '_shift_max' if mode == VARIED_MIN else '_shift_min'

    if fixed_value is None:
        fixed_value = 1.0  # default for either static side

    varied_values = _frange(start, end, step)

    # Validate monotonic & bounds
    if mode == VARIED_MIN:
        for v in varied_values:
            if v > fixed_value:
                raise ValueError(f"_shift_min ({v}) cannot exceed fixed _shift_max ({fixed_value}).")
    else:  # vary max
        for v in varied_values:
            if v < fixed_value:
                raise ValueError(f"_shift_max ({v}) cannot be below fixed _shift_min ({fixed_value}).")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    interval_tag = f"imax{env_cfg['params'].get('interval_max_len','NA')}"
    aggregate_dir = os.path.join(results_root, f'shift_range_scan_{mode}_{interval_tag}_{timestamp}')
    os.makedirs(aggregate_dir, exist_ok=True)

    summary_records = []
    run_dirs = []

    for val in varied_values:
        cfg = yaml.safe_load(yaml.safe_dump(base_cfg))  # deep copy
        cfg.setdefault('experiment', {})
        cfg['experiment']['name'] = f'{varied_param}_{val}'
        cfg['environment']['params'][varied_param] = val
        cfg['environment']['params'][static_param] = fixed_value
        if interval_max_len_override is not None:
            cfg['environment']['params']['interval_max_len'] = interval_max_len_override

        tmp_cfg_path = os.path.join(aggregate_dir, f'cfg_{varied_param}_{val}.yaml')
        with open(tmp_cfg_path, 'w') as tf:
            yaml.safe_dump(cfg, tf)

        run_dir = run_from_config(tmp_cfg_path)
        run_dirs.append(run_dir)

        output_csv_name = cfg['experiment'].get('output_csv', 'results.csv')
        results_csv_path = os.path.join(run_dir, output_csv_name)
        df = pd.read_csv(results_csv_path, index_col=0)
        reward_row = f"Average Reward over {eval_steps} steps"
        policy_row = f"Avg Last Policy Change (over {epochs} runs)"

        for agent_name in df.columns:
            summary_records.append({
                'var_value': val,
                'agent': agent_name,
                'avg_reward': df.loc[reward_row, agent_name],
                'avg_last_policy_change': df.loc[policy_row, agent_name],
                'shift_min': cfg['environment']['params']['_shift_min'],
                'shift_max': cfg['environment']['params']['_shift_max'],
                'varied_param': varied_param,
                'interval_max_len': cfg['environment']['params'].get('interval_max_len')
            })

    summary_df = pd.DataFrame(summary_records)
    summary_csv = os.path.join(aggregate_dir, 'shift_range_summary.csv')
    summary_df.to_csv(summary_csv, index=False, float_format='%.6f')

    if summary_df.empty:
        print('No data collected during shift range scan.')
        return aggregate_dir, summary_df

    # Plot using generic x label
    x_label = 'Shift Min' if mode == VARIED_MIN else 'Shift Max'
    _plot(summary_df, aggregate_dir, x_col='var_value', x_label=x_label)

    # Save manifest
    manifest_path = os.path.join(aggregate_dir, 'run_dirs.txt')
    with open(manifest_path, 'w') as mf:
        for val, rd in zip(varied_values, run_dirs):
            mf.write(f"{varied_param}={val}\t{rd}\n")

    print(f'Aggregate shift range scan results written to: {aggregate_dir}')
    return aggregate_dir, summary_df


def run_shift_plot_from_csv(csv_paths, output_dir=None, combine_method='mean'):
    """Recreate shift range plots from one or more existing summary CSV files.

    Parameters
    ----------
    csv_paths : list[str] | str
        Path(s) to previously generated shift_range_summary.csv files.
    output_dir : str | None
        Directory to write combined plots & CSV. Auto if None.
    combine_method : str
        'mean', 'median', or 'append' for duplicate (var_value, agent) rows.
    """
    if isinstance(csv_paths, str):
        csv_paths = [csv_paths]
    dfs = []
    for p in csv_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f'CSV not found: {p}')
        df = pd.read_csv(p)
        required = {'var_value', 'agent', 'avg_reward', 'avg_last_policy_change', 'varied_param'}
        if not required.issubset(df.columns):
            raise ValueError(f'CSV {p} missing required columns {required}')
        dfs.append(df)
    if not dfs:
        raise ValueError('No valid CSVs provided')
    all_df = pd.concat(dfs, ignore_index=True)

    if combine_method == 'append':
        summary_df = all_df
    else:
        agg_func = 'mean' if combine_method == 'mean' else 'median'
        summary_df = all_df.groupby(['var_value', 'agent', 'varied_param'], as_index=False).agg({
            'avg_reward': agg_func,
            'avg_last_policy_change': agg_func
        })
    varied_param = summary_df['varied_param'].iloc[0] if not summary_df.empty else 'var'
    x_label = 'Shift Min' if varied_param == '_shift_min' else 'Shift Max'

    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join('../results', f'shift_range_replot_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    combined_csv = os.path.join(output_dir, 'combined_shift_range_summary.csv')
    summary_df.to_csv(combined_csv, index=False, float_format='%.6f')
    _plot(summary_df, output_dir, x_col='var_value', x_label=x_label)
    print(f'Recreated shift range plots. Output: {output_dir}')
    return output_dir, summary_df


if __name__ == '__main__':
    # CLI usage examples:
    # python plots/multirun_shift_range.py configs/config.yaml vary_min 0.1 1.0 0.1 1.0
    # python plots/multirun_shift_range.py configs/config.yaml vary_max 1.0 2.0 0.1 1.0
    if len(sys.argv) > 1 and sys.argv[1].lower().endswith('.csv'):
        paths = sys.argv[1].split(',')
        run_shift_plot_from_csv(paths)
    else:
        cfg_path = sys.argv[1] if len(sys.argv) > 1 else 'configs/config.yaml'
        mode = sys.argv[2] if len(sys.argv) > 2 else VARIED_MIN
        start = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
        end = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
        step = float(sys.argv[5]) if len(sys.argv) > 5 else 0.1
        fixed = float(sys.argv[6]) if len(sys.argv) > 6 else 1.0
        run_shift_range_scan(cfg_path, mode=mode, start=start, end=end, step=step, fixed_value=fixed)
