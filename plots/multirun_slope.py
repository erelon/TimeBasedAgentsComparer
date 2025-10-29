import os
import sys
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.express as px  # added for interactive HTML plots

# Ensure root path is on sys.path for package-relative imports when run as script
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from run_experiment import run_from_config  # noqa


def _plot_summary(summary_df: pd.DataFrame, output_dir: str, eval_steps_label: str = 'Average Reward', policy_label: str = 'Avg Last Policy Change'):
    if summary_df.empty:
        print("Provided summary dataframe is empty; skipping plots.")
        return
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style='whitegrid')
    # Average Reward plot (static)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=summary_df, x='slope', y='avg_reward', hue='agent', marker='o')
    plt.title('Average Reward vs Slope')
    plt.xlabel('Slope')
    plt.ylabel(eval_steps_label)
    plt.legend(title='Agent', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'avg_reward_vs_slope.png'))
    plt.close()
    # Interactive version
    fig_reward = px.line(summary_df, x='slope', y='avg_reward', color='agent', markers=True,
                         title='Average Reward vs Slope')
    fig_reward.update_layout(xaxis_title='Slope', yaxis_title=eval_steps_label, legend_title_text='Agent')
    fig_reward.write_html(os.path.join(output_dir, 'avg_reward_vs_slope.html'))
    # Policy Change plot (static)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=summary_df, x='slope', y='avg_last_policy_change', hue='agent', marker='o')
    plt.title('Avg Last Policy Change vs Slope')
    plt.xlabel('Slope')
    plt.ylabel(policy_label)
    plt.legend(title='Agent', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'avg_last_policy_change_vs_slope.png'))
    plt.close()
    # Interactive version
    fig_policy = px.line(summary_df, x='slope', y='avg_last_policy_change', color='agent', markers=True,
                         title='Avg Last Policy Change vs Slope')
    fig_policy.update_layout(xaxis_title='Slope', yaxis_title=policy_label, legend_title_text='Agent')
    fig_policy.write_html(os.path.join(output_dir, 'avg_last_policy_change_vs_slope.html'))


def run_slope_scan(base_config_path: str, slope_start=0.1, slope_end=2.0, slope_step=0.1,
                   scale_modes: list[str] | None = None,
                   interval_max_len_override: int | None = None):
    with open(base_config_path, 'r') as f:
        base_cfg = yaml.safe_load(f)
    if base_cfg is None:
        raise ValueError(f"Config file '{base_config_path}' is empty or invalid YAML.")

    if scale_modes is None or len(scale_modes) == 0:
        scale_modes = [base_cfg.get('environment', {}).get('params', {}).get('_scale_mode', 'exp')]

    exp_cfg = base_cfg.get('experiment', {})
    eval_steps = exp_cfg.get('eval_steps', 1000)
    epochs = exp_cfg.get('epochs', 100)
    results_root = exp_cfg.get('results_root', 'results')

    # Force environment class to slope shifting env if not already
    env_cfg = base_cfg.get('environment', {})
    if env_cfg.get('class') != 'envs.shifting.SlopeShiftingUnevenTwoStates':
        env_cfg['class'] = 'envs.shifting.SlopeShiftingUnevenTwoStates'
        env_cfg.setdefault('params', {})
        env_cfg['params'].setdefault('_slope', 0.99)
        env_cfg['params'].setdefault('_shiftsteps', 50)
        env_cfg['params'].setdefault('_apply_every_step', False)
        env_cfg['params'].setdefault('_scale_mode', 'exp')
        base_cfg['environment'] = env_cfg

    if interval_max_len_override is not None:
        env_cfg.setdefault('params', {})
        env_cfg['params']['interval_max_len'] = interval_max_len_override

    # Remove params not applicable to slope env (carried over from shifting config)
    for obsolete in ['_shift_min', '_shift_max']:
        if obsolete in env_cfg.get('params', {}):
            env_cfg['params'].pop(obsolete)

    slopes = []
    run_dirs = []
    summary_records = []  # one record per (slope, agent, scale_mode)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    interval_tag = f"imax{env_cfg['params'].get('interval_max_len','NA')}"
    modes_tag = "_".join(scale_modes)
    aggregate_dir = os.path.join(results_root, f"slope_scan_{modes_tag}_{interval_tag}_{timestamp}")
    os.makedirs(aggregate_dir, exist_ok=True)

    slope_values = []
    v = slope_start
    while v <= slope_end + 1e-9:  # floating tolerance
        slope_values.append(round(v, 4))
        v += slope_step

    for scale_mode in scale_modes:
        for slope in slope_values:
            cfg = yaml.safe_load(yaml.safe_dump(base_cfg))  # deep copy via serialize
            cfg.setdefault('experiment', {})
            cfg['experiment']['name'] = f"{scale_mode}_slope_{slope}"  # override name for run dir
            # inject slope & scale_mode
            cfg['environment']['params']['_slope'] = slope
            cfg['environment']['params']['_scale_mode'] = scale_mode
            # Remove obsolete params if they exist in cloned config
            for obsolete in ['_shift_min', '_shift_max']:
                cfg['environment']['params'].pop(obsolete, None)
            if interval_max_len_override is not None:
                cfg['environment']['params']['interval_max_len'] = interval_max_len_override
            # Create temp file
            tmp_cfg_path = os.path.join(aggregate_dir, f"cfg_{scale_mode}_slope_{slope}.yaml")
            with open(tmp_cfg_path, 'w') as tf:
                yaml.safe_dump(cfg, tf)
            run_dir = run_from_config(tmp_cfg_path)
            run_dirs.append(run_dir)
            slopes.append(slope)
            # Load results CSV
            exp_cfg_local = cfg.get('experiment', {})
            output_csv_name = exp_cfg_local.get('output_csv', 'results.csv')
            results_csv_path = os.path.join(run_dir, output_csv_name)
            df = pd.read_csv(results_csv_path, index_col=0)
            # Row labels
            reward_row = f"Average Reward over {eval_steps} steps"
            policy_row = f"Avg Last Policy Change (over {epochs} runs)"
            for agent_name in df.columns:
                avg_reward = df.loc[reward_row, agent_name]
                avg_policy_change = df.loc[policy_row, agent_name]
                summary_records.append({
                    'slope': slope,
                    'agent': agent_name,
                    'avg_reward': avg_reward,
                    'avg_last_policy_change': avg_policy_change,
                    'scale_mode': scale_mode,
                    'interval_max_len': cfg['environment']['params'].get('interval_max_len')
                })

    summary_df = pd.DataFrame(summary_records)
    summary_csv = os.path.join(aggregate_dir, 'slope_scan_summary.csv')
    summary_df.to_csv(summary_csv, index=False, float_format='%.6f')

    if summary_df.empty:
        print("No data collected during slope scan; check configuration.")
        return aggregate_dir, summary_df

    # Produce per-mode plots plus combined
    for scale_mode in scale_modes:
        mode_df = summary_df[summary_df['scale_mode'] == scale_mode]
        if mode_df.empty:
            continue
        mode_plot_dir = os.path.join(aggregate_dir, f"plots_{scale_mode}")
        _plot_summary(mode_df, mode_plot_dir)
        # Backwards compatibility: if only one mode, also place legacy PNGs at root
        if len(scale_modes) == 1:
            # Copy (recreate) the two legacy plots at aggregate_dir root naming
            import shutil
            src_reward = os.path.join(mode_plot_dir, 'avg_reward_vs_slope.png')
            src_policy = os.path.join(mode_plot_dir, 'avg_last_policy_change_vs_slope.png')
            dst_reward = os.path.join(aggregate_dir, 'avg_reward_vs_slope.png')
            dst_policy = os.path.join(aggregate_dir, 'avg_last_policy_change_vs_slope.png')
            if os.path.exists(src_reward):
                shutil.copy(src_reward, dst_reward)
            if os.path.exists(src_policy):
                shutil.copy(src_policy, dst_policy)

    # Combined interactive distinguishing scale_mode
    sns.set_theme(style='whitegrid')
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=summary_df, x='slope', y='avg_reward', hue='agent', style='scale_mode', markers=True)
    plt.title('Average Reward vs Slope (combined modes)')
    plt.xlabel('Slope')
    plt.ylabel('Average Reward')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(aggregate_dir, 'avg_reward_vs_slope_combined.png'))
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=summary_df, x='slope', y='avg_last_policy_change', hue='agent', style='scale_mode', markers=True)
    plt.title('Avg Last Policy Change vs Slope (combined modes)')
    plt.xlabel('Slope')
    plt.ylabel('Avg Last Policy Change')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(aggregate_dir, 'avg_last_policy_change_vs_slope_combined.png'))
    plt.close()

    print(f"Aggregate slope scan results written to: {aggregate_dir}")
    return aggregate_dir, summary_df


def run_slope_plot_from_csv(csv_paths, output_dir=None, combine_method='mean'):
    """Recreate slope plots from one or more existing summary CSV files.

    Args:
        csv_paths (list[str]): Paths to previously generated slope_scan_summary.csv files.
        output_dir (str|None): Directory to write plots; defaults to aggregate directory.
        combine_method (str): How to combine duplicates ('mean', 'append', 'median').
    Returns:
        (output_dir, summary_df) tuple.
    """
    if isinstance(csv_paths, str):
        csv_paths = [csv_paths]
    records = []
    for path in csv_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"CSV not found: {path}")
        df = pd.read_csv(path)
        required_cols = {'slope', 'agent', 'avg_reward', 'avg_last_policy_change'}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"CSV {path} missing required columns {required_cols}")
        records.append(df)
    if not records:
        raise ValueError("No valid CSVs provided.")
    all_df = pd.concat(records, ignore_index=True)

    if combine_method == 'append':
        summary_df = all_df
    else:
        agg_func = 'mean' if combine_method == 'mean' else 'median'
        summary_df = all_df.groupby(['slope', 'agent'], as_index=False).agg({
            'avg_reward': agg_func,
            'avg_last_policy_change': agg_func
        })

    # Determine output directory
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join('../results', f'slope_replot_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    combined_csv_path = os.path.join(output_dir, 'combined_slope_summary.csv')
    summary_df.to_csv(combined_csv_path, index=False, float_format='%.6f')
    # produce both static and interactive plots via helper
    _plot_summary(summary_df, output_dir)
    print(f"Recreated plots from CSV(s). Output: {output_dir}")
    return output_dir, summary_df


if __name__ == '__main__':
    # Allow simple direct invocation; if CSV paths passed separated by commas
    if len(sys.argv) > 1 and sys.argv[1].lower().endswith('.csv'):
        paths = sys.argv[1].split(',')
        run_slope_plot_from_csv(paths)
    else:
        cfg_path = sys.argv[1] if len(sys.argv) > 1 else 'configs/config.yaml'
        run_slope_scan(cfg_path)
