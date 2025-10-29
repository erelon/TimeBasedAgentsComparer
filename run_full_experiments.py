"""Orchestrate full experiment suite requested:
- Shift range scans (vary_min and vary_max) under ShiftingUnevenTwoStates
- Slope scans (scale modes: 'linear' and 'exp') under SlopeShiftingUnevenTwoStates
- Each for interval_max_len in {50, 500}

Configuration base: uses provided config path (default configs/config.yaml) but overrides
only necessary parameters per run.

Outputs: directories under results/ with tagged names. A master manifest CSV summarizing
all runs is produced at the end.
"""
from __future__ import annotations
import os
import yaml
import pandas as pd
from datetime import datetime
from plots.multirun_shift_range import run_shift_range_scan
from plots.multirun_slope import run_slope_scan
import argparse

DEFAULT_CONFIG = 'configs/config.yaml'

# Parameter grids
SHIFT_RANGE_MODES = ['vary_min', 'vary_max']
SLOPE_SCALE_MODES = ['exp', 'linear']
INTERVAL_MAX_LENS = [500]

# Sweep specifics (can be customized)
SHIFT_START = 0.1
SHIFT_END = 1.0
SHIFT_STEP = 0.1  # fewer points for speed by default
FIXED_MIN_WHEN_VARY_MAX = 0.1
FIXED_MAX_WHEN_VARY_MIN = 1.0

SLOPE_START = 0.1
SLOPE_END = 2.0
SLOPE_STEP = 0.05


def ensure_results_root(cfg_path: str) -> str:
    with open(cfg_path, 'r') as f:
        base_cfg = yaml.safe_load(f)
    exp_cfg = base_cfg.get('experiment', {})
    root = exp_cfg.get('results_root', 'results')
    os.makedirs(root, exist_ok=True)
    return root


def run_suite(config_path: str = DEFAULT_CONFIG,
              shift_start: float = SHIFT_START,
              shift_end: float = SHIFT_END,
              shift_step: float = SHIFT_STEP,
              slope_start: float = SLOPE_START,
              slope_end: float = SLOPE_END,
              slope_step: float = SLOPE_STEP,
              fast: bool = False,
              shift_modes: list[str] | None = None):
    # Determine which shift modes to run; default to both only if not explicitly constrained
    if shift_modes is None or len(shift_modes) == 0:
        shift_modes = SHIFT_RANGE_MODES
    # Validate
    for m in shift_modes:
        if m not in SHIFT_RANGE_MODES:
            raise ValueError(f"Invalid shift mode '{m}'. Allowed: {SHIFT_RANGE_MODES}")
    results_root = ensure_results_root(config_path)
    suite_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # Tag suite directory with selected modes (avoid mixing vary_min/vary_max unintentionally)
    modes_tag = "-".join(shift_modes)
    suite_dir = os.path.join(results_root, f"full_suite_{modes_tag}_{suite_timestamp}")
    os.makedirs(suite_dir, exist_ok=True)

    manifest_records = []

    # Shift range scans (each mode handled separately; never combined in same plots)
    for interval_max_len in INTERVAL_MAX_LENS:
        for mode in shift_modes:
            if mode == 'vary_min':
                fixed_value = FIXED_MAX_WHEN_VARY_MIN
            else:  # vary_max
                fixed_value = FIXED_MIN_WHEN_VARY_MAX
            print(f"Running shift range scan mode={mode}, interval_max_len={interval_max_len}")
            agg_dir, summary_df = run_shift_range_scan(
                config_path,
                mode=mode,
                start=shift_start,
                end=shift_end,
                step=shift_step,
                fixed_value=fixed_value,
                interval_max_len_override=interval_max_len,
            )
            summary_csv = os.path.join(agg_dir, 'shift_range_summary.csv')
            manifest_records.append({
                'group': 'shift_range',
                'mode': mode,
                'interval_max_len': interval_max_len,
                'aggregate_dir': agg_dir,
                'summary_csv': summary_csv,
                'n_rows': len(summary_df)
            })

    # Slope scans (still may combine multiple scale modes inside dedicated run)
    for interval_max_len in INTERVAL_MAX_LENS:
        print(f"Running slope scan interval_max_len={interval_max_len} modes={SLOPE_SCALE_MODES}")
        agg_dir, summary_df = run_slope_scan(
            config_path,
            slope_start=slope_start,
            slope_end=slope_end,
            slope_step=slope_step,
            scale_modes=SLOPE_SCALE_MODES,
            interval_max_len_override=interval_max_len,
        )
        summary_csv = os.path.join(agg_dir, 'slope_scan_summary.csv')
        manifest_records.append({
            'group': 'slope_scan',
            'mode': ','.join(SLOPE_SCALE_MODES),
            'interval_max_len': interval_max_len,
            'aggregate_dir': agg_dir,
            'summary_csv': summary_csv,
            'n_rows': len(summary_df)
        })

    manifest_df = pd.DataFrame(manifest_records)
    manifest_path = os.path.join(suite_dir, 'full_suite_manifest.csv')
    manifest_df.to_csv(manifest_path, index=False)
    print(f"Full suite complete. Manifest written to {manifest_path}")
    return suite_dir, manifest_df


def main():
    parser = argparse.ArgumentParser(description="Run full experiment suite (shift range & slope scans).")
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG, help='Base config path')
    parser.add_argument('--shift-start', type=float, default=SHIFT_START, help='Shift scan start value')
    parser.add_argument('--shift-end', type=float, default=SHIFT_END, help='Shift scan end value')
    parser.add_argument('--shift-step', type=float, default=SHIFT_STEP, help='Shift scan step size')
    parser.add_argument('--slope-start', type=float, default=SLOPE_START, help='Slope scan start value')
    parser.add_argument('--slope-end', type=float, default=SLOPE_END, help='Slope scan end value')
    parser.add_argument('--slope-step', type=float, default=SLOPE_STEP, help='Slope scan step size')
    parser.add_argument('--shift-mode', type=str, choices=SHIFT_RANGE_MODES, help='Run only this single shift range mode (omit to run both).')
    parser.add_argument('--both-shift-modes', action='store_true', help='Explicitly run both shift modes even if --shift-mode provided.')
    parser.add_argument('--fast', action='store_true', help='Use fast reduced episodes/epochs if supported (override by editing base config).')
    args = parser.parse_args()

    # Decide shift modes list
    if args.shift_mode and not args.both_shift_modes:
        shift_modes = [args.shift_mode]
    else:
        shift_modes = SHIFT_RANGE_MODES

    run_suite(config_path=args.config,
              shift_start=args.shift_start,
              shift_end=args.shift_end,
              shift_step=args.shift_step,
              slope_start=args.slope_start,
              slope_end=args.slope_end,
              slope_step=args.slope_step,
              fast=args.fast,
              shift_modes=shift_modes)


if __name__ == '__main__':
    main()
