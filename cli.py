import argparse
import yaml
from run_experiment import run_from_config
from plots.multirun_slope import run_slope_scan


def main():
    parser = argparse.ArgumentParser(description="Run time-based agents experiment from a YAML config.")
    parser.add_argument("config", nargs="?", default="configs/config.yaml", help="Path to YAML configuration file")
    parser.add_argument("--episodes", type=int, help="Override episodes")
    parser.add_argument("--eval-steps", type=int, help="Override eval steps")
    parser.add_argument("--epochs", type=int, help="Override epochs")
    parser.add_argument("--output", type=str, help="Override output CSV filename")
    parser.add_argument("--slope-scan", action="store_true", help="Run multi-run slope scan (0.1..2.0 step 0.1)")
    parser.add_argument("--slope-start", type=float, default=0.1, help="Slope scan start value")
    parser.add_argument("--slope-end", type=float, default=2.0, help="Slope scan end value")
    parser.add_argument("--slope-step", type=float, default=0.1, help="Slope scan step size")
    args = parser.parse_args()

    if args.slope_scan:
        run_slope_scan(args.config, slope_start=args.slope_start, slope_end=args.slope_end, slope_step=args.slope_step)
        return

    if any([args.episodes, args.eval_steps, args.epochs, args.output]):
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
        exp = cfg.setdefault('experiment', {})
        if args.episodes is not None:
            exp['episodes'] = args.episodes
        if args.eval_steps is not None:
            exp['eval_steps'] = args.eval_steps
        if args.epochs is not None:
            exp['epochs'] = args.epochs
        if args.output is not None:
            exp['output_csv'] = args.output
        patched_path = args.config + ".patched"
        with open(patched_path, 'w') as f:
            yaml.safe_dump(cfg, f)
        run_from_config(patched_path)
    else:
        run_from_config(args.config)

if __name__ == "__main__":
    main()
