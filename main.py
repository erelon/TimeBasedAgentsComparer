import sys
from run_experiment import run_from_config
from plots.multirun_slope import run_slope_scan  # optional multi-run slope scan
from plots.multirun_shift_range import run_shift_range_scan  # new: shift range scan

# ---------------------------------------------------------------------------
# Manual toggles (edit these directly to change behavior; no CLI args needed)
# ---------------------------------------------------------------------------
# Select exactly one of the following modes by setting its flag to True.
# Precedence order below: SHIFT RANGE > SLOPE > SINGLE RUN.
RUN_SHIFT_RANGE_SCAN = True  # Set True to execute shift range sweep (ShiftingUnevenTwoStates)
RUN_SLOPE_SCAN = False  # Set True to execute slope sweep instead of single run
# (If both above are False) -> Single run from config file

# --- Slope scan parameters --------------------------------------------------
SLOPE_START = 0.1  # Starting slope value
SLOPE_END = 2.0  # Ending slope value
SLOPE_STEP = 0.05  # Increment

# --- Shift range scan parameters -------------------------------------------
# Mode options: 'vary_min' (vary _shift_min keep _shift_max fixed) or 'vary_max'
# SHIFT_RANGE_MODE = 'vary_max'
# SHIFT_START = 1.0  # Start value for the varied parameter
# SHIFT_END = 2.0  # End value for the varied parameter
# SHIFT_STEP = 0.1  # Step size for the varied parameter
# SHIFT_FIXED_VALUE = 1.0  # Fixed counterpart (e.g. fixed _shift_max when varying _shift_min)

SHIFT_RANGE_MODE = 'vary_min'
SHIFT_START = 0.1  # Start value for the varied parameter
SHIFT_END = 1.0  # End value for the varied parameter
SHIFT_STEP = 0.1  # Step size for the varied parameter
SHIFT_FIXED_VALUE = 1.0  # Fixed counterpart (e.g. fixed _shift_max when varying _shift_min)

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = 'configs/config.yaml'
    # cfg = 'configs/config_harmonic_compare.yaml'
    # cfg = 'configs/config_smart_compare.yaml'
    if len(sys.argv) > 1:  # still allow overriding config path if provided
        cfg = sys.argv[1]

    if RUN_SHIFT_RANGE_SCAN:
        run_shift_range_scan(
            cfg,
            mode=SHIFT_RANGE_MODE,
            start=SHIFT_START,
            end=SHIFT_END,
            step=SHIFT_STEP,
            fixed_value=SHIFT_FIXED_VALUE,
        )
    elif RUN_SLOPE_SCAN:
        run_slope_scan(cfg, slope_start=SLOPE_START, slope_end=SLOPE_END, slope_step=SLOPE_STEP)
    else:
        run_from_config(cfg)
