# Running `hill_delay_drop.py`

This document explains how to run [hill_delay_drop.py](/c:/Users/miste/OneDrive/Documents/GitHub/capstone-bayesian/hill_delay_drop.py), what input files it expects, and which test files must exist in the `data` folder.

## What The Script Does

`hill_delay_drop.py` runs a leave-one-out fit for the hill-delay-drop recovery model.

It:
- loads diode waveform data for a fixed set of shots
- computes a shifted `t0` for each shot
- fits the training shots
- holds one shot out for prediction
- writes CSV summaries and PNG plots

## Script Section Notes

The notes in this README explain the main sections of `hill_delay_drop.py` so the code structure is easier to follow.

### 1. Time Alignment (`t0`)

The waveform timing is not taken directly from the raw CSV start time.

Instead, the script uses `compute_new_t0_info(...)` from `bayesian_recovery.py`, which defines a new common start time using the PCD signal:

- the PCD channels are averaged
- the averaged PCD waveform is smoothed
- the code finds the time where that averaged PCD reaches `0.5 V`
- then it shifts that crossing time by `100 ns`

So the effective alignment is:

```text
new_t0 = time when average PCD reaches 0.5 V - 100 ns
```

This gives all shots a more uniform reference time before fitting.

### 2. Windowing Is Relative To The Shifted `t0`

The fit window is not applied to the raw absolute time axis.

For each shot, the code defines:

- `window_start_abs_s = new_t0 + start_ns`
- `window_end_abs_s = new_t0 + end_ns`

That means every fit window is chosen relative to the shifted `t0`, not relative to the original file start time.

In practice, this means the model is always looking at the same physical part of the waveform after a common event marker.

### 3. Why The Window Is Chosen

The selected window is meant to represent the diode behavior after the main peak, especially the recovery and return toward baseline.

The goal is to isolate the part of the waveform where:

- the main response has already occurred
- the trace is relaxing after peak behavior
- the signal shape can be modeled as a rise/recovery followed by a delayed drop or return toward baseline

So the window is chosen to capture the physically meaningful recovery region rather than the entire raw waveform.

### 4. What Happens Inside The Window

After the window is extracted:

- the signal can be sign-flipped so shots have a consistent orientation
- a tail baseline is estimated from the end of the selected window
- time is shifted again so the fit starts at `0 ns` within that window
- only positive-time samples are kept for fitting

This creates a consistent fit region across all included shots.

### 5. Leave-One-Out Workflow

The script uses a leave-one-out workflow:

- one shot is chosen as the holdout
- the remaining shots are directly fit first
- a small set of similar training shots is selected
- the holdout parameters are estimated from dose-based regression on those selected training shots

The predicted holdout waveform is then compared against the actual waveform in the chosen window.

## Required Folder Layout

Run the script from the repository root:

```text
capstone-bayesian/
├── hill_delay_drop.py
├── bayesian_recovery.py
├── bruteForce.py
└── data/
    ├── 27272_data.csv
    ├── 27276_data.csv
    ├── 27277_data.csv
    ├── 27278_data.csv
    └── 27279_data.csv
```

## Required Input File Naming

The script expects waveform files in this format:

```text
data/<shot_id>_data.csv
```

Examples:
- `data/27272_data.csv`
- `data/27276_data.csv`
- `data/27279_data.csv`

If the filename does not match that pattern, the script will not find the file.

## Which Test Files Must Be In `data/`

The current code uses these shot IDs:

- `27272`
- `27276`
- `27277`
- `27278`
- `27279`

That means these files must exist:

- `data/27272_data.csv`
- `data/27276_data.csv`
- `data/27277_data.csv`
- `data/27278_data.csv`
- `data/27279_data.csv`

## Python Dependencies

At minimum, the script depends on the packages used by `hill_delay_drop.py`, `bayesian_recovery.py`, and `bruteForce.py`.

Typical required packages:
- `numpy`
- `pandas`
- `matplotlib`
- `scipy`
- `pymc`
- `pytensor`
- `arviz`

## How To Run In The Terminal

Run with the default behavior:

```powershell
python hill_delay_drop.py
```

This will:
- choose a random holdout shot from the allowed set
- train on the remaining shots
- save CSV and plot outputs in the repo root

Run with a specific holdout shot:

```powershell
python hill_delay_drop.py 27279
```

## Output Files

A successful run writes files like:

- `hill_delay_drop_summary.csv`
- `hill_delay_drop_loo_results.csv`
- `hill_delay_drop_window_times.csv`
- `hill_delay_drop_<shot_id>_fit.png`
- `hill_delay_drop_<shot_id>_fit_linear.png`

Example:
- `hill_delay_drop_27279_fit.png`
- `hill_delay_drop_27279_fit_linear.png`

## Notes

- The script name in this repo is `hill_delay_drop.py`, not `delay_hill_drop.py`.
- `shot_data.csv` is not required for this script.
- The `data` directory is required because the script reads each shot from `DATA_DIR / f"{shot_id}_data.csv"`.
- The explanatory comments requested for `hill_delay_drop.py` are documented here in the README rather than being embedded directly in the code.
