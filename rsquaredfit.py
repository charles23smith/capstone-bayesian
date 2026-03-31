"""
Fit a power-law recovery model on shot-specific windows and report R^2.

Model used exactly as requested:
    V(t) = V0 / (1 + t / t_half)^n

Notes:
- This script uses the same NEW-t0 timing definition as bruteForce.py so the
  window CSV is interpreted in the same relative-time coordinates.
- For each fit window, the waveform is polarity-corrected so the dominant
  recovery lobe is positive, and the end-of-window baseline is removed before
  fitting because the requested model asymptotes to 0.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from bruteForce import (
    PCD_TARGET_V,
    T0_SHIFT_NS,
    T0_SHIFT_NS_BY_SHOT,
    build_pcd_avg,
    first_crossing_time_or_nearest,
    load_wide_csv,
    smooth_by_ns,
)


DEFAULT_WINDOWS_CSV = Path(r"E:\window for bayesian R^2 - Sheet1.csv")
DEFAULT_DATA_DIR = Path("data")
DEFAULT_OUT_CSV = Path("rsquared_results.csv")


def ns_to_s(ns: float) -> float:
    return float(ns) * 1e-9


def s_to_ns(s: np.ndarray) -> np.ndarray:
    return np.asarray(s, dtype=float) * 1e9


def hall_like_recovery(t_ns: np.ndarray, v0: float, t_half_ns: float, n: float) -> np.ndarray:
    t_ns = np.asarray(t_ns, dtype=float)
    t_half_ns = max(float(t_half_ns), 1e-12)
    n = max(float(n), 1e-12)
    return float(v0) / np.power(1.0 + np.maximum(t_ns, 0.0) / t_half_ns, n)


def hall_like_recovery_logtime(log10_t_ns: np.ndarray, v0: float, t_half_ns: float, n: float) -> np.ndarray:
    log10_t_ns = np.asarray(log10_t_ns, dtype=float)
    t_ns = np.power(10.0, log10_t_ns)
    return hall_like_recovery(t_ns, v0, t_half_ns, n)


def compute_new_t0_abs(csv_file: Path, shot_id: int) -> tuple[np.ndarray, np.ndarray]:
    series = load_wide_csv(str(csv_file))
    t_diode, v_diode = series["Diode"]
    pcd_avg, _ = build_pcd_avg(series, t_diode)
    pcd_s, _ = smooth_by_ns(t_diode, pcd_avg, 3.0)
    t_cross, _, _ = first_crossing_time_or_nearest(t_diode, pcd_s, PCD_TARGET_V)
    t0_shift_ns = float(T0_SHIFT_NS_BY_SHOT.get(shot_id, T0_SHIFT_NS))
    new_t0_abs = float(t_cross - ns_to_s(t0_shift_ns))
    t_rel_ns = s_to_ns(t_diode - new_t0_abs)
    return t_rel_ns, np.asarray(v_diode, dtype=float)


def prepare_window_data(t_rel_ns: np.ndarray, v_diode: np.ndarray, start_ns: float, end_ns: float):
    mask = (t_rel_ns >= float(start_ns)) & (t_rel_ns <= float(end_ns))
    if np.count_nonzero(mask) < 8:
        raise ValueError("window contains too few points")

    t_win = np.asarray(t_rel_ns[mask], dtype=float)
    v_win = np.asarray(v_diode[mask], dtype=float)

    # Make the dominant recovery lobe positive.
    if abs(float(np.min(v_win))) > abs(float(np.max(v_win))):
        v_win = -v_win

    # Remove small endpoint bias so the requested model can decay toward 0.
    tail_n = max(5, min(25, len(v_win) // 5))
    baseline = float(np.median(v_win[-tail_n:]))
    v_win = v_win - baseline

    # Shift the fit window so the model starts at t=0.
    t_fit_ns = t_win - float(t_win[0])

    keep = np.isfinite(t_fit_ns) & np.isfinite(v_win) & (t_fit_ns > 0.0)
    if np.count_nonzero(keep) < 6:
        raise ValueError("window does not contain enough positive-time points for log10(time) fitting")

    return t_fit_ns[keep], v_win[keep], baseline


def fit_window(t_fit_ns: np.ndarray, v_fit: np.ndarray):
    duration_ns = max(float(t_fit_ns[-1] - t_fit_ns[0]), 1.0)
    v0_guess = max(float(v_fit[0]), float(np.max(v_fit)))
    p0 = [v0_guess, max(duration_ns * 0.25, 1.0), 1.0]
    lb = [0.0, 1e-6, 1e-3]
    ub = [max(float(np.max(v_fit)) * 5.0, 1.0), max(duration_ns * 10.0, 10.0), 12.0]

    log10_t_fit_ns = np.log10(t_fit_ns)

    popt, _ = curve_fit(
        hall_like_recovery_logtime,
        log10_t_fit_ns,
        v_fit,
        p0=p0,
        bounds=(lb, ub),
        maxfev=50000,
    )
    y_hat = hall_like_recovery_logtime(log10_t_fit_ns, *popt)
    ss_res = float(np.sum((v_fit - y_hat) ** 2))
    ss_tot = float(np.sum((v_fit - np.mean(v_fit)) ** 2))
    r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan")
    return popt, y_hat, r_squared


def main():
    parser = argparse.ArgumentParser(description="Fit power-law recovery windows and report R^2.")
    parser.add_argument("--windows-csv", default=str(DEFAULT_WINDOWS_CSV), help="CSV containing test,start time (ns),end time (ns)")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Directory containing <shot>_data.csv files")
    parser.add_argument("--out-csv", default=str(DEFAULT_OUT_CSV), help="Output CSV for fitted parameters and R^2")
    args = parser.parse_args()

    windows_csv = Path(args.windows_csv)
    data_dir = Path(args.data_dir)
    out_csv = Path(args.out_csv)

    windows = pd.read_csv(windows_csv)
    needed = {"test", "start time (ns)", "end time (ns)"}
    missing = needed.difference(windows.columns)
    if missing:
        raise ValueError(f"{windows_csv} is missing required columns: {sorted(missing)}")

    results = []
    for _, row in windows.iterrows():
        shot_id = int(row["test"])
        start_ns = float(row["start time (ns)"])
        end_ns = float(row["end time (ns)"])
        csv_file = data_dir / f"{shot_id}_data.csv"

        result = {
            "test": shot_id,
            "start_time_ns": start_ns,
            "end_time_ns": end_ns,
            "csv_file": str(csv_file),
        }

        try:
            t_rel_ns, v_diode = compute_new_t0_abs(csv_file, shot_id)
            t_fit_ns, v_fit, baseline = prepare_window_data(t_rel_ns, v_diode, start_ns, end_ns)
            (v0, t_half_ns, n), _, r_squared = fit_window(t_fit_ns, v_fit)
            result.update(
                {
                    "points_fit": int(len(t_fit_ns)),
                    "baseline_removed_v": baseline,
                    "V0_fit": float(v0),
                    "t_half_ns_fit": float(t_half_ns),
                    "n_fit": float(n),
                    "r_squared": float(r_squared),
                    "status": "ok",
                }
            )
        except Exception as exc:
            result.update(
                {
                    "points_fit": 0,
                    "baseline_removed_v": np.nan,
                    "V0_fit": np.nan,
                    "t_half_ns_fit": np.nan,
                    "n_fit": np.nan,
                    "r_squared": np.nan,
                    "status": f"error: {exc}",
                }
            )

        results.append(result)

    out_df = pd.DataFrame(results)
    out_df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")
    print(out_df[["test", "V0_fit", "t_half_ns_fit", "n_fit", "r_squared", "status"]].to_string(index=False))

    strong = out_df[(out_df["status"] == "ok") & (out_df["r_squared"] >= 0.95)].copy()
    print("\nShots with R^2 >= 0.95:")
    if strong.empty:
        print("None")
    else:
        print(strong[["test", "r_squared"]].to_string(index=False))


if __name__ == "__main__":
    main()
