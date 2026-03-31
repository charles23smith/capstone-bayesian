import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bayesian_recovery import (
    ALL_SHOT_WINDOWS_NS,
    DATA_DIR,
    RANDOM_SEED,
    compute_new_t0_info,
    direct_fit_windowed_shot,
    ns_to_s,
    regress_parameter,
    rise_then_drop_recovery_logtime_np,
)


OUT_SUMMARY_CSV = Path("hill_delay_drop_summary.csv")
OUT_LOO_CSV = Path("hill_delay_drop_loo_results.csv")
OUT_WINDOW_TIMES_CSV = Path("hill_delay_drop_window_times.csv")
ACTIVE_SHOT_IDS = [27272, 27276, 27277, 27278, 27279]
SHOT_DOSES = {
    27272: 1.28e10,
    27276: 1.40e10,
    27277: 1.06e10,
    27278: 1.16e10,
    27279: 6.22e10,
}
SHOT_WINDOWS_NS = {shot_id: ALL_SHOT_WINDOWS_NS[shot_id] for shot_id in ACTIVE_SHOT_IDS}
FULL_SHOT_WINDOWS_NS = {
    shot_id: (75.0, end_ns)
    for shot_id, (_, end_ns) in SHOT_WINDOWS_NS.items()
}
NEIGHBOR_COUNT = 3
EXCLUDED_TRAIN_SHOT_IDS = set()


def parse_holdout():
    parser = argparse.ArgumentParser(description="Bayesian/hybrid hill-delay-drop leave-one-out model.")
    parser.add_argument("holdout_shot_id", nargs="?", type=int, default=None)
    parser.add_argument("--full", action="store_true", help="Use the broader 75 ns to current endpoint windows.")
    args, unknown = parser.parse_known_args()

    holdout_shot_id = args.holdout_shot_id
    if holdout_shot_id is None:
        for token in unknown:
            if token.startswith("--") and token[2:].isdigit():
                holdout_shot_id = int(token[2:])
                break
    if holdout_shot_id is None:
        rng = random.Random(RANDOM_SEED)
        holdout_shot_id = rng.choice(list(ACTIVE_SHOT_IDS))
        print(f"Random holdout selected: {holdout_shot_id}")
    holdout_shot_id = int(holdout_shot_id)
    if holdout_shot_id not in ACTIVE_SHOT_IDS:
        raise ValueError(f"holdout shot must be one of {ACTIVE_SHOT_IDS}")
    return holdout_shot_id, args.full


def load_windowed_shot_local(shot_id: int, windows_ns):
    start_ns, end_ns = windows_ns[shot_id]
    info = compute_new_t0_info(DATA_DIR / f"{shot_id}_data.csv", shot_id)
    t_diode_abs_s = info["t_diode_abs_s"]
    v_diode = info["v_diode"]
    new_t0_abs_s = float(info["new_t0_abs_s"])

    window_start_abs_s = float(new_t0_abs_s + ns_to_s(start_ns))
    window_end_abs_s = float(new_t0_abs_s + ns_to_s(end_ns))
    mask = (t_diode_abs_s >= window_start_abs_s) & (t_diode_abs_s <= window_end_abs_s)
    if np.count_nonzero(mask) < 8:
        raise ValueError(f"window contains too few points for shot {shot_id}")

    t_win_abs_s = np.asarray(t_diode_abs_s[mask], dtype=float)
    t_win = np.asarray((t_win_abs_s - new_t0_abs_s) * 1e9, dtype=float)
    v_win = np.asarray(v_diode[mask], dtype=float)
    sign_flipped = abs(float(np.min(v_win))) > abs(float(np.max(v_win)))
    if sign_flipped:
        v_win = -v_win

    tail_n = max(8, min(64, len(v_win) // 5))
    baseline = float(np.median(v_win[-tail_n:]))
    t_fit_ns = t_win - float(t_win[0])
    keep = np.isfinite(t_fit_ns) & np.isfinite(v_diode[mask]) & (t_fit_ns > 0.0)
    if np.count_nonzero(keep) < 8:
        raise ValueError(f"window has too few positive-time fit points for shot {shot_id}")

    t_fit_ns = np.asarray(t_fit_ns[keep], dtype=float)
    t_window_rel_ns = np.asarray(t_win[keep], dtype=float)
    v_obs_raw = np.asarray(v_diode[mask][keep], dtype=float)
    return {
        "t_ns": t_fit_ns,
        "t_window_rel_ns": t_window_rel_ns,
        "log10_t_ns": np.log10(t_fit_ns),
        "v_obs_raw": v_obs_raw,
        "baseline": baseline,
        "new_t0_abs_s": new_t0_abs_s,
        "window_start_abs_s": window_start_abs_s,
        "window_end_abs_s": window_end_abs_s,
    }


def _median_abs_dev(values):
    values = np.asarray(values, dtype=float)
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    return med, max(mad, 1e-6)


def select_shape_neighbors(train_fit_rows, holdout_log_dose: float, holdout_v_start_anchor: float):
    if len(train_fit_rows) <= NEIGHBOR_COUNT:
        return train_fit_rows

    _, _, pred_t_half = regress_parameter(train_fit_rows, "t_half", holdout_log_dose)
    _, _, pred_n = regress_parameter(train_fit_rows, "n", holdout_log_dose)
    _, _, pred_t_drop = regress_parameter(train_fit_rows, "t_drop", holdout_log_dose)
    _, _, pred_k_drop = regress_parameter(train_fit_rows, "k_drop", holdout_log_dose)

    _, scale_v_start = _median_abs_dev([row["v_start"] for row in train_fit_rows])
    _, scale_t_half = _median_abs_dev([row["t_half"] for row in train_fit_rows])
    _, scale_n = _median_abs_dev([row["n"] for row in train_fit_rows])
    _, scale_t_drop = _median_abs_dev([row["t_drop"] for row in train_fit_rows])
    _, scale_k_drop = _median_abs_dev([row["k_drop"] for row in train_fit_rows])

    def score(row):
        return (
            0.5 * abs(row["v_start"] - holdout_v_start_anchor) / scale_v_start
            + abs(row["t_half"] - pred_t_half) / scale_t_half
            + abs(row["n"] - pred_n) / scale_n
            + abs(row["t_drop"] - pred_t_drop) / scale_t_drop
            + abs(row["k_drop"] - pred_k_drop) / scale_k_drop
        )

    return sorted(train_fit_rows, key=score)[:NEIGHBOR_COUNT]

def main():
    holdout_shot_id, use_full_windows = parse_holdout()
    shot_ids = list(ACTIVE_SHOT_IDS)
    windows_ns = FULL_SHOT_WINDOWS_NS if use_full_windows else SHOT_WINDOWS_NS
    shot_data = {}

    for shot_id in shot_ids:
        loaded = load_windowed_shot_local(shot_id, windows_ns)
        loaded["log_dose"] = float(np.log(SHOT_DOSES[shot_id]))
        shot_data[shot_id] = loaded

    print("Dose values:")
    for shot_id in shot_ids:
        print(f"{shot_id}: {SHOT_DOSES[shot_id]:.6e}")
    print(f"Window mode: {'full' if use_full_windows else 'default'}")

    train_shot_ids = [
        shot_id
        for shot_id in shot_ids
        if shot_id != holdout_shot_id and shot_id not in EXCLUDED_TRAIN_SHOT_IDS
    ]
    print(f"\nLOO holdout: {holdout_shot_id} | train on {train_shot_ids}")
    print("Method: bayesian_recovery hill+delay-drop hybrid LOO")

    train_fit_rows = [direct_fit_windowed_shot(shot_id, shot_data) for shot_id in train_shot_ids]
    holdout_v_start_anchor = float(shot_data[holdout_shot_id]["v_obs_raw"][0])
    selected_rows = select_shape_neighbors(train_fit_rows, shot_data[holdout_shot_id]["log_dose"], holdout_v_start_anchor)
    summary_df = pd.DataFrame(train_fit_rows)
    summary_df["used_for_regression"] = summary_df["shot_id"].isin([row["shot_id"] for row in selected_rows])
    summary_df.to_csv(OUT_SUMMARY_CSV, index=False)

    log_dose = shot_data[holdout_shot_id]["log_dose"]
    v_start_hat = holdout_v_start_anchor
    _, _, amp_rise_hat = regress_parameter(selected_rows, "amp_rise", log_dose)
    _, _, t_half_hat = regress_parameter(selected_rows, "t_half", log_dose)
    _, _, n_hat = regress_parameter(selected_rows, "n", log_dose)
    _, _, amp_drop_hat = regress_parameter(selected_rows, "amp_drop", log_dose)
    _, _, t_drop_hat = regress_parameter(selected_rows, "t_drop", log_dose)
    _, _, k_drop_hat = regress_parameter(selected_rows, "k_drop", log_dose)

    # Keep regressed shape parameters in a physically meaningful range.
    amp_rise_hat = float(max(1e-4, amp_rise_hat))
    t_half_hat = float(max(1e-3, t_half_hat))
    n_hat = float(max(0.05, n_hat))
    amp_drop_hat = float(max(0.0, amp_drop_hat))
    t_drop_hat = float(max(5.0, t_drop_hat))
    k_drop_hat = float(max(0.5, k_drop_hat))

    log10_t_ns = shot_data[holdout_shot_id]["log10_t_ns"]
    v_obs_raw = shot_data[holdout_shot_id]["v_obs_raw"]
    y_hat = rise_then_drop_recovery_logtime_np(
        log10_t_ns,
        v_start_hat,
        amp_rise_hat,
        t_half_hat,
        n_hat,
        amp_drop_hat,
        t_drop_hat,
        k_drop_hat,
    )

    rmse = float(np.sqrt(np.mean((y_hat - v_obs_raw) ** 2)))
    ss_tot = float(np.sum((v_obs_raw - np.mean(v_obs_raw)) ** 2))
    r_squared = float(1.0 - np.sum((v_obs_raw - y_hat) ** 2) / ss_tot) if ss_tot > 0 else np.nan
    sigma_hat = float(np.std(v_obs_raw - y_hat, ddof=1)) if len(v_obs_raw) > 1 else float("nan")

    loo_df = pd.DataFrame(
        [
            {
                "holdout_shot": holdout_shot_id,
                "train_shots": " ".join(str(x) for x in train_shot_ids),
                "used_train_shots": " ".join(str(row["shot_id"]) for row in selected_rows),
                "dose_rate": SHOT_DOSES[holdout_shot_id],
                "v_start_pred": v_start_hat,
                "amp_rise_pred": amp_rise_hat,
                "t_half_pred": t_half_hat,
                "n_pred": n_hat,
                "amp_drop_pred": amp_drop_hat,
                "t_drop_pred": t_drop_hat,
                "k_drop_pred": k_drop_hat,
                "sigma_fit": sigma_hat,
                "rmse": rmse,
                "r_squared": r_squared,
                "n_points": int(len(v_obs_raw)),
            }
        ]
    )
    loo_df.to_csv(OUT_LOO_CSV, index=False)

    window_rows = []
    window_source = FULL_SHOT_WINDOWS_NS if use_full_windows else SHOT_WINDOWS_NS
    for shot_id, (start_ns, end_ns) in window_source.items():
        info = compute_new_t0_info(DATA_DIR / f"{shot_id}_data.csv", shot_id)
        new_t0_abs_s = float(info["new_t0_abs_s"])
        window_rows.append(
            {
                "shot_id": shot_id,
                "window_start_rel_ns": float(start_ns),
                "window_end_rel_ns": float(end_ns),
                "new_t0_abs_s": new_t0_abs_s,
                "window_start_abs_s": float(new_t0_abs_s + ns_to_s(start_ns)),
                "window_end_abs_s": float(new_t0_abs_s + ns_to_s(end_ns)),
            }
        )
    pd.DataFrame(window_rows).sort_values("shot_id").to_csv(OUT_WINDOW_TIMES_CSV, index=False)

    t_plot_ns = shot_data[holdout_shot_id]["t_window_rel_ns"]
    out_png = Path(f"hill_delay_drop_{holdout_shot_id}_fit.png")
    out_png_linear = Path(f"hill_delay_drop_{holdout_shot_id}_fit_linear.png")

    plt.figure(figsize=(8, 5))
    plt.plot(t_plot_ns, v_obs_raw, color="black", linewidth=1.2, label="Observed")
    plt.plot(t_plot_ns, y_hat, color="red", linewidth=2.0, label="LOO posterior mean fit")
    plt.xscale("log")
    plt.xlabel("Time after NEW t0 (ns)")
    plt.ylabel("Diode voltage (V)")
    plt.title(f"Hill Delay-Drop LOO Fit: Shot {holdout_shot_id}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(t_plot_ns, v_obs_raw, color="black", linewidth=1.2, label="Observed")
    plt.plot(t_plot_ns, y_hat, color="red", linewidth=2.0, label="LOO posterior mean fit")
    plt.xlabel("Time after NEW t0 (ns)")
    plt.ylabel("Diode voltage (V)")
    plt.title(f"Hill Delay-Drop LOO Fit: Shot {holdout_shot_id} (Linear Time)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png_linear, dpi=150)
    plt.close()

    print(
        f"Predicted holdout {holdout_shot_id}: using shots {[row['shot_id'] for row in selected_rows]} | "
        f"v_start={v_start_hat:.4f} amp_rise={amp_rise_hat:.4f} "
        f"t_half={t_half_hat:.4f} n={n_hat:.4f} "
        f"amp_drop={amp_drop_hat:.4f} t_drop={t_drop_hat:.4f} k_drop={k_drop_hat:.4f} "
        f"RMSE={rmse:.6f} R^2={r_squared:.6f}"
    )
    print(f"Saved: {OUT_SUMMARY_CSV}")
    print(f"Saved: {OUT_WINDOW_TIMES_CSV}")
    print(f"Saved: {OUT_LOO_CSV}")
    print(loo_df.to_string(index=False))
    print(f"Saved: {out_png}")
    print(f"Saved: {out_png_linear}")


if __name__ == "__main__":
    main()
