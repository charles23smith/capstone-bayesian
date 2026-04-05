import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from bayesian_recovery import (
    ALL_SHOT_WINDOWS_NS,
    DATA_DIR,
    RANDOM_SEED,
    compute_new_t0_info,
    ns_to_s,
    regress_parameter,
    rise_then_drop_recovery_logtime_np,
)
from bruteForce import smooth_by_ns

OUTPUT_DIR = Path("loo_png_outputs") / "hill_delay_drop"

# Strong delay-hill shots from the comparison sheet, plus the shared
# exponential-fit shots that also behave like simple recoveries here.
# Mid fits are kept intentionally and called out inline; shots marked bad
# on the sheet are excluded.
ACTIVE_SHOT_IDS = [
    27271,
    27272,
    27277,
    27278,
    27279,
    27280,
    27282,
    27283,
    27290,
    27291,
    27294,
]
SHARED_EXP_SHOT_IDS = {
    27271,
    27280,
    27282,
    27283,
    27290,
    27291,
    27294,
}
NEIGHBOR_FAMILY_BY_SHOT = {
    27271: "exp_group",
    27280: "exp_group",
    27282: "exp_group",
    27283: "exp_group",
    27290: "exp_group",
    27291: "exp_group",
    27294: "exp_group",
    27272: "delay_core_group",
    27277: "delay_core_group",
    27278: "delay_core_group",
    27279: "delay_core_group",
}
SHOT_NOTES = {
    27271: "shared_exp",
    27280: "shared_exp",
    27282: "shared_exp",
    27283: "shared_exp",
    27290: "shared_exp",
    27291: "shared_exp",
    27294: "shared_exp",
}
SHOT_DOSES = {
    27271: 1.28e10,
    27272: 1.28e10,
    27273: 1.25e10,
    27276: 1.40e10,
    27277: 1.06e10,
    27278: 1.16e10,
    27279: 6.22e10,
    27280: 5.63e10,
    27282: 6.41e10,
    27283: 8.11e10,
    27285: 4.86e10,
    27286: 4.74e11,
    27290: 1.22e10,
    27291: 9.31e9,
    27294: 5.43e10,
    27295: 5.44e10,
    27296: 4.81e11,
    27297: 6.10e11,
    27298: 1.21e10,
    27299: 1.05e10,
    27300: 5.22e10,
    27301: 5.46e11,
    27302: 8.88e11,
    27303: 1.01e10,
    27304: 6.23e10,
    27305: 5.56e11,
    27306: 1.15e10,
    27307: 5.09e11,
}
SHOT_WINDOWS_NS = {
    # Existing delay-hill windows stay exactly as already established.
    27272: ALL_SHOT_WINDOWS_NS[27272],
    27277: ALL_SHOT_WINDOWS_NS[27277],
    27278: ALL_SHOT_WINDOWS_NS[27278],
    27279: ALL_SHOT_WINDOWS_NS[27279],
    # Added shots use the same windows shown in the model-comparison sheet.
    27271: (200.0, 314.0),
    27280: (200.0, 323.0),
    27282: (200.0, 323.0),
    27283: (200.0, 323.0),
    27290: (137.0, 233.0),
    27291: (139.0, 235.0),
    27294: (140.0, 600.0),
}
AMPLITUDE_REGRESSION_SHOT_IDS = [
    27271,
    27272,
    27273,
    27276,
    27277,
    27278,
    27279,
    27280,
    27282,
    27283,
    27285,
    27286,
    27288,
    27290,
    27291,
    27294,
    27295,
    27296,
    27297,
    27298,
    27299,
    27300,
    27301,
    27302,
    27303,
    27304,
    27305,
    27306,
    27307,
]
AMPLITUDE_WINDOWS_NS = {
    27271: (200.0, 314.0),
    27272: ALL_SHOT_WINDOWS_NS[27272],
    27273: (152.0, 215.0),
    27276: (205.0, 400.0),
    27277: ALL_SHOT_WINDOWS_NS[27277],
    27278: ALL_SHOT_WINDOWS_NS[27278],
    27279: ALL_SHOT_WINDOWS_NS[27279],
    27280: (200.0, 323.0),
    27282: (200.0, 323.0),
    27283: (200.0, 323.0),
    27285: (119.0, 216.0),
    27286: (100.0, 400.0),
    27288: (130.0, 268.0),
    27290: (137.0, 233.0),
    27291: (139.0, 235.0),
    27294: (140.0, 600.0),
    27295: (146.0, 266.0),
    27296: (165.0, 521.0),
    27297: (140.0, 325.0),
    27298: (131.0, 325.0),
    27299: (150.0, 274.0),
    27300: (127.0, 450.0),
    27301: (421.0, 800.0),
    27302: (261.0, 600.0),
    27303: (150.0, 1330.0),
    27304: (160.0, 1500.0),
    27305: (150.0, 2000.0),
    27306: (-39000.0, 40000.0),
    27307: (150.0, 70000.0),
}
FULL_SHOT_WINDOWS_NS = {
    shot_id: (75.0, end_ns)
    for shot_id, (_, end_ns) in SHOT_WINDOWS_NS.items()
}
# The regression step uses only the nearest few training shots in parameter space
# so one unusual waveform does not dominate the holdout prediction.
EXCLUDED_TRAIN_SHOT_IDS = set()
DEFAULT_FIT_SMOOTH_NS = 6.0
FIT_SMOOTH_NS_BY_SHOT = {shot_id: DEFAULT_FIT_SMOOTH_NS for shot_id in ACTIVE_SHOT_IDS}
ROBUST_PREFILTER_SHOT_IDS = set()


def robust_prefilter_local(y: np.ndarray, win: int = 9, mad_mult: float = 3.0):
    y = np.asarray(y, dtype=float).copy()
    if len(y) < 5:
        return y
    win = max(5, int(win))
    if win % 2 == 0:
        win += 1
    half = win // 2
    out = y.copy()
    for i in range(len(y)):
        a = max(0, i - half)
        b = min(len(y), i + half + 1)
        local = y[a:b]
        med = float(np.median(local))
        mad = float(np.median(np.abs(local - med))) + 1e-12
        if abs(y[i] - med) > mad_mult * 1.4826 * mad:
            out[i] = med
    return out


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
    # `compute_new_t0_info` aligns each shot to a common physical reference:
    # the time where the averaged PCD reaches 0.5 V, shifted earlier by 100 ns.
    info = compute_new_t0_info(DATA_DIR / f"{shot_id}_data.csv", shot_id)
    t_diode_abs_s = info["t_diode_abs_s"]
    v_diode = info["v_diode"]
    new_t0_abs_s = float(info["new_t0_abs_s"])

    # The fit window is defined relative to the shifted `new_t0`, not the raw CSV
    # start time, so every shot is evaluated on the same post-trigger region.
    window_start_abs_s = float(new_t0_abs_s + ns_to_s(start_ns))
    window_end_abs_s = float(new_t0_abs_s + ns_to_s(end_ns))
    mask = (t_diode_abs_s >= window_start_abs_s) & (t_diode_abs_s <= window_end_abs_s)
    if np.count_nonzero(mask) < 8:
        raise ValueError(f"window contains too few points for shot {shot_id}")

    t_win_abs_s = np.asarray(t_diode_abs_s[mask], dtype=float)
    t_win = np.asarray((t_win_abs_s - new_t0_abs_s) * 1e9, dtype=float)
    v_win_raw = np.asarray(v_diode[mask], dtype=float)
    if shot_id in ROBUST_PREFILTER_SHOT_IDS:
        # A few extreme spikes can dominate Savitzky-Golay smoothing and create
        # a false early hump; replace those outliers with the local median first.
        v_win_raw = robust_prefilter_local(v_win_raw, win=9, mad_mult=2.5)
    smooth_ns = float(FIT_SMOOTH_NS_BY_SHOT.get(shot_id, DEFAULT_FIT_SMOOTH_NS))
    v_win_fit, smooth_win = smooth_by_ns(t_win_abs_s, v_win_raw, smooth_ns)
    v_win_fit = np.asarray(v_win_fit, dtype=float)

    # Shift local fit time so the window begins at 0 ns, then keep only the
    # positive-time samples used by the log-time model.
    t_fit_ns = t_win - float(t_win[0])
    keep = np.isfinite(t_fit_ns) & np.isfinite(v_win_raw) & np.isfinite(v_win_fit) & (t_fit_ns > 0.0)
    if np.count_nonzero(keep) < 8:
        raise ValueError(f"window has too few positive-time fit points for shot {shot_id}")

    t_fit_ns = np.asarray(t_fit_ns[keep], dtype=float)
    t_window_rel_ns = np.asarray(t_win[keep], dtype=float)
    v_obs_raw = np.asarray(v_win_raw[keep], dtype=float)
    v_obs_fit = np.asarray(v_win_fit[keep], dtype=float)
    return {
        "t_ns": t_fit_ns,
        "t_window_rel_ns": t_window_rel_ns,
        "log10_t_ns": np.log10(t_fit_ns),
        "v_obs_raw": v_obs_raw,
        "v_obs_fit": v_obs_fit,
        "smooth_ns": smooth_ns,
        "smooth_win": int(smooth_win),
        "new_t0_abs_s": new_t0_abs_s,
        "window_start_abs_s": window_start_abs_s,
        "window_end_abs_s": window_end_abs_s,
    }


def direct_fit_windowed_shot_local(shot_id: int, shot_data):
    t_fit_ns = np.asarray(shot_data[shot_id]["t_ns"], dtype=float)
    v_obs_fit = np.asarray(shot_data[shot_id]["v_obs_fit"], dtype=float)

    v_start0 = float(np.clip(v_obs_fit[0], -5.0, 0.5))
    amp_rise0 = float(np.clip(max(float(np.max(v_obs_fit) - v_obs_fit[0]), 0.05), 0.0, 5.0))
    tail_n = max(5, min(25, len(v_obs_fit) // 10))
    amp_drop0 = float(np.clip(np.max(v_obs_fit) - np.median(v_obs_fit[-tail_n:]), 0.0, 3.0))
    t_end = float(t_fit_ns[-1])
    t_drop0 = float(min(max(0.65 * t_end, 20.0), t_end))
    k_drop0 = float(min(max(0.06 * t_end, 1.0), max(80.0, 0.5 * t_end)))
    p0 = [v_start0, amp_rise0, min(25.0, max(5.0, 0.15 * t_end)), 1.5, amp_drop0, t_drop0, k_drop0]
    bounds = (
        [-5.0, 0.0, 1e-3, 0.1, 0.0, 20.0, 1.0],
        [0.5, 5.0, max(400.0, t_end), 20.0, 3.0, t_end, max(80.0, 0.5 * t_end)],
    )
    popt, _ = curve_fit(
        rise_then_drop_recovery_logtime_np,
        np.log10(t_fit_ns),
        v_obs_fit,
        p0=p0,
        bounds=bounds,
        maxfev=200000,
    )
    y_hat = rise_then_drop_recovery_logtime_np(np.log10(t_fit_ns), *popt)
    rmse = float(np.sqrt(np.mean((y_hat - v_obs_fit) ** 2)))
    ss_tot = float(np.sum((v_obs_fit - np.mean(v_obs_fit)) ** 2))
    r_squared = float(1.0 - np.sum((v_obs_fit - y_hat) ** 2) / ss_tot) if ss_tot > 0 else np.nan
    v_start_obs = float(v_obs_fit[0])
    amp_scale = float(max(abs(v_start_obs), 1e-4))
    return {
        "shot_id": shot_id,
        "log_dose": float(shot_data[shot_id]["log_dose"]),
        "neighbor_family": NEIGHBOR_FAMILY_BY_SHOT[shot_id],
        "v_start": v_start_obs,
        "amp_rise": float(popt[1]),
        "amp_rise_rel_start": float(popt[1] / amp_scale),
        "t_half": float(popt[2]),
        "n": float(popt[3]),
        "amp_drop": float(popt[4]),
        "amp_drop_rel_start": float(popt[4] / amp_scale),
        "t_drop": float(popt[5]),
        "k_drop": float(popt[6]),
        "rmse_direct": rmse,
        "r_squared_direct": r_squared,
        "smooth_ns": float(shot_data[shot_id]["smooth_ns"]),
    }


def empirical_amplitude_row(shot_id: int, windows_ns):
    loaded = load_windowed_shot_local(shot_id, windows_ns)
    v_obs_fit = np.asarray(loaded["v_obs_fit"], dtype=float)
    if len(v_obs_fit) < 8:
        raise ValueError(f"too few points for amplitude extraction {shot_id}")
    v_start = float(v_obs_fit[0])
    peak_v = float(np.max(v_obs_fit))
    tail_n = max(5, min(25, len(v_obs_fit) // 8))
    tail_v = float(np.median(v_obs_fit[-tail_n:]))
    amp_rise = float(max(0.0, peak_v - v_start))
    amp_drop = float(max(0.0, peak_v - tail_v))
    amp_scale = float(max(abs(v_start), 1e-4))
    return {
        "shot_id": int(shot_id),
        "log_dose": float(np.log(SHOT_DOSES[shot_id])),
        "amp_rise": amp_rise,
        "amp_drop": amp_drop,
        "amp_rise_rel_start": float(amp_rise / amp_scale),
        "amp_drop_rel_start": float(amp_drop / amp_scale),
    }


def _median_abs_dev(values):
    values = np.asarray(values, dtype=float)
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    return med, max(mad, 1e-6)


def weighted_average_parameter(train_rows, param_name, weight_key="_bayes_weight"):
    y = np.asarray([row[param_name] for row in train_rows], dtype=float)
    w = np.asarray([max(float(row.get(weight_key, 1.0)), 1e-6) for row in train_rows], dtype=float)
    return float(np.average(y, weights=w))


def select_shape_neighbors(train_fit_rows, holdout_v_start_anchor: float):
    if len(train_fit_rows) <= 2:
        return list(train_fit_rows)
    _, scale_v_start = _median_abs_dev([row["v_start"] for row in train_fit_rows])
    ranked = sorted(train_fit_rows, key=lambda row: abs(row["v_start"] - holdout_v_start_anchor))
    selected = ranked[:2]
    second_gap = abs(selected[-1]["v_start"] - holdout_v_start_anchor)
    extra_gap_cutoff = max(1.5 * second_gap, second_gap + 0.5 * scale_v_start)
    for row in ranked[2:]:
        gap = abs(row["v_start"] - holdout_v_start_anchor)
        if gap <= extra_gap_cutoff:
            selected.append(row)
    return selected


def build_bayesian_family_rows(family_rows, selected_rows, holdout_v_start_anchor: float):
    local_rows = list(selected_rows) if selected_rows else list(family_rows)
    if not local_rows:
        return []

    pred_t_half = float(np.median([row["t_half"] for row in local_rows]))
    pred_n = float(np.median([row["n"] for row in local_rows]))
    pred_amp_rise_rel = float(np.median([row["amp_rise_rel_start"] for row in local_rows]))
    pred_amp_drop_rel = float(np.median([row["amp_drop_rel_start"] for row in local_rows]))
    pred_t_drop = float(np.median([row["t_drop"] for row in local_rows]))
    pred_k_drop = float(np.median([row["k_drop"] for row in local_rows]))
    _, scale_v_start = _median_abs_dev([row["v_start"] for row in local_rows])
    _, scale_t_half = _median_abs_dev([row["t_half"] for row in local_rows])
    _, scale_n = _median_abs_dev([row["n"] for row in local_rows])
    _, scale_amp_rise_rel = _median_abs_dev([row["amp_rise_rel_start"] for row in local_rows])
    _, scale_amp_drop_rel = _median_abs_dev([row["amp_drop_rel_start"] for row in local_rows])
    _, scale_t_drop = _median_abs_dev([row["t_drop"] for row in local_rows])
    _, scale_k_drop = _median_abs_dev([row["k_drop"] for row in local_rows])
    weighted_rows = []
    for row in local_rows:
        shape_score = (
            0.9 * abs(row["v_start"] - holdout_v_start_anchor) / scale_v_start
            + abs(row["t_half"] - pred_t_half) / scale_t_half
            + abs(row["n"] - pred_n) / scale_n
            + 0.8 * abs(row["amp_rise_rel_start"] - pred_amp_rise_rel) / scale_amp_rise_rel
            + 0.8 * abs(row["amp_drop_rel_start"] - pred_amp_drop_rel) / scale_amp_drop_rel
            + abs(row["t_drop"] - pred_t_drop) / scale_t_drop
            + abs(row["k_drop"] - pred_k_drop) / scale_k_drop
        )
        weight = float(np.exp(-0.5 * shape_score))
        weighted_row = dict(row)
        weighted_row["_bayes_weight"] = max(weight, 1e-6)
        weighted_rows.append(weighted_row)
    return weighted_rows


def scale_shape_to_zero_endpoint(y_hat: np.ndarray, v_start_anchor: float):
    y_hat = np.asarray(y_hat, dtype=float)
    if len(y_hat) == 0:
        return y_hat, 1.0
    y_end = float(y_hat[-1])
    if y_end <= 0.0 or v_start_anchor >= 0.0:
        return y_hat, 1.0
    denom = float(y_end - v_start_anchor)
    if denom <= 1e-9:
        return y_hat, 1.0
    scale = float(np.clip((-v_start_anchor) / denom, 0.0, 1.0))
    y_scaled = float(v_start_anchor) + scale * (y_hat - float(v_start_anchor))
    return np.asarray(y_scaled, dtype=float), scale


def main():
    holdout_shot_id, use_full_windows = parse_holdout()
    shot_ids = list(ACTIVE_SHOT_IDS)
    windows_ns = FULL_SHOT_WINDOWS_NS if use_full_windows else SHOT_WINDOWS_NS
    shot_data = {}

    for shot_id in shot_ids:
        # Load each shot on the shifted `t0` time base and keep only the chosen
        # recovery window used for fitting and holdout evaluation.
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

    # Fit the training shots directly, then choose the few most shape-compatible
    # neighbors to stabilize the parameter regressions used for the holdout shot.
    train_fit_rows = [direct_fit_windowed_shot_local(shot_id, shot_data) for shot_id in train_shot_ids]
    holdout_v_start_anchor = float(shot_data[holdout_shot_id]["v_obs_fit"][0])
    holdout_neighbor_family = NEIGHBOR_FAMILY_BY_SHOT[holdout_shot_id]
    family_train_rows = [
        row for row in train_fit_rows if row["neighbor_family"] == holdout_neighbor_family
    ]
    if not family_train_rows:
        family_train_rows = train_fit_rows
    selected_rows = select_shape_neighbors(family_train_rows, holdout_v_start_anchor)
    if (
        holdout_neighbor_family == "exp_group"
        and abs(holdout_v_start_anchor) < 0.8
        and len(family_train_rows) >= 3
    ):
        ranked_family_rows = sorted(
            family_train_rows,
            key=lambda row: abs(row["v_start"] - holdout_v_start_anchor),
        )
        selected_rows = ranked_family_rows[:3]
    bayes_rows = build_bayesian_family_rows(
        family_train_rows,
        selected_rows,
        holdout_v_start_anchor,
    )

    # Anchor the holdout at its observed starting value, use the closest
    # same-family starts to build a local Bayesian pool, then derive the shape
    # and amplitudes relative to that starting anchor without any dose regression.
    v_start_hat = holdout_v_start_anchor
    holdout_amp_scale = float(max(abs(holdout_v_start_anchor), 1e-4))
    t_half_hat = weighted_average_parameter(bayes_rows, "t_half")
    n_hat = weighted_average_parameter(bayes_rows, "n")
    t_drop_hat = weighted_average_parameter(bayes_rows, "t_drop")
    k_drop_hat = weighted_average_parameter(bayes_rows, "k_drop")
    amp_rise_rel_hat = weighted_average_parameter(bayes_rows, "amp_rise_rel_start")
    amp_drop_rel_hat = weighted_average_parameter(bayes_rows, "amp_drop_rel_start")
    amp_rise_hat = float(holdout_amp_scale * amp_rise_rel_hat)
    amp_drop_hat = float(holdout_amp_scale * amp_drop_rel_hat)

    # Keep regressed shape parameters in a physically meaningful range.
    amp_rise_hat = float(max(1e-4, amp_rise_hat))
    t_half_hat = float(max(1e-3, t_half_hat))
    n_hat = float(max(0.05, n_hat))
    amp_drop_hat = float(max(0.0, amp_drop_hat))
    t_drop_hat = float(max(5.0, t_drop_hat))
    k_drop_hat = float(max(0.5, k_drop_hat))

    log10_t_ns = shot_data[holdout_shot_id]["log10_t_ns"]
    v_obs_raw = shot_data[holdout_shot_id]["v_obs_raw"]
    v_obs_fit = shot_data[holdout_shot_id]["v_obs_fit"]
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
    y_hat, endpoint_scale = scale_shape_to_zero_endpoint(y_hat, v_start_hat)
    if endpoint_scale < 1.0:
        amp_rise_hat *= endpoint_scale
        amp_drop_hat *= endpoint_scale

    rmse = float(np.sqrt(np.mean((y_hat - v_obs_fit) ** 2)))
    ss_tot = float(np.sum((v_obs_fit - np.mean(v_obs_fit)) ** 2))
    r_squared = float(1.0 - np.sum((v_obs_fit - y_hat) ** 2) / ss_tot) if ss_tot > 0 else np.nan
    sigma_hat = float(np.std(v_obs_fit - y_hat, ddof=1)) if len(v_obs_fit) > 1 else float("nan")

    t_plot_ns = shot_data[holdout_shot_id]["t_window_rel_ns"]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_png_linear = OUTPUT_DIR / f"hill_delay_drop_{holdout_shot_id}_fit_linear.png"

    plt.figure(figsize=(8, 5))
    plt.plot(t_plot_ns, v_obs_raw, color="0.75", linewidth=1.0, label="Observed raw")
    plt.plot(t_plot_ns, v_obs_fit, color="black", linewidth=1.2, label="Observed smoothed")
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
    print(f"Saved: {out_png_linear}")


if __name__ == "__main__":
    main()
