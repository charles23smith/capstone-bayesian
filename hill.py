import argparse
import os
import random
from pathlib import Path

import pytensor

pytensor.config.cxx = ""
pytensor.config.mode = "FAST_COMPILE"
pytensor.config.linker = "py"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from rsquaredfit import compute_new_t0_abs


DATA_DIR = Path("data")
SHOT_DATA_CSV = Path("shot_data.csv")
OUT_SUMMARY_CSV = Path("hill_summary.csv")
OUT_LOO_CSV = Path("hill_loo_results.csv")
RANDOM_SEED = 42
DRAWS = 200
TUNE = 250
TARGET_ACCEPT = 0.98
MAX_POINTS = 1000
NUMPYRO_AVAILABLE = True

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

try:
    import numpyro  # noqa: F401
except ImportError:
    NUMPYRO_AVAILABLE = False


ACTIVE_SHOT_IDS = [27272, 27276, 27277, 27278, 27279, 27286]
DEFAULT_HOLDOUT_SHOT_ID = 27272
WINDOWS_NS = {
    27272: (200.0, 400.0),
    27276: (205.0, 500.0),
    27277: (200.0, 400.0),
    27278: (200.0, 400.0),
    27279: (210.0, 400.0),
    27286: (152.0, 300.0),
}
FULL_WINDOWS_NS = {
    shot_id: (75.0, end_ns)
    for shot_id, (_, end_ns) in WINDOWS_NS.items()
}


def hill_decay(t_ns, v0, t_half_ns, n):
    t_ns = pt.maximum(t_ns, 0.0)
    return v0 / pt.power(1.0 + t_ns / t_half_ns, n)


def hill_decay_np(t_ns, v0, t_half_ns, n):
    t_ns = np.asarray(t_ns, dtype=float)
    t_pos = np.maximum(t_ns, 0.0)
    return float(v0) / np.power(1.0 + t_pos / float(t_half_ns), float(n))


def hill_decay_logtime(log10_t_ns, v0, t_half_ns, n):
    return hill_decay(pt.power(10.0, log10_t_ns), v0, t_half_ns, n)


def hill_decay_logtime_np(log10_t_ns, v0, t_half_ns, n):
    return hill_decay_np(np.power(10.0, np.asarray(log10_t_ns, dtype=float)), v0, t_half_ns, n)


def fit_metrics(y_true, y_hat):
    y_true = np.asarray(y_true, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)
    rmse = float(np.sqrt(np.mean((y_true - y_hat) ** 2)))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r_squared = float(1.0 - np.sum((y_true - y_hat) ** 2) / ss_tot) if ss_tot > 0 else float("nan")
    return rmse, r_squared


def choose_log_spaced_indices(t_ns: np.ndarray, n_keep: int) -> np.ndarray:
    if len(t_ns) <= n_keep:
        return np.arange(len(t_ns), dtype=int)
    shifted = t_ns - float(t_ns[0]) + 1.0
    grid = np.geomspace(1.0, float(shifted[-1]), n_keep)
    idx = np.searchsorted(shifted, grid, side="left")
    idx = np.clip(idx, 0, len(t_ns) - 1)
    idx = np.unique(np.concatenate(([0], idx, [len(t_ns) - 1])))
    return idx.astype(int)


def load_metadata():
    df = pd.read_csv(SHOT_DATA_CSV)
    df["shot_id"] = df["shot_id"].astype(int)
    df["dose_rate"] = df["dose_rate"].astype(float)
    return df.set_index("shot_id")


def load_windowed_trace(shot_id: int, windows_ns):
    start_ns, end_ns = windows_ns[shot_id]
    t_rel_ns, v_diode = compute_new_t0_abs(DATA_DIR / f"{shot_id}_data.csv", shot_id)
    mask = (t_rel_ns >= start_ns) & (t_rel_ns <= end_ns)
    t_actual_ns = np.asarray(t_rel_ns[mask], dtype=float)
    v_actual = np.asarray(v_diode[mask], dtype=float)
    if len(t_actual_ns) < 20:
        raise ValueError(f"too few points in window for shot {shot_id}")

    idx = choose_log_spaced_indices(t_actual_ns, MAX_POINTS)
    t_actual_ns = t_actual_ns[idx]
    v_actual = v_actual[idx]
    t_fit_ns = t_actual_ns - float(t_actual_ns[0])

    tail_n = max(5, min(50, len(v_actual) // 8))
    baseline = float(np.median(v_actual[-tail_n:]))
    sign = -1.0 if float(v_actual[0] - baseline) < 0 else 1.0
    amp_obs = np.maximum(sign * (v_actual - baseline), 0.0)

    keep = np.isfinite(t_fit_ns) & np.isfinite(amp_obs) & (t_fit_ns > 0.0)
    if np.count_nonzero(keep) < 10:
        raise ValueError(f"window has too few positive-time fit points for shot {shot_id}")

    return {
        "shot_id": shot_id,
        "t_actual_ns": t_actual_ns[keep],
        "t_fit_ns": t_fit_ns[keep],
        "log10_t_ns": np.log10(t_fit_ns[keep]),
        "v_actual": v_actual[keep],
        "amp_obs": amp_obs[keep],
        "baseline": baseline,
        "sign": sign,
        "window_start_ns": start_ns,
        "window_end_ns": end_ns,
    }


def fit_bayesian_hill(train_shot_ids, shot_data):
    with pm.Model() as model:
        # TA-style priors on theta = {a1, a2, a3, b1, b2, b3}
        a1 = pm.Normal("a1", 1.0, 2.0)
        a2 = pm.Normal("a2", 20.0, 25.0)
        a3 = pm.Normal("a3", 1.5, 1.0)

        b1 = pm.Normal("b1", 0.0, 0.15)
        b2 = pm.Normal("b2", 0.0, 2.0)
        b3 = pm.Normal("b3", 0.0, 0.10)
        sigma = pm.HalfNormal("sigma", 0.08)

        shot_param_names = []
        for shot_id in train_shot_ids:
            log_dose = shot_data[shot_id]["log_dose"]
            log10_t_ns = shot_data[shot_id]["log10_t_ns"]
            amp_obs = shot_data[shot_id]["amp_obs"]

            # Direct dose-laws from TA notes:
            # V0_i = a1 + b1*log(dose_i)
            # t_half_i = a2 + b2*log(dose_i)
            # n_i = a3 + b3*log(dose_i)
            v0_i = pm.Deterministic(f"v0_{shot_id}", a1 + b1 * log_dose)
            t_half_i = pm.Deterministic(f"t_half_{shot_id}", a2 + b2 * log_dose)
            n_i = pm.Deterministic(f"n_{shot_id}", a3 + b3 * log_dose)

            # Keep the likelihood on the physical region only.
            valid = pt.and_(pt.gt(v0_i, 0.0), pt.and_(pt.gt(t_half_i, 0.0), pt.gt(n_i, 0.0)))
            pm.Potential(f"valid_{shot_id}", pt.switch(valid, 0.0, -np.inf))

            mu_amp = pm.Deterministic(f"mu_amp_{shot_id}", hill_decay_logtime(log10_t_ns, v0_i, t_half_i, n_i))
            pm.Normal(f"obs_{shot_id}", mu=mu_amp, sigma=sigma, observed=amp_obs)
            shot_param_names.extend([f"v0_{shot_id}", f"t_half_{shot_id}", f"n_{shot_id}"])

        sample_kwargs = dict(
            draws=DRAWS,
            tune=TUNE,
            chains=1,
            cores=1,
            random_seed=RANDOM_SEED,
            target_accept=TARGET_ACCEPT,
            progressbar=True,
            return_inferencedata=True,
        )
        if NUMPYRO_AVAILABLE:
            sample_kwargs["nuts_sampler"] = "numpyro"
        trace = pm.sample(**sample_kwargs)

    summary = pm.summary(
        trace,
        var_names=["a1", "a2", "a3", "b1", "b2", "b3", "sigma", *shot_param_names],
    )
    return trace, summary


def main():
    parser = argparse.ArgumentParser(description="TA-style Bayesian Hill leave-one-out model on the delay-shot diode set.")
    parser.add_argument("holdout_shot_id", nargs="?", type=int, default=None)
    parser.add_argument("--full", action="store_true", help="Use the broader 75 ns to current endpoint windows.")
    args, unknown = parser.parse_known_args()
    windows_ns = FULL_WINDOWS_NS if args.full else WINDOWS_NS

    holdout_shot_id = args.holdout_shot_id
    if holdout_shot_id is None:
        for token in unknown:
            if token.startswith("--") and token[2:].isdigit():
                holdout_shot_id = int(token[2:])
                break
    if holdout_shot_id is None:
        holdout_shot_id = random.choice(ACTIVE_SHOT_IDS)
        print(f"Random holdout selected: {holdout_shot_id}")
    holdout_shot_id = int(holdout_shot_id)
    if holdout_shot_id not in ACTIVE_SHOT_IDS:
        raise ValueError(f"holdout shot must be one of {ACTIVE_SHOT_IDS}")

    metadata = load_metadata()
    shot_data = {}
    for shot_id in ACTIVE_SHOT_IDS:
        trace = load_windowed_trace(shot_id, windows_ns)
        trace["dose_rate"] = float(metadata.loc[shot_id, "dose_rate"])
        trace["log_dose"] = float(np.log(trace["dose_rate"]))
        shot_data[shot_id] = trace

    print("Dose values:")
    for shot_id in ACTIVE_SHOT_IDS:
        print(f"{shot_id}: {shot_data[shot_id]['dose_rate']:.6e}")
    print(f"Window mode: {'full' if args.full else 'default'}")

    train_shot_ids = [shot_id for shot_id in ACTIVE_SHOT_IDS if shot_id != holdout_shot_id]
    print(f"\nLOO holdout: {holdout_shot_id} | train on {train_shot_ids}")
    print(f"Sampler: {'NumPyro NUTS' if NUMPYRO_AVAILABLE else 'PyMC NUTS'}")

    trace, summary = fit_bayesian_hill(train_shot_ids, shot_data)
    summary.to_csv(OUT_SUMMARY_CSV)

    post = trace.posterior
    log_dose = shot_data[holdout_shot_id]["log_dose"]
    a1_samples = np.asarray(post["a1"]).ravel()
    a2_samples = np.asarray(post["a2"]).ravel()
    a3_samples = np.asarray(post["a3"]).ravel()
    b1_samples = np.asarray(post["b1"]).ravel()
    b2_samples = np.asarray(post["b2"]).ravel()
    b3_samples = np.asarray(post["b3"]).ravel()
    sigma_hat = float(post["sigma"].mean())

    v0_samples = a1_samples + b1_samples * log_dose
    t_half_samples = a2_samples + b2_samples * log_dose
    n_samples = a3_samples + b3_samples * log_dose
    valid = (v0_samples > 0.0) & (t_half_samples > 0.0) & (n_samples > 0.0)
    if np.any(valid):
        v0_hat = float(np.mean(v0_samples[valid]))
        t_half_hat = float(np.mean(t_half_samples[valid]))
        n_hat = float(np.mean(n_samples[valid]))
    else:
        v0_hat = float(max(np.mean(v0_samples), 1e-6))
        t_half_hat = float(max(np.mean(t_half_samples), 1e-3))
        n_hat = float(max(np.mean(n_samples), 0.05))

    amp_hat = hill_decay_logtime_np(shot_data[holdout_shot_id]["log10_t_ns"], v0_hat, t_half_hat, n_hat)
    v_hat = shot_data[holdout_shot_id]["baseline"] + shot_data[holdout_shot_id]["sign"] * amp_hat
    rmse, r_squared = fit_metrics(shot_data[holdout_shot_id]["v_actual"], v_hat)

    loo_df = pd.DataFrame(
        [
            {
                "holdout_shot": holdout_shot_id,
                "train_shots": " ".join(str(x) for x in train_shot_ids),
                "dose_rate": shot_data[holdout_shot_id]["dose_rate"],
                "v0_pred": v0_hat,
                "t_half_pred": t_half_hat,
                "n_pred": n_hat,
                "sigma_fit": sigma_hat,
                "rmse": rmse,
                "r_squared": r_squared,
                "n_points": int(len(shot_data[holdout_shot_id]["t_fit_ns"])),
            }
        ]
    )
    loo_df.to_csv(OUT_LOO_CSV, index=False)

    out_png = Path(f"hill_{holdout_shot_id}_fit.png")
    out_png_linear = Path(f"hill_{holdout_shot_id}_fit_linear.png")

    plt.figure(figsize=(8, 5))
    plt.plot(shot_data[holdout_shot_id]["t_actual_ns"], shot_data[holdout_shot_id]["v_actual"], color="black", linewidth=1.2, label="Observed")
    plt.plot(shot_data[holdout_shot_id]["t_actual_ns"], v_hat, color="red", linewidth=2.0, label="LOO posterior mean fit")
    plt.xscale("log")
    plt.xlabel("Time after NEW t0 (ns)")
    plt.ylabel("Diode voltage (V)")
    plt.title(f"Hill Bayesian LOO Fit: Shot {holdout_shot_id}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(shot_data[holdout_shot_id]["t_actual_ns"], shot_data[holdout_shot_id]["v_actual"], color="black", linewidth=1.2, label="Observed")
    plt.plot(shot_data[holdout_shot_id]["t_actual_ns"], v_hat, color="red", linewidth=2.0, label="LOO posterior mean fit")
    plt.xlabel("Time after NEW t0 (ns)")
    plt.ylabel("Diode voltage (V)")
    plt.title(f"Hill Bayesian LOO Fit: Shot {holdout_shot_id} (Linear Time)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png_linear, dpi=150)
    plt.close()

    print(
        f"Predicted holdout {holdout_shot_id}: "
        f"v0={v0_hat:.4f} t_half={t_half_hat:.4f} n={n_hat:.4f} "
        f"RMSE={rmse:.6f} R^2={r_squared:.6f}"
    )
    print(f"Saved: {OUT_SUMMARY_CSV}")
    print(f"Saved: {OUT_LOO_CSV}")
    print(loo_df.to_string(index=False))
    print(f"Saved: {out_png}")
    print(f"Saved: {out_png_linear}")


if __name__ == "__main__":
    main()
