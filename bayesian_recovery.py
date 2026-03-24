from pathlib import Path

import pytensor
pytensor.config.cxx = ""
pytensor.config.mode = "FAST_COMPILE"
pytensor.config.linker = "py"

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

from rsquaredfit import compute_new_t0_abs, prepare_window_data


SHOT_ID = 27277
DATA_DIR = Path("data")
OUT_CSV = Path("bayesian_recovery_27277_summary.csv")
OUT_PNG = Path("bayesian_recovery_27277_fit.png")
START_TIME_NS = 200.0
END_TIME_NS = 400.0
SUBSAMPLE = 4

SHOT_DOSES = {
    27277: 1.06e10,
    27278: 1.16e10,
    27279: 6.22e10,
    27286: 4.74e11,
}


def hill_recovery(t_ns, v0, t_half_ns, n):
    return v0 / (1.0 + t_ns / t_half_ns) ** n


def load_windowed_shot(shot_id: int):
    t_rel_ns, v_diode = compute_new_t0_abs(DATA_DIR / f"{shot_id}_data.csv", shot_id)
    t_ns, v_obs = prepare_window_data(
        t_rel_ns,
        v_diode,
        START_TIME_NS,
        END_TIME_NS,
    )[:2]
    return t_ns[::SUBSAMPLE], v_obs[::SUBSAMPLE]


def main():
    dose = SHOT_DOSES[SHOT_ID]
    log_dose = float(np.log(dose))
    t_ns, v_obs = load_windowed_shot(SHOT_ID)

    print("Dose values:")
    for shot, shot_dose in SHOT_DOSES.items():
        print(f"{shot}: {shot_dose:.6e}")

    with pm.Model() as model:
        a1 = pm.Normal("a1", 0.0, 1.0)
        a2 = pm.Normal("a2", 100.0, 200.0)
        a3 = pm.Normal("a3", 1.0, 2.0)
        b1 = pm.Normal("b1", 0.0, 0.1)
        b2 = pm.Normal("b2", 0.0, 20.0)
        b3 = pm.Normal("b3", 0.0, 0.2)
        sigma = pm.HalfNormal("sigma", 0.2)

        v0_i = pm.Deterministic("V0_i", a1 + b1 * log_dose)
        t_half_i = pm.Deterministic("t_half_i", a2 + b2 * log_dose)
        n_i = pm.Deterministic("n_i", a3 + b3 * log_dose)

        pm.Potential("v0_positive", pm.math.switch(pm.math.gt(v0_i, 0.0), 0.0, -np.inf))
        pm.Potential("t_half_positive", pm.math.switch(pm.math.gt(t_half_i, 0.0), 0.0, -np.inf))
        pm.Potential("n_positive", pm.math.switch(pm.math.gt(n_i, 0.0), 0.0, -np.inf))

        mu = hill_recovery(t_ns, v0_i, t_half_i, n_i)
        pm.Normal("obs", mu=mu, sigma=sigma, observed=v_obs)

        trace = pm.sample(
            draws=250,
            tune=250,
            chains=1,
            cores=1,
            random_seed=42,
            target_accept=0.9,
            progressbar=True,
            return_inferencedata=True,
        )

    summary = az.summary(trace, var_names=["a1", "a2", "a3", "b1", "b2", "b3", "sigma", "V0_i", "t_half_i", "n_i"])
    summary.to_csv(OUT_CSV)
    print(f"\nSaved: {OUT_CSV}")
    print(summary.to_string())

    post = trace.posterior
    v0_hat = float(post["V0_i"].mean())
    t_half_hat = float(post["t_half_i"].mean())
    n_hat = float(post["n_i"].mean())
    v_hat = hill_recovery(t_ns, v0_hat, t_half_hat, n_hat)

    plt.figure(figsize=(8, 5))
    plt.plot(t_ns, v_obs, color="black", linewidth=1.2, label="Observed")
    plt.plot(t_ns, v_hat, color="red", linewidth=2.0, label="Posterior mean fit")
    plt.xlabel("Time since window start (ns)")
    plt.ylabel("Voltage (V)")
    plt.title(f"Bayesian Recovery Fit: Shot {SHOT_ID}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    plt.close()
    print(f"Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
