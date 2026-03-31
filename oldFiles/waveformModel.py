"""
27271_autostop_numpyro.py
Bayesian diode waveform fit around FIRST negative peak with AUTO-STOP on slope reversal.

Key idea:
- Start at FIRST negative peak (diode minimum).
- Collect data forward until the recovery stops being monotone (slope flips negative) → likely next shot/next lobe.
- Fit ONLY that clean recovery region.

Model in y-domain (negated voltage y = -V):
    y(t) = b + A1 + A2                                  for t < d
         = b + A1*exp(-(t-d)/tau1) + A2*exp(-(t-d)/tau2) for t >= d

Then original voltage is V(t) = -y(t).

Uses PyMC + NumPyro (JAX) for fast sampling on Windows.
Prevents PyTensor from compiling lazylinker (fixes your MinGW 64-bit error).

Outputs in ./diode_fit_outputs:
  00_peak_window_full.png        (sanity: peak + max window)
  01_window_autostop.png         (sanity: chosen segment + cutoff)
  01b_init_fit_curvefit.png      (init fit in y-domain)
  02_fit_bayes_delayed_doubleexp.png (fit w/ 50/95% credible bands)
  03_residuals.png               (V-domain residuals)
"""

# -----------------------------
# MUST BE FIRST (before importing pymc/pytensor)
# Prevent PyTensor from trying to compile lazylinker with your broken MinGW.
# This does NOT slow sampling much because NumPyro/JAX does the heavy work.
# -----------------------------
import os
os.environ["PYTENSOR_FLAGS"] = "cxx=,linker=py,mode=FAST_COMPILE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter, find_peaks
from scipy.optimize import curve_fit

import pymc as pm
import arviz as az


# -----------------------------
# USER SETTINGS
# -----------------------------
CSV_FILE = "27271_data.csv"          # <-- your csv (time, volts)
OUT_DIR  = "diode_fit_outputs"

NEGATE = True                         # use y = -V for fitting
SEED = 42

# Peak finding (first significant peak in y-domain)
SMOOTH_NS_PEAK = 2.0                  # smoothing for peak pick
PEAK_PROM_FRAC = 0.08                 # prominence threshold fraction (try 0.03–0.15 if needed)

# Auto-stop settings (stop when recovery slope flips negative)
MAX_WINDOW_NS = 300.0                 # safety cap; doesn't need to be 170
MIN_STOP_NS   = 25.0                  # ignore slope flips before this time (noise)
SMOOTH_NS_SLOPE = 3.0                 # smoothing used for slope detection
K_SUSTAIN     = 10                    # require K consecutive negative slopes to confirm turnaround

# Fit subsampling
SUBSAMPLE_N = 400                      # number of points used in Bayesian fit

# Bayesian sampling (NumPyro)
CHAINS = 2
CORES  = 1                             # keep 1 on Windows
DRAWS  = 1200
TUNE   = 1200
TARGET_ACCEPT = 0.92


# -----------------------------
# Numpy helper: delayed double-exp
# -----------------------------
def delayed_doubleexp_np(t, b, A1, tau1, A2, tau2, d):
    t = np.asarray(t)
    y = np.empty_like(t, dtype=float)
    pre = t < d
    y[pre] = b + A1 + A2
    td = t[~pre] - d
    y[~pre] = b + A1*np.exp(-td/tau1) + A2*np.exp(-td/tau2)
    return y


def _oddify(n: int) -> int:
    return n if (n % 2 == 1) else (n + 1)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # -----------------------------
    # Load data
    # -----------------------------
    data = pd.read_csv(CSV_FILE)
    time = data.iloc[:, 0].to_numpy(dtype=float)
    vraw = data.iloc[:, 1].to_numpy(dtype=float)

    # y-domain signal for peak selection / fitting
    ysig = -vraw if NEGATE else vraw

    print(f"Loaded {len(time)} points from {CSV_FILE}")
    print(f"Voltage range (after NEGATE={NEGATE}): {ysig.min():.4f} to {ysig.max():.4f} V")

    # -----------------------------
    # STEP 1: Find FIRST significant peak in y-domain
    # This corresponds to FIRST negative peak in V-domain if NEGATE=True.
    # -----------------------------
    dt = float(np.median(np.diff(time)))
    win_peak = int(max(9, round((SMOOTH_NS_PEAK * 1e-9) / dt)))
    win_peak = _oddify(win_peak)
    win_peak = min(win_peak, max(9, (len(ysig)//2)*2 - 1))
    if win_peak % 2 == 0:
        win_peak -= 1

    y_smooth = savgol_filter(ysig, window_length=win_peak, polyorder=3)

    yrange = float(np.max(y_smooth) - np.min(y_smooth))
    prom = max(1e-6, PEAK_PROM_FRAC * yrange)

    peaks, props = find_peaks(y_smooth, prominence=prom)
    if len(peaks) == 0:
        raise RuntimeError(
            "No peaks found. Try lowering PEAK_PROM_FRAC (e.g., 0.03) or check NEGATE."
        )

    peak_idx = int(peaks[0])
    peak_time = float(time[peak_idx])

    print("\nChosen FIRST significant peak:")
    print(f"  idx = {peak_idx}")
    print(f"  t0  = {peak_time*1e6:.5f} us")
    print(f"  V(t0) original = {vraw[peak_idx]:.4f} V")
    print(f"  y(t0) (negated) = {ysig[peak_idx]:.4f} V")

    # -----------------------------
    # STEP 2: Candidate window forward from peak (up to MAX_WINDOW_NS)
    # -----------------------------
    t_end = peak_time + MAX_WINDOW_NS * 1e-9
    mask = (time >= peak_time) & (time <= t_end)

    tw_full = time[mask] - peak_time         # seconds from peak
    y_full  = ysig[mask]                     # y-domain data (for fitting)
    V_full  = -y_full if NEGATE else y_full  # V-domain data (for slope logic/plots)

    if len(tw_full) < 80:
        raise RuntimeError("Not enough points after the peak. Increase MAX_WINDOW_NS or check data.")

    # -----------------------------
    # STEP 3: AUTO-STOP when slope turns negative (recovery ends / next shot begins)
    # Recovery in V-domain should have positive slope (V increasing toward 0).
    # -----------------------------
    dtw = float(np.median(np.diff(tw_full)))
    win_slope = int(max(9, round((SMOOTH_NS_SLOPE * 1e-9) / dtw)))
    win_slope = _oddify(win_slope)
    win_slope = min(win_slope, max(9, (len(V_full)//2)*2 - 1))
    if win_slope % 2 == 0:
        win_slope -= 1

    V_s = savgol_filter(V_full, window_length=win_slope, polyorder=3)
    dVdt = np.gradient(V_s, tw_full)

    min_i = int(np.searchsorted(tw_full, MIN_STOP_NS * 1e-9))
    neg = dVdt < 0

    stop_idx = len(tw_full) - 1
    for i in range(min_i, len(tw_full) - K_SUSTAIN):
        if np.all(neg[i:i+K_SUSTAIN]):
            stop_idx = i
            break

    t_cut = float(tw_full[stop_idx])
    print(f"\nAuto-stop cutoff at t = {t_cut*1e9:.2f} ns (slope turns negative → next shot/lobe)")

    tw_seg = tw_full[:stop_idx+1]
    y_seg  = y_full[:stop_idx+1]
    V_seg  = V_full[:stop_idx+1]

    # If somehow too short, relax sustain requirement
    if len(tw_seg) < 80:
        print("WARNING: segment too short; relaxing K_SUSTAIN from 10 to 5.")
        K2 = 5
        stop_idx = len(tw_full) - 1
        for i in range(min_i, len(tw_full) - K2):
            if np.all(neg[i:i+K2]):
                stop_idx = i
                break
        t_cut = float(tw_full[stop_idx])
        tw_seg = tw_full[:stop_idx+1]
        y_seg  = y_full[:stop_idx+1]
        V_seg  = V_full[:stop_idx+1]
        print(f"New cutoff at t = {t_cut*1e9:.2f} ns, seg pts = {len(tw_seg)}")

    print(f"Final chosen segment: {len(tw_seg)} points, duration {tw_seg[-1]*1e9:.2f} ns")

    # -----------------------------
    # Plot sanity: full trace w/ peak + max window cap
    # -----------------------------
    plt.figure(figsize=(12, 5))
    plt.plot(time*1e6, vraw, "k-", lw=1, alpha=0.75, label="Original V (diode)")
    plt.plot(peak_time*1e6, vraw[peak_idx], "r*", ms=14, label="Chosen first negative peak")
    plt.axvspan(peak_time*1e6, (peak_time + MAX_WINDOW_NS*1e-9)*1e6,
                color="orange", alpha=0.15, label=f"Max window cap ({MAX_WINDOW_NS:.0f} ns)")
    plt.xlabel("Time (us)")
    plt.ylabel("Volts")
    plt.title("Chosen Peak + Max Window Cap (sanity)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    p0 = os.path.join(OUT_DIR, "00_peak_window_full.png")
    plt.savefig(p0, dpi=170)
    plt.close()
    print(f"Saved: {p0}")

    # Plot sanity: chosen auto-stop segment + cutoff
    plt.figure(figsize=(11, 5))
    plt.plot(tw_full*1e9, V_full, "0.75", lw=1, label="V within max cap")
    plt.plot(tw_seg*1e9, V_seg, "b.-", ms=3, label="V used for fit (auto-stop)")
    plt.axvline(t_cut*1e9, color="orange", ls="--", lw=2, label=f"cutoff {t_cut*1e9:.2f} ns")
    plt.xlabel("Time from first negative peak (ns)")
    plt.ylabel("Volts (original sign)")
    plt.title("Auto-stop Segment (stops when slope turns negative)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    p1 = os.path.join(OUT_DIR, "01_window_autostop.png")
    plt.savefig(p1, dpi=170)
    plt.close()
    print(f"Saved: {p1}")

    # -----------------------------
    # Subsample for fitting
    # -----------------------------
    N = min(SUBSAMPLE_N, len(tw_seg))
    idx = np.linspace(0, len(tw_seg) - 1, N, dtype=int)
    tw = tw_seg[idx]
    y  = y_seg[idx]

    print(f"\nSubsampled for fit: {len(tw)} points, duration {tw[-1]*1e9:.2f} ns")
    print(f"y(t=0) = {y[0]:.4f} V, y(t=end) = {y[-1]:.4f} V")

    # -----------------------------
    # curve_fit init (good starting values)
    # -----------------------------
    tail_n = max(10, len(y)//10)
    b0 = float(np.mean(y[-tail_n:]))
    y0 = float(y[0])
    A_tot = max(1e-6, y0 - b0)

    # delay guess = when y drops 5% toward baseline
    thresh = y0 - 0.05 * A_tot
    d0 = float(tw[np.argmax(y < thresh)]) if np.any(y < thresh) else 40e-9

    A1_0, A2_0 = 0.55*A_tot, 0.45*A_tot
    tau1_0, tau2_0 = 10e-9, 25e-9
    p_init = [b0, A1_0, tau1_0, A2_0, tau2_0, d0]

    lb = [b0 - 3*A_tot, 0.0, 1e-9, 0.0, 2e-9,  0.0]
    ub = [b0 + 3*A_tot, 5*A_tot, 250e-9, 5*A_tot, 600e-9, 150e-9]

    print("\nRunning curve_fit init...")
    try:
        popt, _ = curve_fit(delayed_doubleexp_np, tw, y, p0=p_init, bounds=(lb, ub), maxfev=30000)
    except Exception as e:
        print("curve_fit failed, using heuristic init:", e)
        popt = np.array(p_init, dtype=float)

    b_fit, A1_fit, tau1_fit, A2_fit, tau2_fit, d_fit = map(float, popt)
    print("Init (curve_fit) parameters:")
    print(f"  b    = {b_fit:.6f}")
    print(f"  A1   = {A1_fit:.6f}, tau1 = {tau1_fit*1e9:.3f} ns")
    print(f"  A2   = {A2_fit:.6f}, tau2 = {tau2_fit*1e9:.3f} ns")
    print(f"  d    = {d_fit*1e9:.3f} ns")

    # Init fit plot (y-domain)
    tplt = np.linspace(0, tw[-1], 1500)
    plt.figure(figsize=(12, 5))
    plt.plot(tw*1e9, y, "b.", ms=4, label="y data (negated domain)")
    plt.plot(tplt*1e9, delayed_doubleexp_np(tplt, *popt), "r-", lw=2.5, label="curve_fit init")
    plt.xlabel("Time from peak (ns)")
    plt.ylabel("y = -V (if NEGATE=True)")
    plt.title("Initialization Fit (curve_fit) in y-domain")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    pinit = os.path.join(OUT_DIR, "01b_init_fit_curvefit.png")
    plt.savefig(pinit, dpi=170)
    plt.close()
    print(f"Saved: {pinit}")

    # -----------------------------
    # Bayesian model with NumPyro NUTS
    # -----------------------------
    res0 = y - delayed_doubleexp_np(tw, *popt)
    sig0 = float(np.std(res0) + 1e-6)

    with pm.Model() as model:
        b = pm.Normal("baseline", mu=b_fit, sigma=max(0.05, 0.15*A_tot))

        A1 = pm.HalfNormal("A1", sigma=max(abs(A1_fit), 0.2))
        A2 = pm.HalfNormal("A2", sigma=max(abs(A2_fit), 0.2))

        tau1 = pm.LogNormal("tau1", mu=np.log(max(tau1_fit, 2e-9)), sigma=0.6)
        tau2_delta = pm.LogNormal("tau2_delta", mu=np.log(max(tau2_fit - tau1_fit, 2e-9)), sigma=0.7)
        tau2 = pm.Deterministic("tau2", tau1 + tau2_delta)

        d = pm.TruncatedNormal("d", mu=d_fit, sigma=8e-9, lower=0.0, upper=200e-9)

        sigma = pm.HalfNormal("sigma", sigma=max(sig0*2.0, 1e-4))
        nu = pm.Exponential("nu", lam=1/10) + 2

        t = tw
        mu_pre = b + A1 + A2
        mu_post = b + A1*pm.math.exp(-(t - d)/tau1) + A2*pm.math.exp(-(t - d)/tau2)
        mu = pm.math.switch(t < d, mu_pre, mu_post)

        pm.StudentT("obs", nu=nu, mu=mu, sigma=sigma, observed=y)

        print("\nSampling with NumPyro NUTS (fast)...")
        trace = pm.sample(
            draws=DRAWS,
            tune=TUNE,
            chains=CHAINS,
            cores=CORES,
            target_accept=TARGET_ACCEPT,
            random_seed=SEED,
            nuts_sampler="numpyro",
            progressbar=True,
        )

    # -----------------------------
    # Posterior curves + credible bands
    # -----------------------------
    post = trace.posterior

    def flat(name):
        return post[name].values.reshape(-1)

    b_s = flat("baseline")
    A1_s = flat("A1")
    A2_s = flat("A2")
    tau1_s = flat("tau1")
    tau2_s = flat("tau2")
    d_s = flat("d")

    S = len(b_s)
    tgrid = np.linspace(0, tw[-1], 1800)

    curves = np.zeros((S, len(tgrid)))
    for i in range(S):
        curves[i] = delayed_doubleexp_np(tgrid, b_s[i], A1_s[i], tau1_s[i], A2_s[i], tau2_s[i], d_s[i])

    mean_c = curves.mean(axis=0)
    lo95, hi95 = np.percentile(curves, [2.5, 97.5], axis=0)
    lo50, hi50 = np.percentile(curves, [25, 75], axis=0)

    # Metrics in y-domain
    yfit = np.interp(tw, tgrid, mean_c)
    resid_y = y - yfit
    rmse_y = float(np.sqrt(np.mean(resid_y**2)))
    r2 = float(1 - np.sum(resid_y**2) / np.sum((y - y.mean())**2))

    # Print posterior summary
    summ = az.summary(trace, var_names=["baseline","A1","tau1","A2","tau2","d","sigma","nu"])
    print("\n" + "="*74)
    print("POSTERIOR SUMMARY (NumPyro)")
    print("="*74)
    print(summ)

    def mean_hdi(x):
        m = float(np.mean(x))
        lo, hi = np.percentile(x, [2.5, 97.5])
        return m, float(lo), float(hi)

    b_m, b_lo, b_hi = mean_hdi(b_s)
    A1_m, A1_lo, A1_hi = mean_hdi(A1_s)
    A2_m, A2_lo, A2_hi = mean_hdi(A2_s)
    t1_m, t1_lo, t1_hi = mean_hdi(tau1_s)
    t2_m, t2_lo, t2_hi = mean_hdi(tau2_s)
    d_m, d_lo, d_hi = mean_hdi(d_s)

    print("\n" + "="*74)
    print("FIT RESULTS (y = -V if NEGATE=True)")
    print("="*74)
    print(f"R²   = {r2:.6f}")
    print(f"RMSE = {rmse_y:.6f} V (y-domain)\n")
    print(f"b      = {b_m:.6f}   [{b_lo:.6f}, {b_hi:.6f}]")
    print(f"A1     = {A1_m:.6f}   [{A1_lo:.6f}, {A1_hi:.6f}]")
    print(f"tau1   = {t1_m*1e9:.4f} ns   [{t1_lo*1e9:.4f}, {t1_hi*1e9:.4f}] ns")
    print(f"A2     = {A2_m:.6f}   [{A2_lo:.6f}, {A2_hi:.6f}]")
    print(f"tau2   = {t2_m*1e9:.4f} ns   [{t2_lo*1e9:.4f}, {t2_hi*1e9:.4f}] ns")
    print(f"d      = {d_m*1e9:.4f} ns   [{d_lo*1e9:.4f}, {d_hi*1e9:.4f}] ns")
    print("="*74)

    # Convert to original V-domain for plotting:
    V_data = (-y) if NEGATE else y
    V_mean = (-mean_c) if NEGATE else mean_c

    # When negating CI bands, lower/upper swap
    V_lo95 = (-hi95) if NEGATE else lo95
    V_hi95 = (-lo95) if NEGATE else hi95
    V_lo50 = (-hi50) if NEGATE else lo50
    V_hi50 = (-lo50) if NEGATE else hi50

    # Main fit plot (V-domain)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.fill_between(tgrid*1e9, V_lo95, V_hi95, alpha=0.20, color="red", label="95% credible")
    ax.fill_between(tgrid*1e9, V_lo50, V_hi50, alpha=0.40, color="red", label="50% credible")
    ax.plot(tgrid*1e9, V_mean, "r-", lw=2.5, label="posterior mean")
    ax.plot(tw*1e9, V_data, "bo", ms=3.5, alpha=0.85, label=f"data ({len(tw)} pts)")
    ax.axvline(d_m*1e9, color="orange", ls="--", lw=2, label=f"delay d ≈ {d_m*1e9:.2f} ns")
    ax.set_title(f"Delayed Double-Exponential Fit (auto-stop) | R²={r2:.6f} | RMSE(y)={rmse_y:.4g} V", fontweight="bold")
    ax.set_xlabel("Time from first negative peak (ns)")
    ax.set_ylabel("Volts (original sign)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    p2 = os.path.join(OUT_DIR, "02_fit_bayes_delayed_doubleexp.png")
    plt.savefig(p2, dpi=170)
    plt.close()
    print(f"Saved: {p2}")

    # Residual plot in V-domain
    V_fit_w = np.interp(tw, tgrid, V_mean)
    V_resid = V_data - V_fit_w

    plt.figure(figsize=(11, 4.5))
    plt.scatter(tw*1e9, V_resid, s=18, alpha=0.75)
    plt.axhline(0, lw=2, color="red")
    plt.title("Residuals vs time (V-domain, posterior mean)")
    plt.xlabel("Time from peak (ns)")
    plt.ylabel("Residual (V)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    p3 = os.path.join(OUT_DIR, "03_residuals.png")
    plt.savefig(p3, dpi=170)
    plt.close()
    print(f"Saved: {p3}")

    # Xyce-friendly equation
    print("\nXyce-style equation (t in seconds, t=0 at peak):")
    print("V(t) = - ( b + A1 + A2 )                                 for t < d")
    print("V(t) = - ( b + A1*exp(-(t-d)/tau1) + A2*exp(-(t-d)/tau2) ) for t >= d")
    print(f"b={b_m:.6f}, A1={A1_m:.6f}, tau1={t1_m:.6e}, A2={A2_m:.6f}, tau2={t2_m:.6e}, d={d_m:.6e}")

    print("\nDONE. Check these first:")
    print(f"  {os.path.join(OUT_DIR, '01_window_autostop.png')}")
    print(f"  {os.path.join(OUT_DIR, '02_fit_bayes_delayed_doubleexp.png')}")


if __name__ == "__main__":
    # Windows safe entry point
    from multiprocessing import freeze_support
    freeze_support()
    main()
