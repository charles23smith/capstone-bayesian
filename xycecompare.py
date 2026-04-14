import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter, MaxNLocator, MultipleLocator
from pathlib import Path
import argparse

DATA_DIR = Path(r"E:\CapstoneXyce")
SIM_FILE = DATA_DIR / "shot27298.cir.prn"
EXP_FILE = DATA_DIR / "LFXR_Lovejoy_Sept2021_27298.csv"

ZOOM_LEFT_PAD_NS = 200.0
ZOOM_RIGHT_PAD_NS = 600.0
PCD_TARGET_V = 0.5
T0_SHIFT_NS = 100.0
PCD_EXCLUDE = {"PCD3_B"}

PLOT_PRESETS_NS = {
    "27271": {
        "full_xlim": (-500.0, 5000.0),
        "zoom_xlim": (3200.0, 6000.0),
        "ylim": (-6.45, 2.0),
    },
    "27272": {
        "full_xlim": (-250.0, 5000.0),
        "zoom_xlim": (-175.0, 1200.0),
        "ylim": (-5.05, 0.22),
    },
    "27273": {
        "full_xlim": (-50.0, 950.0),
        "zoom_xlim": (-50.0, 950.0),
        "ylim": (-1.2, 15.5),
    },
    "27275": {
        "full_xlim": (-120.0, 500.0),
        "zoom_xlim": (-120.0, 500.0),
        "ylim": (-10.5, 8.0),
        "sim_label": "modeled (exp1 + exp2)",
    },
    "27276": {
        "full_xlim": (-120.0, 4250.0),
        "zoom_xlim": (-120.0, 4250.0),
        "ylim": (-2.05, 0.2),
        "sim_label": "modeled (exp1 + exp2)",
    },
    "27277": {
        "full_xlim": (-120.0, 700.0),
        "zoom_xlim": (-120.0, 700.0),
        "ylim": (-5.0, 0.1),
        "sim_label": "modeled (exp1 + exp2)",
    },
    "27290": {
        "full_xlim": (-8.0e-7, 2.0e-7),
        "zoom_xlim": (-8.0e-7, 2.0e-7),
        "ylim": (0.0, 1.25),
        "exp_label": "Experimental",
        "sim_label": "Simulated",
        "positive_axis": True,
        "eng_axes": True,
        "x_major_step": 2.0e-7,
        "plot_time_offset_s": -6.0e-7,
        "xlabel": "Time (s)",
        "ylabel": "Voltage (V)",
        "title": "27290 Experiment vs Simulated",
        "show_t0_marker": False,
    },
    "27294": {
        "full_xlim": (0.0, 5.0e-6),
        "zoom_xlim": (0.0, 5.0e-6),
        "ylim": (0.0, 1.18),
        "sim_label": "modeled (exp1 + exp2)",
        "positive_axis": True,
        "eng_axes": True,
        "x_major_step": 1.0e-6,
        "xlabel": "Time (s)",
        "ylabel": "Voltage (V)",
        "title": "Shot 27294: Experiment vs Simulation",
    },
    "27296": {
        "full_xlim": (0.0, 1.5e-6),
        "zoom_xlim": (0.0, 1.5e-6),
        "ylim": (0.0, 2.0),
        "sim_label": "Simulation",
        "positive_axis": True,
        "eng_axes": True,
        "x_major_step": 2.5e-7,
        "xlabel": "Time (s)",
        "ylabel": "Voltage (V)",
        "title": "27296 Experiment vs Simulated",
        "show_t0_marker": False,
    },
    "27298": {
        "full_xlim": (0.0, 1.25e-6),
        "zoom_xlim": (0.0, 1.25e-6),
        "ylim": (-0.9, 1.25),
        "sim_label": "modeled (exp1 + exp2)",
        "eng_axes": True,
        "x_major_step": 2.0e-7,
        "xlabel": "Time (s)",
        "ylabel": "Voltage (V)",
        "title": "27298 Experiment vs Simulated",
        "show_t0_marker": False,
    },
    "27291": {
        "full_xlim": (-100.0, 800.0),
        "zoom_xlim": (-100.0, 800.0),
        "ylim": (-0.72, 0.08),
    },
    "27306": {
        "full_xlim": (-1000.0, 60000.0),
        "zoom_xlim": (-1000.0, 60000.0),
        "ylim": (-3.05, 3.05),
    },
    "27307": {
        "full_xlim": (0.0, 6.0e-5),
        "zoom_xlim": (0.0, 6.0e-5),
        "ylim": (0.0, 3.0),
        "exp_label": "Experimental",
        "sim_label": "Simulation",
        "positive_axis": True,
        "eng_axes": True,
        "x_major_step": 1.0e-5,
        "xlabel": "Time (s)",
        "ylabel": "Voltage (V)",
        "title": "27307 Experiment vs Simulated",
        "show_t0_marker": False,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Compare Xyce output against experimental shot data.")
    parser.add_argument("--sim-file", type=Path, default=SIM_FILE, help="Path to the Xyce .prn file.")
    parser.add_argument("--exp-file", type=Path, default=EXP_FILE, help="Path to the experimental CSV file.")
    parser.add_argument(
        "--shot-label",
        type=str,
        default=None,
        help="Override the shot label used in output filenames, e.g. 27291 or 27291*.",
    )
    parser.add_argument(
        "--zero-impedance",
        action="store_true",
        help="Mark the run as the 0-impedance variant. Appends '*' to the shot label.",
    )
    return parser.parse_args()


def infer_shot_label(sim_file: Path, exp_file: Path) -> str:
    for path in (sim_file, exp_file):
        digits = "".join(ch for ch in path.stem if ch.isdigit())
        if digits:
            return digits
    return "unknown"


def resolve_shot_label(args) -> str:
    if args.shot_label:
        shot_label = args.shot_label
    else:
        shot_label = infer_shot_label(args.sim_file, args.exp_file)

    if args.zero_impedance and not shot_label.endswith("*"):
        shot_label = f"{shot_label}*"

    return shot_label


def base_shot_label(shot_label: str) -> str:
    return shot_label.rstrip("*")


def safe_shot_label_for_filename(shot_label: str) -> str:
    return shot_label.replace("*", "_star")


def plot_preset_for_shot(base_label: str):
    return PLOT_PRESETS_NS.get(base_label, None)


def auto_window_from_sim_ns(t_sim_ns: np.ndarray, v_sim: np.ndarray):
    if len(t_sim_ns) == 0:
        return None

    baseline_n = max(5, min(len(v_sim) // 10, 50))
    baseline = float(np.median(v_sim[:baseline_n]))
    v_span = float(np.nanmax(v_sim) - np.nanmin(v_sim))
    threshold = max(0.2, 0.05 * v_span)

    active_idx = np.flatnonzero(np.abs(v_sim - baseline) >= threshold)
    onset_idx = int(active_idx[0]) if len(active_idx) else 0
    stop_idx = len(t_sim_ns) - 1

    return (
        float(t_sim_ns[onset_idx] - ZOOM_LEFT_PAD_NS),
        float(t_sim_ns[stop_idx] + ZOOM_RIGHT_PAD_NS),
    )


def first_crossing_time_or_nearest(t: np.ndarray, y: np.ndarray, target: float):
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    above = np.flatnonzero(y >= target)
    if len(above):
        i = int(above[0])
        if i > 0:
            y0 = float(y[i - 1])
            y1 = float(y[i])
            t0 = float(t[i - 1])
            t1 = float(t[i])
            if abs(y1 - y0) > 1e-18:
                frac = float((target - y0) / (y1 - y0))
                frac = float(np.clip(frac, 0.0, 1.0))
                return t0 + frac * (t1 - t0), "crossing"
        return float(t[i]), "crossing"

    i = int(np.argmin(np.abs(y - target)))
    return float(t[i]), "nearest"


def experimental_t0_from_pcd(exp: pd.DataFrame):
    pcd_cols = [c for c in exp.columns if c.startswith("PCD") and c not in PCD_EXCLUDE]
    if not pcd_cols:
        return None, None, []

    time_col = None
    for candidate in ("time3", "time2", "time1"):
        if candidate in exp.columns:
            time_col = candidate
            break
    if time_col is None:
        return None, None, pcd_cols

    t_pcd = pd.to_numeric(exp[time_col], errors="coerce").to_numpy()
    pcd_avg = exp[pcd_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1).to_numpy()
    mask = np.isfinite(t_pcd) & np.isfinite(pcd_avg)
    if not np.any(mask):
        return None, None, pcd_cols

    t_cross, mode = first_crossing_time_or_nearest(t_pcd[mask], pcd_avg[mask], PCD_TARGET_V)
    t0 = float(t_cross - T0_SHIFT_NS * 1e-9)
    return t0, mode, pcd_cols

# =========================
# Load Xyce simulation
# =========================

def main():
    args = parse_args()
    shot_label = resolve_shot_label(args)
    base_label = base_shot_label(shot_label)
    file_label = safe_shot_label_for_filename(shot_label)
    sim = pd.read_csv(
        args.sim_file,
        sep=r"\s+",
        engine="python"
    )

    sim["TIME"] = pd.to_numeric(sim["TIME"], errors="coerce")
    sim["V(TRIG)"] = pd.to_numeric(sim["V(TRIG)"], errors="coerce")

    sim = sim.dropna(subset=["TIME", "V(TRIG)"])

    t_sim = sim["TIME"].to_numpy()
    v_sim = -sim["V(TRIG)"].to_numpy()
    preset = plot_preset_for_shot(base_label)
    # =========================
    # Load experimental data
    # =========================
    exp = pd.read_csv(args.exp_file)

    t_exp = pd.to_numeric(exp["time1"], errors="coerce").to_numpy()
    v_exp = pd.to_numeric(exp["Diode"], errors="coerce").to_numpy()
    valid_exp = np.isfinite(t_exp) & np.isfinite(v_exp)
    t_exp = t_exp[valid_exp]
    v_exp = v_exp[valid_exp]

    exp_t0, exp_t0_mode, pcd_cols = experimental_t0_from_pcd(exp)
    if exp_t0 is not None:
        t_exp = t_exp - exp_t0
        print(
            f"Experimental t0 from PCD avg ({', '.join(pcd_cols)}), mode={exp_t0_mode}: "
            f"{exp_t0:.9e} s"
        )
    else:
        t_exp = t_exp - np.min(t_exp)
        print("Experimental t0 fallback: using minimum experimental time as 0.")

    # =========================
    # Interpolate experiment onto simulation time grid
    # =========================
    v_exp_interp = np.interp(t_sim, t_exp, v_exp)

    # =========================
    # Compute Loss
    # =========================
    mse = np.mean((v_sim - v_exp_interp)**2)
    rmse = np.sqrt(mse)

    print(f"MSE  = {mse:.6e}")
    print(f"RMSE = {rmse:.6e}")

    # =========================
    # Plot single overlay
    # =========================
    if preset is not None:
        positive_axis = bool(preset.get("positive_axis", False))
        eng_axes = bool(preset.get("eng_axes", False))
        plot_time_offset_s = float(preset.get("plot_time_offset_s", 0.0))
        v_exp_plot = (-v_exp if positive_axis else v_exp).astype(float)
        v_sim_plot = (-v_sim if positive_axis else v_sim).astype(float)
        if eng_axes:
            t_exp_plot = (t_exp + plot_time_offset_s).astype(float)
            t_sim_plot = (t_sim + plot_time_offset_s).astype(float)
            auto_window = auto_window_from_sim_ns((t_sim_plot * 1e9).astype(float), v_sim_plot)
        else:
            t_exp_plot = ((t_exp + plot_time_offset_s) * 1e9).astype(float)
            t_sim_plot = ((t_sim + plot_time_offset_s) * 1e9).astype(float)
            auto_window = auto_window_from_sim_ns(t_sim_plot, v_sim_plot)
        exp_color = "#404040"
        sim_color = "red"
        xlabel = preset.get("xlabel", "Time")
        ylabel = preset.get("ylabel", "Volts")
        exp_label = preset.get("exp_label", "Diode (actual)")
        sim_label = preset.get("sim_label", "Simulation")
        title = preset.get("title", "Modeled Waveform Overlay")
    else:
        t_exp_plot = t_exp
        t_sim_plot = t_sim
        auto_window = None
        v_exp_plot = v_exp
        v_sim_plot = v_sim
        exp_color = "#bfbfbf"
        sim_color = "#1b9e77"
        xlabel = "Time (s)"
        ylabel = "Voltage across diode (V)"
        exp_label = "Experiment"
        sim_label = "Simulation"
        title = "Xyce vs Experiment Overlay"

    fig, ax = plt.subplots(figsize=(12.8, 5.6) if preset is not None else (8.5, 6.0))

    show_t0_marker = bool(preset.get("show_t0_marker", True)) if preset is not None else False

    ax.plot(t_exp_plot, v_exp_plot, color=exp_color, linewidth=1.6 if preset is not None else 1.8, label=exp_label)
    ax.plot(t_sim_plot, v_sim_plot, color=sim_color, linewidth=2.4 if preset is not None else 2.0, label=sim_label)
    if preset is not None and show_t0_marker:
        ax.axvline(0.0, color="red", linestyle="--", linewidth=2.0, label="t0 (PCD 0.5 V - 100ns)")

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper right", frameon=True if preset is not None else False)
    ax.grid(True, alpha=0.25)

    if preset is None or preset.get("eng_axes", False):
        ax.xaxis.set_major_formatter(EngFormatter(unit="s"))
        ax.yaxis.set_major_formatter(EngFormatter(unit="V"))
    x_major_step = preset.get("x_major_step") if preset is not None else None
    if x_major_step is not None:
        ax.xaxis.set_major_locator(MultipleLocator(x_major_step))
    else:
        ax.xaxis.set_major_locator(MaxNLocator(7))
    ax.yaxis.set_major_locator(MaxNLocator(8))
    if preset is not None:
        full_xlim = preset.get("full_xlim", auto_window)
        if full_xlim is not None:
            ax.set_xlim(*full_xlim)
        ax.set_ylim(*preset["ylim"])
    else:
        ax.set_xlim(0.0, 3.0e-6)
        ax.set_ylim(0.0, 2.0)

    plt.tight_layout()
    out_path = f"shot{file_label}_overlay.png"
    plt.savefig(out_path, dpi=300)
    print(f"Overlay plot saved as {out_path}")
    plt.close(fig)

    fig_zoom, ax_zoom = plt.subplots(figsize=(13.0, 5.6) if preset is not None else (8.5, 6.0))
    ax_zoom.plot(t_exp_plot, v_exp_plot, color=exp_color, linewidth=1.6 if preset is not None else 1.8, label=exp_label)
    ax_zoom.plot(t_sim_plot, v_sim_plot, color=sim_color, linewidth=2.4 if preset is not None else 2.0, label=sim_label)
    if preset is not None and show_t0_marker:
        ax_zoom.axvline(0.0, color="red", linestyle="--", linewidth=2.0, label="t0 (PCD 0.5 V - 100ns)")

    ax_zoom.set_title(title, fontweight="bold")
    ax_zoom.set_xlabel(xlabel)
    ax_zoom.set_ylabel(ylabel)
    ax_zoom.legend(loc="upper right", frameon=True if preset is not None else False)
    ax_zoom.grid(True, alpha=0.25)

    if preset is None or preset.get("eng_axes", False):
        ax_zoom.xaxis.set_major_formatter(EngFormatter(unit="s"))
        ax_zoom.yaxis.set_major_formatter(EngFormatter(unit="V"))
    if x_major_step is not None:
        ax_zoom.xaxis.set_major_locator(MultipleLocator(x_major_step))
    else:
        ax_zoom.xaxis.set_major_locator(MaxNLocator(7))
    ax_zoom.yaxis.set_major_locator(MaxNLocator(8))
    if preset is not None:
        zoom_xlim = preset.get("zoom_xlim", auto_window)
        if zoom_xlim is not None:
            ax_zoom.set_xlim(*zoom_xlim)
        ax_zoom.set_ylim(*preset["ylim"])
    else:
        ax_zoom.set_xlim(0.0, 1.5e-6)
        ax_zoom.set_ylim(0.0, 2.0)

    plt.tight_layout()
    zoom_out_path = f"shot{file_label}_overlay_zoom_0_1p5us.png"
    plt.savefig(zoom_out_path, dpi=300)
    print(f"Zoom overlay plot saved as {zoom_out_path}")
    plt.close(fig_zoom)

    if base_label == "27291":
        fig_special, ax_special = plt.subplots(figsize=(8.5, 6.0))
        ax_special.plot(t_exp, v_exp, color="#bfbfbf", linewidth=1.8, label="Experiment")
        ax_special.plot(t_sim, v_sim, color="#1b9e77", linewidth=2.0, label="Simulation")

        ax_special.set_xlabel("Time (s)")
        ax_special.set_ylabel("Voltage across diode (V)")
        ax_special.legend(loc="upper right", frameon=False)
        ax_special.grid(True, alpha=0.25)

        ax_special.xaxis.set_major_formatter(EngFormatter(unit="s"))
        ax_special.yaxis.set_major_formatter(EngFormatter(unit="V"))
        ax_special.xaxis.set_major_locator(MaxNLocator(6))
        ax_special.yaxis.set_major_locator(MaxNLocator(8))
        ax_special.set_xlim(250e-9, 500e-9)
        ax_special.set_ylim(0.0, 2.0)

        plt.tight_layout()
        special_out_path = f"shot{file_label}_overlay_zoom_250_500ns.png"
        plt.savefig(special_out_path, dpi=300)
        print(f"Special zoom overlay plot saved as {special_out_path}")
        plt.close(fig_special)


if __name__ == "__main__":
    main()
