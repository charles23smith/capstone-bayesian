import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter, MaxNLocator
from pathlib import Path
import argparse

DATA_DIR = Path(r"C:\CapstoneXyce")
SIM_FILE = DATA_DIR / "shot27291.cir.prn"
EXP_FILE = DATA_DIR / "LFXR_Lovejoy_Sept2021_27291.csv"


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
    v_sim = sim["V(TRIG)"].to_numpy()

    # =========================
    # Load experimental data
    # =========================
    exp = pd.read_csv(args.exp_file)

    t_exp = exp["time1"].to_numpy()
    v_exp = exp["Diode"].to_numpy() * -1

    # Shift experimental time so it starts at 0 (like you did for pulse generation)
    t_exp = t_exp - np.min(t_exp)

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
    fig, ax = plt.subplots(figsize=(8.5, 6.0))

    ax.plot(t_exp, v_exp, color="#bfbfbf", linewidth=1.8, label="Experiment")
    ax.plot(t_sim, v_sim, color="#1b9e77", linewidth=2.0, label="Simulation")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage across diode (V)")
    ax.legend(loc="upper right", frameon=False)
    ax.grid(True, alpha=0.25)

    ax.xaxis.set_major_formatter(EngFormatter(unit="s"))
    ax.yaxis.set_major_formatter(EngFormatter(unit="V"))
    ax.xaxis.set_major_locator(MaxNLocator(7))
    ax.yaxis.set_major_locator(MaxNLocator(8))
    ax.set_xlim(0.0, 3.0e-6)
    ax.set_ylim(0.0, 2.0)

    plt.tight_layout()
    out_path = f"shot{file_label}_overlay.png"
    plt.savefig(out_path, dpi=300)
    print(f"Overlay plot saved as {out_path}")
    plt.close(fig)

    fig_zoom, ax_zoom = plt.subplots(figsize=(8.5, 6.0))
    ax_zoom.plot(t_exp, v_exp, color="#bfbfbf", linewidth=1.8, label="Experiment")
    ax_zoom.plot(t_sim, v_sim, color="#1b9e77", linewidth=2.0, label="Simulation")

    ax_zoom.set_xlabel("Time (s)")
    ax_zoom.set_ylabel("Voltage across diode (V)")
    ax_zoom.legend(loc="upper right", frameon=False)
    ax_zoom.grid(True, alpha=0.25)

    ax_zoom.xaxis.set_major_formatter(EngFormatter(unit="s"))
    ax_zoom.yaxis.set_major_formatter(EngFormatter(unit="V"))
    ax_zoom.xaxis.set_major_locator(MaxNLocator(7))
    ax_zoom.yaxis.set_major_locator(MaxNLocator(8))
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
