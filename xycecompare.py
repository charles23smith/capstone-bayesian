import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter, MaxNLocator
from pathlib import Path

DATA_DIR = Path(r"E:\CapstoneXyce")
SIM_FILE = DATA_DIR / "shot27296.cir.prn"
EXP_FILE = DATA_DIR / "LFXR_Lovejoy_Sept2021_27296.csv"


def infer_shot_label() -> str:
    for path in (SIM_FILE, EXP_FILE):
        digits = "".join(ch for ch in path.stem if ch.isdigit())
        if digits:
            return digits
    return "unknown"

# =========================
# Load Xyce simulation
# =========================

def main():
    shot_label = infer_shot_label()
    sim = pd.read_csv(
        SIM_FILE,
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
    exp = pd.read_csv(EXP_FILE)

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
    out_path = f"shot{shot_label}_overlay.png"
    plt.savefig(out_path, dpi=300)
    print(f"Overlay plot saved as {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
