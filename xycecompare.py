import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter, MaxNLocator
from pathlib import Path

DATA_DIR = Path(r"E:\CapstoneXyce")
SIM_FILE = DATA_DIR / "shot27294.cir.prn"
EXP_FILE = DATA_DIR / "LFXR_Lovejoy_Sept2021_27294.csv"

# =========================
# Load Xyce simulation
# =========================

def main():
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
    # Plot overlay
    # =========================
    fig, ax = plt.subplots()

    ax.plot(t_exp, v_exp, label="Experiment", alpha=0.6)
    ax.plot(t_sim, v_sim, label="Simulation", linewidth=2)

    ax.set_title("Shot 27294: Experiment vs Simulation")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage (V)")
    ax.legend()
    ax.grid(True)

    ax.xaxis.set_major_formatter(EngFormatter(unit="s"))
    ax.yaxis.set_major_formatter(EngFormatter(unit="V"))
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.yaxis.set_major_locator(MaxNLocator(6))

    plt.tight_layout()
    plt.savefig("shot27294_overlay.png", dpi=300)
    print("Overlay plot saved as shot27294_overlay.png")


if __name__ == "__main__":
    main()
