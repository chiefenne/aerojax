"""Stage 6 validation on the PyAero NACA 0012 mesh: polar and C_p vs experiment.

Runs the fully turbulent polar at Re = 6e6 (alpha 0, 4, 8, 12 as a restart
chain, about 30 minutes on two cores) with ``aerojax polar`` and plots it
against Ladson's tripped data (NASA TM 4074) and the TMR SA values, and the
C_p at alpha = 10 against Gregory & O'Reilly (NASA R&M 3726, Re 2.88e6).

    python examples/stage6_pyaero_validation.py [--plot-only]

Output: ``examples/cases/output/naca0012_pyaero_rans/`` (runs) and
``examples/output/stage6_pyaero_validation.png``.
"""

import subprocess
import sys
from importlib.resources import files
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
CASE = HERE / "cases" / "naca0012_pyaero_rans.toml"
RUNS = HERE / "cases" / "output" / "naca0012_pyaero_rans"
OUT = HERE / "output"
ANGLES = (0.0, 4.0, 8.0, 10.0, 12.0)
COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
TMR = {0.0: (0.0, 0.0081925), 10.0: (1.0909, 0.012311), 15.0: (1.5461, 0.021237)}


def data(name):
    """Return packaged experimental data as a structured array."""
    path = files("aerojax.cases") / "data" / name
    # Two comment lines (source) precede the header.
    return np.genfromtxt(path, delimiter=",", names=True, skip_header=2)


def main():
    """Run the polar (unless --plot-only) and plot."""
    OUT.mkdir(exist_ok=True)
    if "--plot-only" not in sys.argv:
        args = [sys.executable, "-m", "aerojax.cli", "polar", str(CASE), "--alpha"]
        subprocess.run(args + [f"{a:g}" for a in ANGLES], check=True)
    polar = np.genfromtxt(
        RUNS / "naca0012_pyaero_rans.polar.csv", delimiter=",", names=True
    )
    for row in polar:
        print(
            f"  alpha {row['alpha']:5.1f}: CL = {row['CL']:.4f}, CD = {row['CD']:.5f} "
            f"(pressure {row['CD_pressure']:.5f}, viscous {row['CD_viscous']:.5f}), "
            f"{int(row['iterations'])} iterations"
        )
    plot(polar)


def plot(polar):
    """Polar vs Ladson/TMR, C_p vs Gregory & O'Reilly."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: no plot")
        return

    ladson = data("naca0012_ladson_re6e6_tripped.csv")
    gregory = data("naca0012_gregory_cp.csv")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    ax = axes[0]
    for grit, marker in (("80", "s"), ("120", "^")):
        sel = (
            ladson["grit"] == int(grit)
            if ladson["grit"].dtype.kind != "U"
            else ladson["grit"] == grit
        )
        ax.plot(
            ladson["alpha"][sel],
            ladson["cl"][sel],
            marker,
            color="0.4",
            mfc="white",
            ms=6,
            label=f"Ladson, {grit} grit",
        )
    ax.plot(
        [a for a in TMR], [TMR[a][0] for a in TMR], "k*", ms=11, label="TMR (CFL3D, SA)"
    )
    ax.plot(
        polar["alpha"],
        polar["CL"],
        "o-",
        color=COLORS[0],
        lw=2,
        ms=7,
        label="aerojax, PyAero mesh",
    )
    ax.set(xlabel="α [deg]", ylabel="CL", title="NACA 0012, Re = 6e6, fully turbulent")

    ax = axes[1]
    for grit, marker in (("80", "s"), ("120", "^")):
        sel = (
            ladson["grit"] == int(grit)
            if ladson["grit"].dtype.kind != "U"
            else ladson["grit"] == grit
        )
        ax.plot(
            ladson["alpha"][sel],
            ladson["cd"][sel],
            marker,
            color="0.4",
            mfc="white",
            ms=6,
            label=f"Ladson, {grit} grit",
        )
    ax.plot(
        [a for a in TMR], [TMR[a][1] for a in TMR], "k*", ms=11, label="TMR (CFL3D, SA)"
    )
    ax.plot(
        polar["alpha"], polar["CD"], "o-", color=COLORS[0], lw=2, ms=7, label="aerojax"
    )
    ax.plot(
        polar["alpha"],
        polar["CD_viscous"],
        "o--",
        color=COLORS[1],
        lw=1.5,
        ms=5,
        label="aerojax, viscous part",
    )
    ax.set(xlabel="α [deg]", ylabel="CD", title="drag", xlim=(-1, 16), ylim=(0, 0.03))

    ax = axes[2]
    wall_csv = RUNS / "alpha_10" / "naca0012_pyaero_rans_alpha_10.wall_airfoil.csv"
    wall = (
        np.genfromtxt(wall_csv, delimiter=",", names=True)
        if wall_csv.exists()
        else None
    )
    if wall is not None:
        for sel, ls, label in (
            (wall["y"] >= 0, "-", "aerojax upper"),
            (wall["y"] < 0, "--", "aerojax lower"),
        ):
            o = np.argsort(wall["x"][sel])
            ax.plot(
                wall["x"][sel][o],
                wall["cp"][sel][o],
                ls,
                color=COLORS[0],
                lw=1.8,
                label=label,
            )
    g = gregory[gregory["alpha"] == 10.0]
    ax.plot(
        g["x_c"],
        g["cp"],
        "o",
        color="0.3",
        mfc="white",
        ms=5,
        label="Gregory & O'Reilly, Re 2.88e6, upper",
    )
    ax.invert_yaxis()
    ax.set(xlabel="x / c", ylabel="Cp", title="α = 10°")
    for ax in axes:
        ax.grid(True, color="0.88", lw=0.8)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "stage6_pyaero_validation.png", dpi=130)
    print(f"plot: {OUT / 'stage6_pyaero_validation.png'}")


if __name__ == "__main__":
    main()
