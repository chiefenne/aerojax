"""End-to-end run of a TOML case and the command line."""

import tomllib

import numpy as np
import pytest

from aerojax.cli import main
from aerojax.mesh import rectangle, write_su2
from aerojax.run import run_case

COUETTE = """
[case]
name = "couette_test"

[mesh]
file = "channel.su2"

[physics]
model = "laminar"

[fluid]
density = 2.0
viscosity = 0.3

[reference]
velocity = 3.0
length = 0.5

[initial]
velocity = [0.0, 0.0]

[boundary.bottom]
type = "wall"

[boundary.top]
type = "wall"
velocity = [3.0, 0.0]

[boundary.left]
type = "periodic"
partner = "right"
translation = [1.0, 0.0]

[numerics]
limiter = "none"

[solver]
residual_drop = 9.0

[output]
directory = "out"
"""


@pytest.fixture
def case_dir(tmp_path):
    mesh = rectangle(6, 6, x=(0.0, 1.0), y=(0.0, 0.5), cell_type="quad")
    write_su2(tmp_path / "channel.su2", mesh)
    (tmp_path / "couette.toml").write_text(COUETTE)
    return tmp_path


def test_run_case_dimensional_couette(case_dir):
    """Dimensional inputs (rho = 2, mu = 0.3, H = 0.5 m, U = 3 m/s)."""
    result = run_case(case_dir / "couette.toml", console=False)
    assert result.converged
    y = result.mesh.cell_centroid[:, 1]
    np.testing.assert_allclose(result.w[:, 1], 3.0 * y / 0.5, atol=1e-8)
    np.testing.assert_allclose(result.w[:, 2], 0.0, atol=1e-8)
    np.testing.assert_allclose(result.w[:, 0], 0.0, atol=1e-7)
    out = case_dir / "out"
    for suffix in (".vtu", ".log", ".history.csv", ".resolved.toml"):
        assert (out / f"couette_test{suffix}").exists()
    resolved = tomllib.loads((out / "couette_test.resolved.toml").read_text())
    assert resolved["fluid"]["viscosity"] == 0.3
    log = (out / "couette_test.log").read_text()
    assert "converged              yes" in log and "log(R_u)" in log
    # Wall loads in SI: shear mu U / H = 0.3 * 3 / 0.5 = 1.8 Pa, q = 9 Pa.
    wall = np.genfromtxt(
        out / "couette_test.wall_bottom.csv", delimiter=",", names=True
    )
    np.testing.assert_allclose(wall["tau_x"], 1.8, rtol=1e-6)
    np.testing.assert_allclose(wall["cf"], 0.2, rtol=1e-6)
    assert result.coefficients.cd == pytest.approx(0.0, abs=1e-8)  # top + bottom cancel
    assert "CL,CD,CM" in (out / "couette_test.history.csv").read_text().splitlines()[0]


def test_cli_check_and_run(case_dir, capsys):
    assert main(["check", str(case_dir / "couette.toml")]) == 0
    assert '[boundary."top"]' in capsys.readouterr().out
    assert main(["run", str(case_dir / "couette.toml"), "--quiet"]) == 0
    (case_dir / "bad.toml").write_text(COUETTE.replace("limiter", "limitr"))
    assert main(["check", str(case_dir / "bad.toml")]) == 1
    assert "did you mean 'numerics.limiter'" in capsys.readouterr().err


def unsteady_couette_exact(y, t, height, velocity, viscosity, terms=200):
    """Impulsively started Couette flow (lid starts at t = 0)."""
    n = np.arange(1, terms + 1)[:, None]
    series = (
        (-1.0) ** n
        / n
        * np.exp(-(n**2) * np.pi**2 * viscosity * t / height**2)
        * np.sin(n * np.pi * y[None, :] / height)
    )
    return velocity * (y / height + 2.0 / np.pi * series.sum(axis=0))


UNSTEADY = (
    COUETTE.replace('name = "couette_test"', 'name = "couette_unsteady"')
    .replace("density = 2.0\nviscosity = 0.3", "density = 1.0\nviscosity = 0.1")
    .replace("velocity = 3.0\nlength = 0.5", "velocity = 1.0\nlength = 0.5")
    .replace("velocity = [3.0, 0.0]", "velocity = [1.0, 0.0]")
    + """
[time]
mode = "unsteady"
time_step = 0.05
end_time = 0.5
"""
)


def test_run_case_unsteady_couette(tmp_path):
    """Starting Couette flow (rho = 1, mu = 0.1, H = 0.5 m, U = 1 m/s) vs series."""
    mesh = rectangle(4, 32, x=(0.0, 1.0), y=(0.0, 0.5), cell_type="quad")
    write_su2(tmp_path / "channel.su2", mesh)
    (tmp_path / "couette.toml").write_text(
        UNSTEADY.replace('directory = "out"', 'directory = "out"\ninterval = 4')
    )
    result = run_case(tmp_path / "couette.toml", console=False)
    assert result.converged
    y = result.mesh.cell_centroid[:, 1]
    exact = unsteady_couette_exact(y, 0.5, 0.5, 1.0, 0.1)
    assert np.abs(result.w[:, 1] - exact).max() < 0.01
    out = tmp_path / "out"
    assert result.history["step"][-1] == 10
    assert (out / "couette_unsteady.pvd").exists()
    for step in (4, 8, 10):
        assert (out / f"couette_unsteady_{step:06d}.vtu").exists()
    log = (out / "couette_unsteady.log").read_text()
    assert "inner  drop" in log and "time steps             10 (bdf2" in log
    head = (out / "couette_unsteady.history.csv").read_text().splitlines()[0]
    assert head.startswith("step,time,residual_p") and "inner_residual_drop" in head
