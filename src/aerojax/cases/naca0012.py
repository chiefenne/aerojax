"""R2: NACA 0012 at Re = 6e6 (NASA TMR 2DN00 verification case), SA-neg.

The TMR case is M = 0.15 compressible with a farfield at ~500 chords; here
the incompressible solver runs on the O-grid of :func:`naca0012_ogrid`
(farfield 500 chords, no vortex correction, P10). Compared: C_L, C_D and
the pressure distribution at alpha = 0, 10 and 15 degrees.

``TMR_REFERENCE`` holds the grid-converged SA values of CFL3D/FUN3D as
remembered from the TMR page (to be confirmed there); the incompressible
results are expected within a few percent (compressibility at M = 0.15
raises C_L by about 1 % through the Prandtl-Glauert factor).
"""

from typing import NamedTuple

import numpy as np

from aerojax.driver import IncompressibleProblem
from aerojax.mesh import naca0012_ogrid
from aerojax.physics import forces as fz
from aerojax.physics import incompressible as inc
from aerojax.solver import PseudoTimeSettings

TMR_REFERENCE = {
    0.0: {"cl": 0.0, "cd": 0.00819},
    10.0: {"cl": 1.0909, "cd": 0.01231},
    15.0: {"cl": 1.5091, "cd": 0.02124},
}


class AirfoilResult(NamedTuple):
    """R2 result."""

    problem: object
    w: np.ndarray
    converged: bool
    history: dict
    alpha: float
    cl: float
    cd: float
    cd_pressure: float
    cd_viscous: float
    cm: float
    x: np.ndarray
    y: np.ndarray
    cp: np.ndarray
    cf: np.ndarray
    y_plus_max: float
    reference: dict


def solve(
    alpha=10.0,
    n_surface=256,
    n_radial=128,
    reynolds=6.0e6,
    farfield_radius=500.0,
    settings=None,
    w0=None,
    mesh=None,
    callback=None,
    switch_iteration=40,
):
    """Run the NACA 0012 at the angle of attack ``alpha`` (degrees).

    ``w0`` restarts from a previous state (e.g. the next angle); ``mesh``
    replaces the generated O-grid (e.g. a PyAero mesh with boundaries
    "airfoil" and "farfield").
    """
    mesh = (
        naca0012_ogrid(n_surface, n_radial, reynolds=reynolds) if mesh is None else mesh
    )
    a = np.radians(alpha)
    conditions = {
        "airfoil": {"type": "wall"},
        "farfield": {
            "type": "farfield",
            "velocity": (np.cos(a), np.sin(a)),
            "pressure": 0.0,
        },
    }
    # Local Venkatakrishnan limiter: the global-range threshold leaves the
    # thin near-wall cells unlimited and diverges on the curved nose (D23);
    # the limiter is frozen after 1.5 orders or ``switch_iteration`` iterations,
    # where Newton-Krylov takes over (the approximate-Jacobian iteration stalls
    # one to two orders below the start on these meshes).
    model = inc.IncompressibleModel(
        density=1.0,
        viscosity=1.0 / reynolds,
        turbulence="sa-neg",
        limiter="venkatakrishnan",
    )
    problem = IncompressibleProblem(mesh, model, conditions)
    result = problem.solve(
        problem.initial_state((np.cos(a), np.sin(a))) if w0 is None else w0,
        settings
        or PseudoTimeSettings(
            residual_drop=7.0,
            max_iterations=400,
            nk_switch_residual_drop=1.5,
            nk_switch_iteration=switch_iteration,
        ),
        callback=callback,
        freeze_limiter_after_iterations=switch_iteration,
        freeze_limiter_after_residual_drop=1.5,
    )
    q = 0.5
    loads = problem.wall_loads(result.u)
    faces = problem.boundary_faces(["airfoil"])
    forces = problem.forces(result.u, ["airfoil"], (0.25, 0.0), loads=loads)
    c = fz.coefficients(forces, q, 1.0, alpha)
    d = fz.wall_distribution(loads, problem.geom, faces, q)
    owner = np.asarray(problem.geom.b_owner)[faces]
    dist = np.asarray(problem.geom.wall_distance)[owner]
    tau = np.asarray(loads.traction)[faces]
    y_plus = dist * np.sqrt(np.hypot(tau[:, 0], tau[:, 1])) * reynolds
    return AirfoilResult(
        problem,
        result.u,
        result.converged,
        result.history,
        alpha,
        c.cl,
        c.cd,
        c.cd_pressure,
        c.cd_viscous,
        c.cm,
        d["x"],
        d["y"],
        d["cp"],
        d["cf"],
        float(y_plus.max()),
        TMR_REFERENCE.get(float(alpha)),
    )
