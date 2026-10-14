"""Cylinder cases: V4 inviscid cylinder and B2 DFG benchmark 2D-1.

V4: potential flow around a cylinder of radius R in a uniform stream U,
C_p = 1 - 4 sin^2(theta), zero drag and lift. The Euler equations on an
O-grid with a slip wall on the cylinder and a farfield boundary approach this
solution under mesh refinement.

B2: laminar flow around a cylinder in a channel (Schaefer & Turek 1996,
benchmark 2D-1): channel 2.2 x 0.41, cylinder of diameter D = 0.1 at
(0.2, 0.2), parabolic inflow with U_max = 0.3 (mean velocity U_m = 0.2),
nu = 1e-3, Re = U_m D / nu = 20. Reference values (featflow, Nabh 1998):
c_D = 5.57953523384, c_L = 0.010618948146, Delta p = p(0.15, 0.2) -
p(0.25, 0.2) = 0.11752016697, coefficients with q = rho U_m^2 / 2 and D.
"""

from typing import NamedTuple

import numpy as np

from aerojax.driver import IncompressibleProblem
from aerojax.mesh import annulus, cylinder_in_channel
from aerojax.physics import forces as fz
from aerojax.physics import incompressible as inc
from aerojax.solver import PseudoTimeSettings

DFG_REFERENCE = {"cd": 5.57953523384, "cl": 0.010618948146, "dp": 0.11752016697}


class InviscidResult(NamedTuple):
    """V4 result."""

    problem: object
    w: np.ndarray
    converged: bool
    history: dict
    theta: np.ndarray
    cp: np.ndarray
    cp_exact: np.ndarray
    cd: float
    cl: float


def inviscid(n_theta=64, n_radial=32, outer_radius=20.0, settings=None):
    """Euler flow around a cylinder on an O-grid (V4).

    Parameters
    ----------
    n_theta, n_radial : int
        O-grid resolution.
    outer_radius : float
        Farfield radius in cylinder radii (R = 0.5, so D = 1).
    settings : PseudoTimeSettings, optional

    Returns
    -------
    InviscidResult
        ``theta`` and ``cp`` per cylinder face, ``cp_exact`` = 1 - 4 sin^2.
    """
    radius = 0.5
    mesh = annulus(
        n_theta,
        n_radial,
        radius=radius,
        outer_radius=outer_radius * radius,
        growth=1.0 + 6.0 / n_radial,
    )
    conditions = {
        "cylinder": {"type": "slip-wall"},
        "farfield": {"type": "farfield", "velocity": (1.0, 0.0), "pressure": 0.0},
    }
    model = inc.IncompressibleModel(density=1.0, viscosity=0.0)
    problem = IncompressibleProblem(mesh, model, conditions)
    result = problem.solve(
        problem.initial_state((1.0, 0.0)),
        settings or PseudoTimeSettings(residual_drop=8.0, max_iterations=400),
    )
    faces = problem.boundary_faces(["cylinder"])
    q = 0.5
    d = fz.wall_distribution(problem.wall_loads(result.u), problem.geom, faces, q)
    theta = np.arctan2(d["y"], d["x"])
    c = fz.coefficients(problem.forces(result.u, ["cylinder"]), q, 2.0 * radius)
    return InviscidResult(
        problem,
        result.u,
        result.converged,
        result.history,
        theta,
        d["cp"],
        1.0 - 4.0 * np.sin(theta) ** 2,
        c.cd,
        c.cl,
    )


class DFGResult(NamedTuple):
    """B2 result with the reference values of benchmark 2D-1."""

    problem: object
    w: np.ndarray
    converged: bool
    history: dict
    cd: float
    cl: float
    dp: float
    reference: dict


def dfg_2d1(m=32, n_radial=20, cell_type="quad", settings=None):
    """DFG benchmark 2D-1 (Re = 20) on the block-structured mesh.

    Parameters
    ----------
    m : int
        Cells per side of the square around the cylinder (mesh resolution).
    n_radial : int
        Cells between cylinder and square.
    cell_type : {"quad", "triangle"}
    settings : PseudoTimeSettings, optional

    Returns
    -------
    DFGResult
    """
    height, u_max, nu, diameter = 0.41, 0.3, 1.0e-3, 0.1
    centre = (0.2, 0.2)
    mesh = cylinder_in_channel(m, n_radial, cell_type=cell_type, growth=1.12)

    def inflow(x, y):
        return 4.0 * u_max * y * (height - y) / height**2, 0.0 * y

    conditions = {
        "inlet": {"type": "velocity-inlet", "velocity": inflow},
        "outlet": {"type": "pressure-outlet", "pressure": 0.0},
        "wall": {"type": "wall"},
        "cylinder": {"type": "wall"},
    }
    model = inc.IncompressibleModel(density=1.0, viscosity=nu)
    problem = IncompressibleProblem(mesh, model, conditions)
    result = problem.solve(
        problem.initial_state((2.0 * u_max / 3.0, 0.0)),
        settings or PseudoTimeSettings(residual_drop=8.0, max_iterations=300),
    )
    u_mean = 2.0 * u_max / 3.0
    q = 0.5 * u_mean**2
    forces = problem.forces(result.u, ["cylinder"], centre)
    c = fz.coefficients(forces, q, diameter)
    loads = problem.wall_loads(result.u)
    faces = problem.boundary_faces(["cylinder"])
    front = surface_pressure(loads, problem.geom, faces, (centre[0] - 0.05, centre[1]))
    back = surface_pressure(loads, problem.geom, faces, (centre[0] + 0.05, centre[1]))
    return DFGResult(
        problem,
        result.u,
        result.converged,
        result.history,
        c.cd,
        c.cl,
        front - back,
        DFG_REFERENCE,
    )


def surface_pressure(loads, geom, faces, point):
    """Pressure at a point on a wall: inverse-distance of the two nearest faces."""
    centres = np.asarray(geom.b_centre)[faces]
    p = np.asarray(loads.pressure)[faces]
    d = np.linalg.norm(centres - np.asarray(point), axis=1)
    k = np.argsort(d)[:2]
    w = 1.0 / np.maximum(d[k], 1e-300)
    return float(np.sum(p[k] * w) / np.sum(w))
