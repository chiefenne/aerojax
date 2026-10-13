"""Implicit pseudo-time stepping with SER CFL control (D25, D35).

Each iteration solves

    (diag(A_i / dtau_i) + J1) du = -R(u),    dtau_i = CFL * A_i / lambda_i,

with the approximate Jacobian J1 and the spectral radius lambda_i of cell i,
so the diagonal term is lambda_i / CFL. The CFL number follows switched
evolution relaxation (Mulder & van Leer 1985):

    CFL_n = min(CFL_max, CFL_(n-1) * |R_(n-1)| / |R_n|).

Safeguards: if the linear solve fails, the update is not finite, or the
residual grows by more than ``reject_factor``, the update is rejected and the
CFL number is multiplied by ``cut_factor``.
"""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import scipy.sparse as sp

from aerojax.solver.linear import LinearSolverError, solve_direct


@dataclass(frozen=True)
class PseudoTimeSettings:
    """Settings of the pseudo-time iteration.

    Attributes
    ----------
    cfl_start : float
        Initial CFL number.
    cfl_max : float
        Upper limit of the CFL number.
    cut_factor : float
        CFL reduction after a rejected update.
    reject_factor : float
        An update is rejected if it raises the residual norm by more than
        this factor.
    cfl_min : float
        The iteration stops (not converged) if the CFL number drops below.
    max_iterations : int
        Maximum number of iterations, rejected ones included.
    residual_drop : float
        Convergence: orders of magnitude of residual reduction.
    """

    cfl_start: float = 10.0
    cfl_max: float = 1.0e10
    cut_factor: float = 0.1
    reject_factor: float = 10.0
    cfl_min: float = 1.0e-6
    max_iterations: int = 1000
    residual_drop: float = 8.0


class SteadyResult(NamedTuple):
    """Result of :func:`solve_steady`.

    Attributes
    ----------
    u : ndarray, shape (n_cells, n_var)
        Final solution.
    converged : bool
        True if the residual dropped by ``residual_drop`` orders.
    history : dict of str to ndarray
        Per iteration: ``iteration``, ``residual`` (RMS norm after the
        iteration), ``cfl`` (CFL number used), ``accepted``.
    """

    u: np.ndarray
    converged: bool
    history: dict


def rms(r):
    """Root-mean-square norm of a residual array."""
    return float(np.sqrt(np.mean(np.square(r))))


def solve_steady(residual, jacobian, spectral_radius, u0, settings=None):
    """Drive ``residual(u)`` to zero by implicit pseudo-time stepping.

    Parameters
    ----------
    residual : callable
        ``residual(u) -> (n_cells, n_var)`` array.
    jacobian : callable
        ``jacobian(u) -> scipy.sparse`` approximate Jacobian dR/du.
    spectral_radius : callable
        ``spectral_radius(u) -> (n_cells,)``: sum over faces of the flux
        eigenvalue magnitude times face length.
    u0 : array, shape (n_cells, n_var)
        Initial solution.
    settings : PseudoTimeSettings, optional

    Returns
    -------
    SteadyResult
    """
    s = settings or PseudoTimeSettings()
    u = np.array(u0, dtype=float)
    r = np.asarray(residual(u))
    norm0 = norm = rms(r)
    target = norm0 * 10.0 ** (-s.residual_drop)
    cfl = s.cfl_start
    hist = {"iteration": [0], "residual": [norm], "cfl": [cfl], "accepted": [True]}
    converged = norm <= target
    for it in range(1, s.max_iterations + 1):
        if converged or cfl < s.cfl_min:
            break
        n_var = u.shape[1]
        diag = np.repeat(np.asarray(spectral_radius(u)) / cfl, n_var)
        matrix = jacobian(u) + sp.diags(diag)
        accepted = False
        try:
            du = solve_direct(matrix, -r.ravel()).reshape(u.shape)
            u_new = u + du
            r_new = np.asarray(residual(u_new))
            norm_new = rms(r_new)
            accepted = np.isfinite(norm_new) and norm_new <= s.reject_factor * norm
        except LinearSolverError:
            pass
        hist["iteration"].append(it)
        hist["cfl"].append(cfl)
        hist["accepted"].append(accepted)
        if accepted:
            cfl = min(s.cfl_max, cfl * norm / max(norm_new, np.finfo(float).tiny))
            u, r, norm = u_new, r_new, norm_new
            converged = norm <= target
        else:
            cfl *= s.cut_factor
        hist["residual"].append(norm)
    history = {k: np.asarray(v) for k, v in hist.items()}
    return SteadyResult(u, bool(converged), history)
