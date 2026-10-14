"""Implicit pseudo-time stepping with SER CFL control (D25, D35, D36).

Each iteration solves

    (T(u, CFL) + J1(u)) du = -R(u),

with the approximate Jacobian J1 and the pseudo-time matrix T. For a scalar
equation T = diag(lambda_i / CFL) with the spectral radius lambda_i of cell i
(the pseudo-time step is dtau_i = CFL A_i / lambda_i); for the preconditioned
flow equations T holds the blocks Gamma_i lambda_i / CFL. The CFL number
follows switched evolution relaxation (Mulder & van Leer 1985):

    CFL_n = min(CFL_max, CFL_(n-1) * max(|R_(n-1)| / |R_n|, growth_factor)),

i.e. the CFL number grows at least by ``growth_factor`` per accepted
iteration while the residual decreases (SU2 raises its CFL by a factor 2
under the same condition); during a slow transient SER alone would keep it
small. Safeguards: if the linear solve fails, the update is not finite, or the
residual grows by more than ``reject_factor``, the update is rejected and the
CFL number is multiplied by ``cut_factor``.
"""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

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
    first_order_iterations : int
        Iterations evaluated with ``residual_startup`` before switching to
        ``residual`` (D36).
    growth_factor : float
        Minimum CFL growth per accepted iteration with decreasing residual.
    """

    cfl_start: float = 10.0
    cfl_max: float = 1.0e10
    cut_factor: float = 0.1
    reject_factor: float = 10.0
    cfl_min: float = 1.0e-6
    max_iterations: int = 1000
    residual_drop: float = 8.0
    first_order_iterations: int = 0
    growth_factor: float = 2.0


class SteadyResult(NamedTuple):
    """Result of :func:`solve_steady` and :func:`solve_explicit`.

    Attributes
    ----------
    u : ndarray, shape (n_cells, n_var)
        Final solution.
    converged : bool
        True if the residual dropped by ``residual_drop`` orders.
    history : dict of str to ndarray
        Per iteration: ``iteration``, ``residual`` (RMS norm after the
        iteration), ``residuals`` (RMS norm per variable), ``cfl``,
        ``accepted``.
    """

    u: np.ndarray
    converged: bool
    history: dict


def rms(r):
    """Root-mean-square norm of a residual array."""
    return float(np.sqrt(np.mean(np.square(r))))


def rms_per_variable(r):
    """RMS norm of each column of a residual array."""
    return np.sqrt(np.mean(np.square(r), axis=0))


def solve_steady(
    residual,
    jacobian,
    time_term,
    u0,
    settings=None,
    residual_startup=None,
    callback=None,
):
    """Drive ``residual(u)`` to zero by implicit pseudo-time stepping.

    Parameters
    ----------
    residual : callable
        ``residual(u) -> (n_cells, n_var)`` array.
    jacobian : callable
        ``jacobian(u) -> scipy.sparse`` approximate Jacobian dR/du.
    time_term : callable
        ``time_term(u, cfl) -> scipy.sparse`` pseudo-time matrix T.
    u0 : array, shape (n_cells, n_var)
        Initial solution.
    settings : PseudoTimeSettings, optional
    residual_startup : callable, optional
        Residual used for the first ``settings.first_order_iterations``
        iterations (e.g. first-order convective states).
    callback : callable, optional
        ``callback(iteration, u, residual_norms, cfl, accepted)`` after every
        iteration (e.g. logging).

    Returns
    -------
    SteadyResult
    """
    s = settings or PseudoTimeSettings()
    u = np.array(u0, dtype=float)

    def res(u, it):
        if residual_startup is not None and it <= s.first_order_iterations:
            return np.asarray(residual_startup(u))
        return np.asarray(residual(u))

    r = res(u, 1)
    norm = rms(r)
    norm0 = max(norm, np.finfo(float).tiny)
    target = norm0 * 10.0 ** (-s.residual_drop)
    cfl = s.cfl_start
    hist = {
        "iteration": [0],
        "residual": [norm],
        "residuals": [rms_per_variable(r)],
        "cfl": [cfl],
        "accepted": [True],
    }
    converged = norm <= target
    for it in range(1, s.max_iterations + 1):
        if converged or cfl < s.cfl_min:
            break
        if it == s.first_order_iterations + 1 and residual_startup is not None:
            r = res(u, it)  # switch to the full residual
            norm = rms(r)
        matrix = jacobian(u) + time_term(u, cfl)
        accepted = False
        try:
            du = solve_direct(matrix, -r.ravel()).reshape(u.shape)
            u_new = u + du
            r_new = res(u_new, it + 1)
            norm_new = rms(r_new)
            accepted = np.isfinite(norm_new) and norm_new <= s.reject_factor * norm
        except LinearSolverError:
            pass
        hist["iteration"].append(it)
        hist["cfl"].append(cfl)
        hist["accepted"].append(accepted)
        if accepted:
            ratio = norm / max(norm_new, np.finfo(float).tiny)
            if ratio > 1.0:
                ratio = max(ratio, s.growth_factor)
            cfl = min(s.cfl_max, cfl * ratio)
            u, r, norm = u_new, r_new, norm_new
            converged = norm <= target
        else:
            cfl *= s.cut_factor
        hist["residual"].append(norm)
        hist["residuals"].append(rms_per_variable(r))
        if callback is not None:
            callback(it, u, hist["residuals"][-1], hist["cfl"][-1], accepted)
    history = {k: np.asarray(v) for k, v in hist.items()}
    return SteadyResult(u, bool(converged), history)


def solve_explicit(
    residual,
    update,
    u0,
    cfl=0.8,
    max_iterations=10000,
    residual_drop=8.0,
    stages=(1.0 / 3.0, 0.5, 1.0),
    callback=None,
):
    """Explicit multi-stage pseudo-time stepping, reference mode (D25).

    Stage k: u_k = u_n - alpha_k * update(u_(k-1), R(u_(k-1)), cfl), where
    ``update`` returns dtau_i Gamma_i^-1 R_i / A_i, i.e. the explicit
    increment for CFL = cfl.

    Parameters
    ----------
    residual : callable
    update : callable
        ``update(u, r, cfl) -> (n_cells, n_var)``.
    u0 : array
    cfl : float
    max_iterations : int
    residual_drop : float
    stages : tuple of float
        Runge-Kutta stage coefficients.
    callback : callable, optional

    Returns
    -------
    SteadyResult
    """
    u = np.array(u0, dtype=float)
    r = np.asarray(residual(u))
    norm = rms(r)
    target = max(norm, np.finfo(float).tiny) * 10.0 ** (-residual_drop)
    hist = {
        "iteration": [0],
        "residual": [norm],
        "residuals": [rms_per_variable(r)],
        "cfl": [cfl],
        "accepted": [True],
    }
    converged = norm <= target
    for it in range(1, max_iterations + 1):
        if converged:
            break
        u_stage = u
        for alpha in stages:
            r_stage = r if alpha == stages[0] else np.asarray(residual(u_stage))
            u_stage = u - alpha * np.asarray(update(u_stage, r_stage, cfl))
        u = u_stage
        r = np.asarray(residual(u))
        norm = rms(r)
        if not np.isfinite(norm):
            break
        converged = norm <= target
        hist["iteration"].append(it)
        hist["residual"].append(norm)
        hist["residuals"].append(rms_per_variable(r))
        hist["cfl"].append(cfl)
        hist["accepted"].append(True)
        if callback is not None:
            callback(it, u, hist["residuals"][-1], cfl, True)
    history = {k: np.asarray(v) for k, v in hist.items()}
    return SteadyResult(u, bool(converged), history)
