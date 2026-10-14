"""Implicit pseudo-time stepping with SER CFL control (D25, D35, D36).

Each iteration solves

    (T(u, CFL) + J(u)) du = -R(u),

with the pseudo-time matrix T and a Jacobian J. For a scalar equation
T = diag(lambda_i / CFL) with the spectral radius lambda_i of cell i (the
pseudo-time step is dtau_i = CFL A_i / lambda_i); for the preconditioned flow
equations T holds the blocks Gamma_i lambda_i / CFL.

Two phases (D25, as SU2's ``NEWTON_KRYLOV``):

1. Start-up with the approximate first-order Jacobian J1, solved by the
   sparse linear solver (direct LU or GMRES-ILU).
2. Newton-Krylov once the residual has dropped by ``nk_switch_residual_drop``
   orders: the exact Jacobian of the full residual is applied matrix-free
   through ``jvp`` (JAX forward mode), solved by GMRES with the factorization
   of T + J1 as preconditioner.

The CFL number follows switched evolution relaxation (Mulder & van Leer
1985):

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

from aerojax.solver.linear import (
    LinearSettings,
    LinearSolverError,
    factorize,
    solve_matrix_free,
)


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
    newton_krylov : bool
        Switch to Newton-Krylov (needs ``jvp``).
    nk_switch_residual_drop : float
        Residual drop (orders of magnitude) at which the switch happens.
    nk_linear_tolerance : float
        Relative tolerance of the GMRES solve in the Newton-Krylov phase.
    nk_restart : int
        GMRES restart length in the Newton-Krylov phase.
    nk_max_krylov : int
        Maximum GMRES iterations per Newton step.
    nk_refactor_interval : int
        In the Newton-Krylov phase, rebuild the preconditioner (first-order
        Jacobian plus pseudo-time term, factorized) only every n-th step and
        reuse it otherwise; the factorization dominates the cost on large
        meshes. 1 = every step.
    linear : LinearSettings
        Sparse linear solver of phase 1 and preconditioner of phase 2.
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
    newton_krylov: bool = True
    nk_switch_residual_drop: float = 2.0
    nk_linear_tolerance: float = 1.0e-2
    nk_restart: int = 50
    nk_max_krylov: int = 200
    nk_refactor_interval: int = 1
    linear: LinearSettings = LinearSettings()


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
        ``accepted``, ``newton_krylov`` (phase flag), ``linear_iterations``
        (GMRES iterations of the step, 0 for a direct solve).
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
    jvp=None,
    reference_norm=None,
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
        iteration (e.g. logging). ``callback.info`` may be set beforehand to
        a dict that receives ``newton_krylov`` and ``linear_iterations``.
    jvp : callable, optional
        ``jvp(u, du) -> dR/du(u) @ du`` of the full residual; enables the
        Newton-Krylov phase.
    reference_norm : float, optional
        Residual norm that ``residual_drop`` and the Newton-Krylov switch
        refer to; default the norm of the initial residual.

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
    norm0 = max(
        norm if reference_norm is None else reference_norm, np.finfo(float).tiny
    )
    target = norm0 * 10.0 ** (-s.residual_drop)
    nk_threshold = norm0 * 10.0 ** (-s.nk_switch_residual_drop)
    cfl = s.cfl_start
    hist = {
        "iteration": [0],
        "residual": [norm],
        "residuals": [rms_per_variable(r)],
        "cfl": [cfl],
        "accepted": [True],
        "newton_krylov": [False],
        "linear_iterations": [0],
    }
    converged = norm <= target
    if callback is not None:
        callback(0, u, hist["residuals"][0], cfl, True)
    lin, nk_steps_since_factor = None, 0
    for it in range(1, s.max_iterations + 1):
        if converged or cfl < s.cfl_min:
            break
        startup = residual_startup is not None and it <= s.first_order_iterations
        if it == s.first_order_iterations + 1 and residual_startup is not None:
            r = res(u, it)  # switch to the full residual
            norm = rms(r)
        use_nk = (
            s.newton_krylov and jvp is not None and not startup and norm <= nk_threshold
        )
        t_matrix = time_term(u, cfl)
        accepted, lin_its = False, 0
        try:
            reuse = (
                use_nk
                and lin is not None
                and nk_steps_since_factor < s.nk_refactor_interval
            )
            if not reuse:
                lin = factorize(jacobian(u) + t_matrix, s.linear)
                nk_steps_since_factor = 0
            if use_nk:
                nk_steps_since_factor += 1
                du, lin_its = solve_matrix_free(
                    _newton_operator(t_matrix, jvp, u),
                    -r.ravel(),
                    lin.precondition,
                    tolerance=s.nk_linear_tolerance,
                    restart=s.nk_restart,
                    max_iterations=s.nk_max_krylov,
                )
            else:
                du = lin.solve(-r.ravel())
                lin_its = lin.iterations
            u_new = u + du.reshape(u.shape)
            r_new = res(u_new, it + 1)
            norm_new = rms(r_new)
            accepted = np.isfinite(norm_new) and norm_new <= s.reject_factor * norm
        except LinearSolverError:
            pass
        hist["iteration"].append(it)
        hist["cfl"].append(cfl)
        hist["accepted"].append(accepted)
        hist["newton_krylov"].append(use_nk)
        hist["linear_iterations"].append(lin_its)
        if accepted:
            ratio = norm / max(norm_new, np.finfo(float).tiny)
            if ratio > 1.0:
                ratio = max(ratio, s.growth_factor)
            cfl = min(s.cfl_max, cfl * ratio)
            u, r, norm = u_new, r_new, norm_new
            converged = norm <= target
        else:
            cfl *= s.cut_factor
            lin = None  # rebuild the preconditioner after a rejection
        hist["residual"].append(norm)
        hist["residuals"].append(rms_per_variable(r))
        if callback is not None:
            info = getattr(callback, "info", None)
            if info is not None:
                info["newton_krylov"] = use_nk
                info["linear_iterations"] = lin_its
            callback(it, u, hist["residuals"][-1], hist["cfl"][-1], accepted)
    history = {k: np.asarray(v) for k, v in hist.items()}
    return SteadyResult(u, bool(converged), history)


def _newton_operator(t_matrix, jvp, u):
    """Return v -> (T + J(u)) v for the matrix-free Newton-Krylov solve."""

    def apply(v):
        return t_matrix @ v + np.asarray(jvp(u, v.reshape(u.shape))).ravel()

    return apply


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
        "newton_krylov": [False],
        "linear_iterations": [0],
    }
    converged = norm <= target
    if callback is not None:
        callback(0, u, hist["residuals"][0], cfl, True)
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
        hist["newton_krylov"].append(False)
        hist["linear_iterations"].append(0)
        if callback is not None:
            callback(it, u, hist["residuals"][-1], cfl, True)
    history = {k: np.asarray(v) for k, v in hist.items()}
    return SteadyResult(u, bool(converged), history)
