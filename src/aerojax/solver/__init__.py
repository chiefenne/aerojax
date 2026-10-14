"""Residual assembly, Jacobian, linear solvers and pseudo-time stepping."""

from aerojax.solver.jacobian import block_diagonal, first_order_jacobian, replace_rows
from aerojax.solver.linear import (
    LinearSettings,
    LinearSolverError,
    factorize,
    solve_direct,
    solve_matrix_free,
)
from aerojax.solver.pseudo_time import (
    PseudoTimeSettings,
    SteadyResult,
    rms,
    rms_per_variable,
    solve_explicit,
    solve_steady,
)
from aerojax.solver.residual import divergence

__all__ = [
    "LinearSettings",
    "LinearSolverError",
    "PseudoTimeSettings",
    "SteadyResult",
    "block_diagonal",
    "divergence",
    "factorize",
    "first_order_jacobian",
    "replace_rows",
    "rms",
    "rms_per_variable",
    "solve_direct",
    "solve_explicit",
    "solve_matrix_free",
    "solve_steady",
]
