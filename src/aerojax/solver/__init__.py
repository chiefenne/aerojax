"""Residual assembly, Jacobian, linear solvers, pseudo-time and dual time stepping."""

from aerojax.solver.dual_time import (
    DualTimeSettings,
    UnsteadyResult,
    bdf_coefficients,
    solve_unsteady,
)
from aerojax.solver.jacobian import (
    JacobianPattern,
    block_diagonal,
    first_order_jacobian,
    replace_rows,
)
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
    "DualTimeSettings",
    "LinearSettings",
    "LinearSolverError",
    "PseudoTimeSettings",
    "SteadyResult",
    "UnsteadyResult",
    "bdf_coefficients",
    "block_diagonal",
    "divergence",
    "factorize",
    "JacobianPattern",
    "first_order_jacobian",
    "replace_rows",
    "rms",
    "rms_per_variable",
    "solve_direct",
    "solve_explicit",
    "solve_matrix_free",
    "solve_steady",
    "solve_unsteady",
]
