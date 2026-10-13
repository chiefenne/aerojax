"""Linear solvers for the implicit iterations (D32).

Stage 1 provides the direct sparse LU (SciPy SuperLU); GMRES with ILU follows
in stage 3.
"""

import numpy as np
import scipy.sparse.linalg as spla


class LinearSolverError(RuntimeError):
    """Raised when the linear system cannot be solved."""


def solve_direct(matrix, rhs):
    """Solve ``matrix @ x = rhs`` with a sparse LU factorization.

    Parameters
    ----------
    matrix : scipy.sparse matrix, shape (n, n)
    rhs : ndarray, shape (n,)

    Returns
    -------
    ndarray, shape (n,)

    Raises
    ------
    LinearSolverError
        If the matrix is singular or the solution is not finite.
    """
    try:
        x = spla.splu(matrix.tocsc()).solve(rhs)
    except RuntimeError as exc:  # "Factor is exactly singular"
        raise LinearSolverError(str(exc)) from exc
    if not np.all(np.isfinite(x)):
        raise LinearSolverError("non-finite solution")
    return x
