"""Approximate (first-order) Jacobian by automatic differentiation (D25).

The Jacobian is assembled from compact face fluxes that depend only on the
two cells next to a face (interior faces) or on the owner cell (boundary
faces). JAX differentiates each face flux; the blocks are assembled into a
SciPy sparse matrix with block size n_var, unknowns ordered cell by cell
(index ``i * n_var + v``), matching ``u.ravel()``.
"""

from functools import partial

import jax
import numpy as np
import scipy.sparse as sp


@partial(jax.jit, static_argnums=(1, 2))
def _face_blocks(geom, interior_flux, boundary_flux, u, u_b):
    """Flux derivatives per face, scaled by the face length."""
    o, n, b = geom.f_owner, geom.f_neigh, geom.b_owner
    d_inner = jax.vmap(jax.jacfwd(interior_flux, argnums=(0, 1)))
    d_o, d_n = d_inner(u[o], u[n], geom.f_normal, geom.f_delta)
    d_bnd = jax.vmap(jax.jacfwd(boundary_flux, argnums=0))
    d_b = d_bnd(u[b], u_b, geom.b_normal, geom.b_delta)
    fl = geom.f_length[:, None, None]
    return d_o * fl, d_n * fl, d_b * geom.b_length[:, None, None]


def first_order_jacobian(geom, interior_flux, boundary_flux, u, u_b):
    """Assemble dR1/du of the compact first-order residual.

    Parameters
    ----------
    geom : FVGeometry
    interior_flux : callable
        ``f(u_o, u_n, normal, delta) -> (n_var,)``: flux density through an
        interior face (owner to neighbour); ``delta`` = owner-to-neighbour
        centroid vector. Must be a hashable function (it is a jit static
        argument), e.g. a module-level function or a ``functools.partial``.
    boundary_flux : callable
        ``f(u_o, u_b, normal, delta) -> (n_var,)``: outward flux density
        through a boundary face; ``delta`` = owner centroid to face centre.
    u : array, shape (n_cells, n_var)
    u_b : array, shape (n_bfaces, n_var)
        Boundary-face values (held fixed in the derivative).

    Returns
    -------
    scipy.sparse.csr_matrix, shape (n_cells * n_var, n_cells * n_var)
    """
    n_cells, n_var = u.shape
    d_o, d_n, d_b = (
        np.asarray(a) for a in _face_blocks(geom, interior_flux, boundary_flux, u, u_b)
    )
    o, n, b = (np.asarray(a) for a in (geom.f_owner, geom.f_neigh, geom.b_owner))
    block_rows = np.concatenate((o, o, n, n, b))
    block_cols = np.concatenate((o, n, o, n, b))
    blocks = np.concatenate((d_o, d_n, -d_o, -d_n, d_b))

    k = np.arange(n_var)
    rows = (block_rows[:, None, None] * n_var + k[None, :, None]).repeat(n_var, 2)
    cols = (block_cols[:, None, None] * n_var + k[None, None, :]).repeat(n_var, 1)
    size = n_cells * n_var
    return sp.coo_matrix(
        (blocks.ravel(), (rows.ravel(), cols.ravel())), shape=(size, size)
    ).tocsr()
