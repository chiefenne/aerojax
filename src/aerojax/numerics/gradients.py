"""Weighted least-squares cell gradients (requirement D22).

The stencil of a cell consists of all cells sharing at least one node with it
(vertex neighbours) and all boundary faces sharing a node with it; boundary
faces enter with their face-centre values. The gradient of cell i minimises

    sum_k ( w_k * (phi_k - phi_i - grad(phi)_i . d_k) )^2,   w_k = |d_k|^(-q),

with d_k the vector from the cell centroid to the stencil point. This is the
weighting convention of Syrakos et al. (2017): q = 0 unweighted, q = 1 inverse
distance (as SU2's weighted least squares), q = 3/2 their recommendation for
second-order accuracy at boundary cells.
"""

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp


def lsq_stencil(mesh, boundary_faces, weight_exponent=1.0):
    """Precompute stencil and coefficients of the least-squares gradient.

    Parameters
    ----------
    mesh : Mesh
        The mesh.
    boundary_faces : ndarray
        Mesh indices of the boundary faces, in the order in which their values
        are passed to :func:`lsq_gradient`.
    weight_exponent : float
        Exponent q of the weights |d|^(-q).

    Returns
    -------
    cell : ndarray, shape (n_entries,)
        Cell of each stencil entry.
    src : ndarray, shape (n_entries,)
        Source index: < n_cells for cells, n_cells + j for boundary face j.
    coeff : ndarray, shape (n_entries, 2)
        Coefficients c such that grad_i = sum c * (phi_src - phi_i).
    """
    n_cells, n_nodes = mesh.n_cells, mesh.n_nodes
    rows = np.repeat(np.arange(n_cells), mesh.cell_n_nodes)
    incidence = sp.csr_matrix(
        (np.ones(rows.size), (rows, mesh.cell_nodes)), shape=(n_cells, n_nodes)
    )
    nb = len(boundary_faces)
    b_rows = np.repeat(np.arange(nb), 2)
    b_cols = mesh.face_nodes[boundary_faces].ravel()
    b_inc = sp.csr_matrix((np.ones(2 * nb), (b_rows, b_cols)), shape=(nb, n_nodes))
    cell_cell = (incidence @ incidence.T).tocoo()
    keep = cell_cell.row != cell_cell.col
    cell_bnd = (incidence @ b_inc.T).tocoo()

    cell = np.concatenate((cell_cell.row[keep], cell_bnd.row))
    src = np.concatenate((cell_cell.col[keep], n_cells + cell_bnd.col))
    points = np.vstack((mesh.cell_centroid, mesh.face_centre[boundary_faces]))
    order = np.lexsort((src, cell))
    cell, src = cell[order], src[order]

    d = points[src] - mesh.cell_centroid[cell]
    w2 = np.hypot(d[:, 0], d[:, 1]) ** (-2.0 * weight_exponent)
    m11 = np.bincount(cell, w2 * d[:, 0] ** 2, n_cells)
    m12 = np.bincount(cell, w2 * d[:, 0] * d[:, 1], n_cells)
    m22 = np.bincount(cell, w2 * d[:, 1] ** 2, n_cells)
    det = m11 * m22 - m12**2
    if np.any(det <= 1e-14 * (m11 * m22)):
        raise ValueError("degenerate least-squares stencil (collinear points)")
    inv11, inv12, inv22 = m22 / det, -m12 / det, m11 / det
    coeff = np.column_stack(
        (
            w2 * (inv11[cell] * d[:, 0] + inv12[cell] * d[:, 1]),
            w2 * (inv12[cell] * d[:, 0] + inv22[cell] * d[:, 1]),
        )
    )
    return cell, src, coeff


@jax.jit
def lsq_gradient(geom, u, u_b):
    """Least-squares gradients of cell values.

    Parameters
    ----------
    geom : FVGeometry
        Geometry with the precomputed stencil.
    u : jax array, shape (n_cells, n_var)
        Cell values.
    u_b : jax array, shape (n_bfaces, n_var)
        Boundary-face values, in boundary-face order.

    Returns
    -------
    jax array, shape (n_cells, n_var, 2)
    """
    values = jnp.concatenate((u, u_b))[geom.lsq_src]
    diff = values - u[geom.lsq_cell]
    contrib = diff[:, :, None] * geom.lsq_coeff[:, None, :]
    return jax.ops.segment_sum(contrib, geom.lsq_cell, num_segments=u.shape[0])
