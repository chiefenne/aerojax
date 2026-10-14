"""MUSCL face reconstruction and the Venkatakrishnan-Wang limiter (D21, D23).

Face states are extrapolated from the cell centroid to the face centre:

    u_face = u_i + psi_i * grad(u)_i . (x_f - x_i)

with the limiter psi_i in [0, 1] per cell and variable.
"""

import jax
import jax.numpy as jnp


@jax.jit
def venkatakrishnan_wang(geom, u, u_b, grad, threshold=0.05):
    """Venkatakrishnan limiter with Wang's scale-independent threshold.

    The smoothness threshold is eps = threshold * (max(u) - min(u)), with the
    global range of each variable (Wang's suggested threshold range is
    0.01-0.20). Minimum and maximum per cell are taken over the cell itself and
    its face neighbours, including boundary-face values.

    For threshold = 0 the reconstructed face values stay within these bounds;
    for threshold > 0 they may exceed them by O(eps), which is what keeps the
    limiter inactive in smooth regions.

    Parameters
    ----------
    geom : FVGeometry
    u : jax array, shape (n_cells, n_var)
    u_b : jax array, shape (n_bfaces, n_var)
    grad : jax array, shape (n_cells, n_var, 2)
    threshold : float
        Wang's epsilon' (relative to the global range).

    Returns
    -------
    jax array, shape (n_cells, n_var)
        Limiter values psi.
    """
    n = u.shape[0]
    o, nb, b = geom.f_owner, geom.f_neigh, geom.b_owner
    u_max = jnp.maximum(
        jnp.maximum(u, jax.ops.segment_max(u[nb], o, num_segments=n)),
        jnp.maximum(
            jax.ops.segment_max(u[o], nb, num_segments=n),
            jax.ops.segment_max(u_b, b, num_segments=n),
        ),
    )
    u_min = jnp.minimum(
        jnp.minimum(u, jax.ops.segment_min(u[nb], o, num_segments=n)),
        jnp.minimum(
            jax.ops.segment_min(u[o], nb, num_segments=n),
            jax.ops.segment_min(u_b, b, num_segments=n),
        ),
    )
    all_values = jnp.concatenate((u, u_b))
    eps2 = (threshold * (all_values.max(axis=0) - all_values.min(axis=0))) ** 2

    def face_limiter(cells, dx):
        d2 = jnp.einsum("fvk,fk->fv", grad[cells], dx)
        d1 = jnp.where(d2 > 0.0, u_max[cells] - u[cells], u_min[cells] - u[cells])
        num = d1**2 + eps2 + 2.0 * d1 * d2
        den = d1**2 + 2.0 * d2**2 + d1 * d2 + eps2
        return jnp.where(den > 0.0, num / jnp.where(den > 0.0, den, 1.0), 1.0)

    psi = jnp.ones_like(u)
    psi = jnp.minimum(psi, jax.ops.segment_min(face_limiter(o, geom.f_dx_owner), o, n))
    psi = jnp.minimum(
        psi, jax.ops.segment_min(face_limiter(nb, geom.f_dx_neigh), nb, n)
    )
    psi = jnp.minimum(psi, jax.ops.segment_min(face_limiter(b, geom.b_delta), b, n))
    return jnp.clip(psi, 0.0, 1.0)


@jax.jit
def reconstruct(geom, u, grad, psi):
    """Second-order (MUSCL) face states.

    Parameters
    ----------
    geom : FVGeometry
    u : jax array, shape (n_cells, n_var)
    grad : jax array, shape (n_cells, n_var, 2)
    psi : jax array, shape (n_cells, n_var)
        Limiter values (ones for unlimited reconstruction).

    Returns
    -------
    u_left, u_right : jax arrays, shape (n_ifaces, n_var)
        Owner-side and neighbour-side states at interior faces.
    u_inner : jax array, shape (n_bfaces, n_var)
        Interior-side state at boundary faces.
    """

    def extrapolate(cells, dx):
        return u[cells] + psi[cells] * jnp.einsum("fvk,fk->fv", grad[cells], dx)

    return (
        extrapolate(geom.f_owner, geom.f_dx_owner),
        extrapolate(geom.f_neigh, geom.f_dx_neigh),
        extrapolate(geom.b_owner, geom.b_delta),
    )
