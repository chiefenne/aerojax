"""Finite-volume geometry as JAX arrays.

:func:`build_geometry` converts a :class:`~aerojax.mesh.Mesh` into the arrays
used by the jitted numerics: interior faces, boundary faces (grouped by
boundary name) and the least-squares gradient stencil. All preprocessing is
done once in NumPy.
"""

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from aerojax.numerics.gradients import lsq_stencil


class FVGeometry(NamedTuple):
    """Geometry arrays for jitted finite-volume operators (a JAX pytree).

    Interior faces are prefixed ``f_``, boundary faces ``b_``. Face normals
    point from owner to neighbour, and out of the domain on boundary faces.
    ``f_delta`` is the vector from the owner to the neighbour centroid,
    ``b_delta`` the vector from the owner centroid to the boundary-face centre.
    Boundary faces are ordered by boundary name (see :class:`Discretization`).
    """

    area: jnp.ndarray
    centroid: jnp.ndarray
    f_owner: jnp.ndarray
    f_neigh: jnp.ndarray
    f_normal: jnp.ndarray
    f_length: jnp.ndarray
    f_centre: jnp.ndarray
    f_delta: jnp.ndarray
    b_owner: jnp.ndarray
    b_normal: jnp.ndarray
    b_length: jnp.ndarray
    b_centre: jnp.ndarray
    b_delta: jnp.ndarray
    lsq_cell: jnp.ndarray
    lsq_src: jnp.ndarray
    lsq_coeff: jnp.ndarray


class Discretization(NamedTuple):
    """Mesh, its JAX geometry, and the position of each boundary in ``b_*``.

    Attributes
    ----------
    mesh : Mesh
        The source mesh.
    geom : FVGeometry
        Arrays for the jitted operators.
    boundary_slices : dict of str to slice
        Range of each named set of boundary faces in the ``b_*`` arrays.
    boundary_face_ids : ndarray
        Mesh face index of every boundary face, in ``b_*`` order.
    """

    mesh: object
    geom: FVGeometry
    boundary_slices: dict
    boundary_face_ids: np.ndarray


def build_geometry(mesh, weight_exponent=1.0):
    """Build the finite-volume geometry of a mesh.

    Parameters
    ----------
    mesh : Mesh
        The mesh.
    weight_exponent : float
        Exponent q of the least-squares weights 1/d^q (see
        :func:`~aerojax.numerics.gradients.lsq_stencil`).

    Returns
    -------
    Discretization
    """
    interior = np.flatnonzero(mesh.neighbour >= 0)
    slices, ids, start = {}, [], 0
    for name, faces in mesh.boundary_faces.items():
        slices[name] = slice(start, start + faces.size)
        ids.append(faces)
        start += faces.size
    bnd = np.concatenate(ids) if ids else np.zeros(0, dtype=np.int64)

    cell, src, coeff = lsq_stencil(mesh, bnd, weight_exponent)
    o, n, ob = mesh.owner[interior], mesh.neighbour[interior], mesh.owner[bnd]
    c = mesh.cell_centroid
    geom = FVGeometry(
        area=jnp.asarray(mesh.cell_area),
        centroid=jnp.asarray(mesh.cell_centroid),
        f_owner=jnp.asarray(o),
        f_neigh=jnp.asarray(n),
        f_normal=jnp.asarray(mesh.face_normal[interior]),
        f_length=jnp.asarray(mesh.face_length[interior]),
        f_centre=jnp.asarray(mesh.face_centre[interior]),
        f_delta=jnp.asarray(c[n] - c[o]),
        b_owner=jnp.asarray(ob),
        b_normal=jnp.asarray(mesh.face_normal[bnd]),
        b_length=jnp.asarray(mesh.face_length[bnd]),
        b_centre=jnp.asarray(mesh.face_centre[bnd]),
        b_delta=jnp.asarray(mesh.face_centre[bnd] - c[ob]),
        lsq_cell=jnp.asarray(cell),
        lsq_src=jnp.asarray(src),
        lsq_coeff=jnp.asarray(coeff),
    )
    return Discretization(mesh, geom, slices, bnd)
