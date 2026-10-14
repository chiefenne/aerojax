"""Mesh data structure, generators and readers."""

from aerojax.mesh.cylinder import annulus, cylinder_in_channel, split_boundary
from aerojax.mesh.generators import geometric_growth, rectangle, structured
from aerojax.mesh.mesh import Mesh, MeshError
from aerojax.mesh.su2 import read_su2, write_su2

__all__ = [
    "Mesh",
    "MeshError",
    "annulus",
    "cylinder_in_channel",
    "geometric_growth",
    "read_su2",
    "rectangle",
    "split_boundary",
    "structured",
    "write_su2",
]
