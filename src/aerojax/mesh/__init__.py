"""Mesh data structure, generators and readers."""

from aerojax.mesh.cylinder import annulus, cylinder_in_channel, split_boundary
from aerojax.mesh.generators import rectangle
from aerojax.mesh.mesh import Mesh, MeshError
from aerojax.mesh.su2 import read_su2, write_su2

__all__ = [
    "Mesh",
    "MeshError",
    "annulus",
    "cylinder_in_channel",
    "read_su2",
    "rectangle",
    "split_boundary",
    "write_su2",
]
