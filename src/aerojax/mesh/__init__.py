"""Mesh data structure, generators and readers."""

from aerojax.mesh.generators import rectangle
from aerojax.mesh.mesh import Mesh, MeshError
from aerojax.mesh.su2 import read_su2, write_su2

__all__ = ["Mesh", "MeshError", "read_su2", "rectangle", "write_su2"]
