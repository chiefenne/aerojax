# Changelog

## 0.1 – Stage 1: operators and implicit machinery (2026-10-14)

- Finite-volume geometry as JAX arrays (`aerojax.numerics.build_geometry`).
- Weighted least-squares gradients on the vertex-neighbour stencil with
  boundary-face values, weights 1/d^q (D22).
- MUSCL reconstruction and the Venkatakrishnan–Wang limiter (D23).
- Face gradients: α-damping with the damping term at the edge midpoint, and
  the edge-correction fallback (D24).
- Residual assembly from face fluxes; first-order Jacobian from AD of compact
  face fluxes, assembled as a sparse block matrix (D25).
- Direct sparse LU (D32) and implicit pseudo-time stepping with SER CFL
  control and rejection safeguards (D35).
- Model problem: steady scalar convection–diffusion with an exact exponential
  solution (`aerojax.physics.convection_diffusion`, `aerojax.cases`).
- Verification: observed order 1.9–2.1 on the two finest of four levels for
  quadrilateral, triangle, polygon and mixed meshes (regular and perturbed);
  convergence by 11 orders of magnitude in 17–29 iterations.
- Example `examples/stage1_convection_diffusion.py`.

## 0.0 – Stage 0: foundations (2026-10-13)

- Face-based mesh data structure for cell-centred finite volumes: arbitrary
  polygon cells, faces with owner and neighbour cells, outward unit normals,
  face lengths and centres, cell areas and centroids.
- Face-level boundary checks (D39 R1): every boundary face in exactly one
  named set, sets contain boundary faces only. Errors report face counts and
  coordinates.
- Rectangle mesh generator (D42): quadrilateral, triangle, polygon
  (centroid dual) and mixed cells, tanh clustering in y, random perturbation
  of interior nodes.
- Reader for 2D SU2 meshes (`.su2`, D43), compatible with PyAero's export.
- VTU writer with cell and point data, binary or ASCII (D45).
- Optional matplotlib mesh plot (D46).
- Unit tests and the example `examples/stage0_meshes.py`.
