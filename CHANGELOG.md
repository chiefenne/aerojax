# Changelog

## 0.2 – Stage 2: steady laminar incompressible flow (not yet tagged)

- Incompressible Navier–Stokes in the SU2 INC formulation (D5, D8): working
  variables (p, u, v), preconditioned pseudo-time with β² = 4.1·max|v|²,
  FDS convective flux (D21) with the exact |Γ⁻¹A| (SU2's closed form has
  twice the tangential dissipation), viscous flux with α-damping gradients
  (D24), body force, Euler mode.
- Boundary conditions (D38): wall (moving), symmetry/slip wall, velocity
  inlet, pressure outlet, farfield; periodic pairs as translated interior
  faces (D39 R4); pressure pin without pressure boundary (R2); mass-balance
  check (R3); walls as slip walls in Euler mode (R5).
- Solver: block pseudo-time term, SER with minimum CFL growth factor (D35),
  first-order start (D36), explicit Runge–Kutta reference mode (D25).
- Python API: `aerojax.driver.IncompressibleProblem` with function-valued
  boundary values (D41).
- TOML case files (D27–D33): schema with defaults, validation with
  suggestions, resolved configuration with comments, dimensional input and
  internal non-dimensionalisation (D51), Reynolds-number input (D31).
- Output: `.log` (console and file), history CSV, VTU fields (D45).
- CLI: `aerojax run case.toml`, `aerojax check case.toml`.
- Mesh: `.su2` writer, x-clustering in the rectangle generator.
- Verification: V1 Couette to round-off (quad, triangle, mixed), V2
  Poiseuille order 1.9–2.1 on four mesh types, B3 cavity vs Ghia et al.
  (Re 100 and 1000), explicit = implicit solution.
- Examples: `examples/stage2_laminar.py`, TOML cases in `examples/cases/`.

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
