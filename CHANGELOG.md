# Changelog

## Unreleased

- RANS log and history header: four residual columns (`log(R_nu)` added), so
  the CFL, NK, GMRES and coefficient columns line up again.

## 0.7 – Restarts, polars, validation on the PyAero mesh (2026-10-15)

- Restart files (`<name>.restart.npz`, `initial.restart`) and `aerojax polar`
  (angle sweep as a restart chain, merged `<name>.polar.csv`).
- Prescribed transition per angle of attack (`physics.transition.by_alpha`,
  interpolated to the case's angle, used by `aerojax polar`); `ramp` blends
  the SA production in over a length.
- Validation of the fully turbulent NACA 0012 polar on the PyAero y⁺ = 1
  mesh against Ladson (NASA TM 4074) and of the C_p at 10° against Gregory
  & O'Reilly; packaged data `naca0012_ladson_re6e6_tripped.csv`,
  `naca0012_gregory_cp.csv`, `naca0012_abbott_re6e6_smooth.csv`; example
  `examples/stage6_pyaero_validation.py`; prescribed-transition case
  `examples/cases/naca0012_pyaero_transition.toml`; `docs/pyaero.md`
  sections on polars, transition and validation.

## 0.6 – Stage 6: NACA 0012 (2026-10-15)

- Airfoil O-grid generator (`aerojax.mesh.airfoil`): TMR NACA 0012
  definition, cosine surface spacing with a small blunt trailing edge
  (0.14 % chord: the fan of nearly flat cells of a sharp trailing edge is
  unstable at high Reynolds numbers), marching along smoothed normals
  blended into the radial direction, farfield at 500 chords, fold check.
- Classic Venkatakrishnan limiter with the local threshold ε² = (K h)³
  (`limiter = "venkatakrishnan"`): the global-range threshold of the
  Venkatakrishnan–Wang limiter leaves the thin wall cells of a high-Re
  boundary layer unlimited, and the reconstruction diverges on the curved
  nose. Newton–Krylov switch at a fixed iteration
  (`newton_krylov.switch_at_iteration`) in addition to the residual drop.
- Case R2 (`aerojax.cases.naca0012`): C_L, C_D (pressure/viscous), C_M, C_p
  and C_f at Re 6e6 vs the TMR SA values. 256 x 128 O-grid (33k cells):
  C_L -0.008 / 1.076 / 1.536 at α = 0 / 10 / 15° (TMR 0 / 1.0909 / 1.5461),
  viscous drag close to TMR, pressure drag too high on the coarse O-grid
  wake (see docs/requirements.md D40). PyAero meshes (boundary faces
  `airfoil`, `farfield`) can be used directly.
- Numerics standard for subsonic RANS (D54): limiter released after the
  start-up (`limiter_freeze.release`), per-equation convergence measure
  (each residual relative to its own peak, worst equation decides),
  backtracking line search, Eisenstat–Walker forcing and an adaptive CFL
  cap in the Newton phase. On the PyAero y⁺ = 1 mesh of the NACA 0012
  (100k cells, Re 6e6, α = 0) this converges 13 orders in 44 iterations
  with quadratic Newton convergence; C_D = 0.00828 vs TMR 0.00819, and at
  α = 10° with the vortex correction C_L = 1.0807 vs 1.0909 (TOML case
  `examples/cases/naca0012_pyaero_rans.toml`).
- Point-vortex farfield correction (P10, `reference.vortex_correction`):
  freestream plus the lift-induced velocity of a vortex at the moment
  centre, pressure from Bernoulli; removes the farfield error of the lift on
  a 10-chord domain (Euler, 10°). Several farfield boundaries with the same
  freestream count as one (PyAero's inlet/outlet/top/bottom).
- PyAero mesh of the NACA 0012 (`examples/meshes/naca0012_pyaero_50k.su2`,
  C-type, 52k cells) read and run as is; its wall spacing suits laminar
  flow (y⁺ ≈ 900 at Re 6e6, see docs/requirements.md).
- Documentation: `docs/pyaero.md`, the PyAero-to-aerojax workflow (first
  layer from y⁺, farfield distance, case file, log, limits).
- Examples: `examples/stage6_naca0012.py`, TOML cases
  `examples/cases/naca0012_rans.toml` and `naca0012_pyaero_euler.toml`.

## 0.5 – Stage 5: RANS with the SA-neg model (2026-10-14)

- Spalart–Allmaras, negative variant (NASA TMR SA-neg, with f_t2 and the
  modified S̃), as fourth variable ν̃ of the coupled state (D11, D13):
  upwind convection with the face-average velocity, α-damping diffusion,
  cell sources with AD-safe formulation; eddy viscosity in the mean-flow
  viscous flux; ν̃ = 0 at walls, prescribed at inflow faces.
- Wall distance of every cell centre (KD-tree candidates, exact
  point-segment distance) with the nearest wall face.
- Segregated/coupled first-order Jacobian (D34), Newton–Krylov coupled in
  both; `linear_solver.ordering` option. Default segregated confirmed on the
  flat plate (coupled: 2× the iterations, 10× the LU time).
- Prescribed transition (D15): `physics.transition.laminar_region` by
  boundary, side and x/y bounds; SA production switched off there.
- Turbulence inputs (D33): `turbulence` tables in `[initial]` and at inflow
  boundaries in the three forms (ν_t/ν + Tu, k + ℓ, ν̃/ν), equivalent values
  reported in the log, warning outside the TMR range ν̃/ν = 3–5; y⁺ in the
  wall CSV, ν̃ and ν_t/ν in the VTU output.
- Verification R1: turbulent flat plate on the TMR domain, C_f(0.97) +1.3 %
  and C_D +1.8 % vs the TMR SA values on 7.7k cells (+1.2 / +1.5 % on 23k),
  u⁺(y⁺) follows the log law. GMRES-ILU verified on the 23k-cell plate.
- Examples: `examples/stage5_rans.py`, TOML case
  `examples/cases/flat_plate_rans.toml`.

## 0.4 – Stage 4: unsteady flow (2026-10-14)

- Dual time stepping (D9, D37): BDF2 (BDF1 for the first step, or
  throughout) with the steady solver as inner iteration — extrapolated start
  2uⁿ − uⁿ⁻¹, large starting pseudo-CFL, Newton–Krylov from the first inner
  iteration, inner residual drop relative to R*(uⁿ). 1–2 inner iterations per
  step in smooth flow. `IncompressibleProblem.solve_unsteady`,
  `aerojax.solver.solve_unsteady`, `DualTimeSettings`.
- Case file: `[time] mode = "unsteady"`, `scheme`, `time_step`, `end_time`,
  `inner_iterations`, `inner_residual_drop`, `inner_cfl`.
- Output (D45): unsteady log and history layouts (one line per time step with
  inner iterations and drop; inner lines with `verbosity = "detailed"`), field
  series `<name>_<step>.vtu` collected in a `.pvd` file.
- Verification: V3 Taylor–Green vortex (doubly periodic, exact pressure pin):
  observed temporal orders 2.02/2.08 (BDF2) and 0.99/1.00 (BDF1), velocity
  error 0.5 % on 32²; impulsively started Couette flow vs the series solution
  from TOML. Benchmark B2 DFG 2D-2 (Re 100): periodic shedding with St
  0.278, cD,max 3.20, cL,max 1.06, Δp 2.44 on 4.4k cells (reference 0.295–0.305,
  3.22–3.24, 0.99–1.01, 2.46–2.50).
- Examples: `examples/stage4_unsteady.py`, TOML case
  `examples/cases/couette_unsteady.toml`.

## 0.3 – Stage 3: Newton–Krylov, boundary conditions, Euler and benchmark cases (2026-10-14)

- Newton–Krylov phase (D25): once the residual has dropped by
  `switch_at_residual_drop` orders, each iteration solves the exact
  linearisation (Jacobian-vector products by `jax.jvp` of the full
  second-order residual) with matrix-free GMRES, right-preconditioned by the
  factorized first-order system. Cavity Re 1000 on 64²: 123 → 20 iterations.
  Optional reuse of the preconditioner (`refactor_interval`).
- Linear solver options (D32): `direct` (SuperLU with the MMD_AT_PLUS_A
  ordering and threshold pivoting, 5–10× less fill than COLAMD on the
  coupled system) or `gmres-ilu` (GMRES with an incomplete LU, options for
  drop tolerance and fill).
- Limiter freezing (`numerics.limiter_freeze`) after a number of iterations
  or a residual drop, so limited cases converge to machine precision.
- Boundary conditions (D38): total-pressure inlet and transpiration wall
  (blowing/suction); the full set of stage 3 is now available from TOML.
- Forces and wall loads (D45): C_L, C_D, C_M (with `reference.angle_of_attack`
  and `moment_center`) per iteration in the log and history, pressure and
  viscous drag in the summary, and a wall CSV (x, y, p, Cp, τ_x, τ_y, Cf) per
  force boundary (`output.force_boundaries`).
- Mesh generators (D42): O-grid `annulus`, block-structured
  `cylinder_in_channel` (DFG geometry), `split_boundary`.
- Verification and benchmarks: V4 inviscid cylinder (Cp vs potential flow,
  error halves under refinement, zero lift), B1 Blasius flat plate at
  Re_L = 1e5 (Cf within 1 %, profile within 0.006), B2 DFG benchmark 2D-1
  (cD within 0.3 %, Δp within 0.5 % of the reference on 13k cells).
- First profiling pass (D52): per-iteration cost at 37k cells dominated by
  the sparse LU (≈1 s) and the Jacobian assembly (0.2 s); residual 33 ms,
  Jacobian-vector product 46 ms.
- Examples: `examples/stage3_benchmarks.py`, TOML case
  `examples/cases/cylinder_euler.toml`.

## 0.2 – Stage 2: steady laminar incompressible flow (2026-10-14)

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
