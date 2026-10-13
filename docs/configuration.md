# aerojax – Case configuration (draft)

**Status:** draft for review. Key names and defaults are proposals. Entries
marked *open: Ox* depend on open requirement steps in
[requirements.md](requirements.md).


## 1. Principles

- **Format:** TOML 1.0, read with the standard library (`tomllib`).
- **Names:** literature terminology and established acronyms (D26). SU2
  option names are not copied.
  - Keys are `snake_case`.
  - String values are lowercase with hyphens, e.g. `"sa-neg"`,
    `"venkatakrishnan-wang"`.
  - Acronyms keep their usual meaning: `cfl`, `muscl`, `fds`, `bdf2`,
    `gmres`, `ilu`.
- **Units:** SI and angles in degrees. Units are documented and shown in the
  example comments, not encoded in key names. *Open: O12
  (non-dimensionalization).*
- **Structure:** initial conditions and boundary conditions are separate
  sections. The farfield is a boundary condition (D28).
- **Defaults:** every key that has a sensible default is optional, so a
  minimal case file stays short.
- **Validation** before the run starts:
  - Unknown keys are rejected with a "did you mean …?" suggestion.
  - Types and value ranges are checked.
  - Combinations are checked for consistency, e.g. `turbulence` requires
    `model = "rans"`.
  - Every set of boundary faces in the mesh must have exactly one boundary
    condition, and every boundary face must belong to exactly one set (no
    unassigned faces, no face in several sets). Further rules for boundary
    conditions: D39.
- **Reproducibility:** the resolved configuration, including all defaults, is
  written to the output directory. Each key carries a comment with its
  description and unit, and says whether the value was set explicitly or is a
  default.
- **Examples:** every test case ships with a commented example case file
  (D30).
- **Python API:** exposes the same structure as the case file (D41). There,
  boundary values may also be functions u(x, y, t); the case file supports
  constant values (profiles: P7).


## 2. Sections

| Section | Content |
|---------|---------|
| `[case]` | Name, description |
| `[mesh]` | Mesh file, scaling |
| `[physics]` | Flow model, turbulence model, transition |
| `[fluid]` | Fluid properties |
| `[initial]` | Initial conditions |
| `[boundary.<name>]` | Boundary condition for the boundary faces called `<name>` in the mesh |
| `[reference]` | Reference values for force and moment coefficients |
| `[numerics]` | Spatial discretization |
| `[solver]` | Pseudo-time / nonlinear solver, convergence |
| `[time]` | Steady or unsteady, physical time stepping |
| `[output]` | Result files, monitoring |


## 3. Boundary faces

A *set of boundary faces* is a named group of mesh faces on the domain
boundary, e.g. `airfoil`, `inlet`, `outer`. The mesh file defines the names.
The case file assigns one boundary condition to each name:

```toml
[boundary.airfoil]        # "airfoil" = name of the boundary faces in the mesh
type = "wall"
```

Names containing characters other than letters, digits, `_` and `-` must be
quoted: `[boundary."wall 1"]`.

The same concept in other tools:

| Tool | Term |
|------|------|
| SU2 | marker (`MARKER_TAG`) |
| OpenFOAM | patch |
| Ansys Fluent | boundary zone |
| Gmsh | physical group (physical curve in 2D) |
| CGNS | boundary condition (`BC_t`) / family |


## 4. Examples

### 4.1 Laminar plane Poiseuille flow (channel)

```toml
# Laminar plane Poiseuille flow, Re = 100 (based on channel height and mean velocity)

[mesh]
file = "channel_quad_L2.su2"     # boundary faces: inlet, outlet, upper, lower

[physics]
model = "laminar"                # "euler" | "laminar" | "rans"

[fluid]
density = 1.0                    # kg/m^3
viscosity = 0.01                 # dynamic viscosity, Pa s

[initial]                        # fluid at rest
velocity = [0.0, 0.0]            # m/s
pressure = 0.0                   # Pa, gauge

[boundary.inlet]
type = "velocity-inlet"
velocity = [1.0, 0.0]            # m/s, uniform profile

[boundary.outlet]
type = "pressure-outlet"
pressure = 0.0                   # Pa, gauge; fixes the pressure level

[boundary.upper]
type = "wall"                    # no-slip

[boundary.lower]
type = "wall"
```

### 4.2 Airfoil in free flight

```toml
# NACA 0012, alpha = 4 deg, fully turbulent

[mesh]
file = "naca0012.su2"            # boundary faces: airfoil, outer

[physics]
model = "rans"
turbulence = "sa-neg"

[fluid]
density = 1.225                  # kg/m^3
viscosity = 1.789e-5             # dynamic viscosity, Pa s

[initial]
from_boundary = "outer"          # uniform field = farfield state.
                                 # May be omitted: with exactly one farfield
                                 # boundary this is the default.

[boundary.airfoil]
type = "wall"

[boundary.outer]
type = "farfield"                # imposes the freestream state
velocity = 14.6                  # m/s, magnitude
angle_of_attack = 4.0            # deg
pressure = 0.0                   # Pa, gauge
turbulence = { turbulent_viscosity_ratio = 0.21 }   # nu_t/nu; for SA equal
                                 # to nu_tilde/nu = 3 (NASA TMR). Same as the
                                 # default, shown here for illustration.

[reference]
length = 1.0                     # m, chord; velocity, density and flow
                                 # direction are taken from boundary "outer"
```


## 5. Reference

Notation: **req.** = required; **–** = no default. The `#` column links an
entry to a decision (D) or an open step (O) in `requirements.md`.

### `[case]`

| Key | Type | Default | Description | # |
|-----|------|---------|-------------|---|
| `name` | string | file name without extension | Case name, used for output files | |
| `description` | string | `""` | Free text | |

### `[mesh]`

| Key | Type | Default | Description | # |
|-----|------|---------|-------------|---|
| `file` | path | req. | Mesh file, path relative to the case file. Format from the extension: `.su2`; PLOT3D 2D (`.p2dfmt`, `.x`) | D43 |
| `scale` | float | `1.0` | Factor applied to all coordinates (e.g. `0.001` for mm → m) | |

Test-case meshes generated in code are passed through the Python API, not the
case file (D42).

`[mesh.boundary_faces]`, PLOT3D only (proposal, D43). PLOT3D files carry no
boundary names, so they are defined here, one entry per name:

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `edge` | `"i_min"` \| `"i_max"` \| `"j_min"` \| `"j_max"` | req. | Grid edge of the (single) block |
| `x_min`, `x_max`, `y_min`, `y_max` | float | – | Optional coordinate range (m) to split an edge; a face belongs to the set if its centre lies inside |

Coordinate ranges instead of index ranges keep one definition valid for every
level of a grid family. Coincident nodes on block edges (e.g. the wake cut of a
C-grid) are merged, so the cut becomes interior and needs no entry.

```toml
[mesh]
file = "flatplate_clust2_2levelsdown_137x97.p2dfmt"   # NASA TMR grid, PLOT3D 2D

[mesh.boundary_faces]                    # PLOT3D files carry no boundary names
symmetry = { edge = "j_min", x_max = 0.0 }    # upstream of the plate
plate    = { edge = "j_min", x_min = 0.0 }
inlet    = { edge = "i_min" }
outlet   = { edge = "i_max" }
top      = { edge = "j_max" }
```

### `[physics]`

| Key | Type | Default | Description | # |
|-----|------|---------|-------------|---|
| `model` | `"euler"` \| `"laminar"` \| `"rans"` | req. | Governing equations | D10 |
| `turbulence` | `"sa-neg"` | `"sa-neg"` if `model = "rans"` | Turbulence model; `"sst"` later | D11, L2 |
| `mean_pressure_gradient` | [float, float] | – | Driving force for periodic channels: constant mean pressure gradient (dp/dx, dp/dy) in Pa/m, applied as a momentum source | D39 |

`[physics.transition]`, only for `model = "rans"`:

| Key | Type | Default | Description | # |
|-----|------|---------|-------------|---|
| `mode` | `"none"` \| `"prescribed"` | `"none"` | `"none"` = fully turbulent | D15 |
| `laminar_region` | array of tables | – | Regions where SA production is switched off. Exact form (e.g. `{ boundary = "airfoil", side = "upper", x_max = 0.35 }`) is settled in the transition stage | D15 |

### `[fluid]`

| Key | Type | Default | Description | # |
|-----|------|---------|-------------|---|
| `density` | float | req. | ρ in kg/m³ (constant, D5) | |
| `viscosity` | float | – | Dynamic viscosity μ in Pa·s | |
| `reynolds_number` | float | – | Re = ρ U_ref L_ref / μ. μ is derived from it, using `[reference]` velocity and length | D31 |

Exactly one of `viscosity` and `reynolds_number` is required, except for
`model = "euler"`. Non-dimensional setup: `density = 1.0`, reference velocity
and length `1.0`, `reynolds_number = Re` → μ = 1/Re.

### `[initial]`

Either explicit values or `from_boundary`, not both.

| Key | Type | Default | Description | # |
|-----|------|---------|-------------|---|
| `velocity` | [float, float] | – | (u, v) in m/s | D16 |
| `pressure` | float | – | Gauge pressure in Pa | D16 |
| `turbulence` | table | ν̃/ν = 3 | See [Turbulence values](#turbulence-values) | D16, D33 |
| `from_boundary` | string | name of the farfield boundary, if exactly one exists | Uniform field equal to the state of this boundary | D28 |
| `restart` | path | – | Start from a previous solution (later) | |

If `[initial]` is missing and the case has no single farfield boundary, the
case is rejected with a message asking for `[initial]`.

### `[boundary.<name>]`

One table per set of boundary faces. `<name>` is the name in the mesh file.

| Key | Type | Default | Description | # |
|-----|------|---------|-------------|---|
| `type` | string | req. | Boundary condition type, see below | D38 |

Boundary-condition types (D38). All values are constant (profiles and time
dependence: P7). Types marked "inflow" take a `turbulence` table
([Turbulence values](#turbulence-values)):

| `type` | Parameters |
|--------|-----------------------|
| `"wall"` | No-slip wall. Optional `velocity = [u, v]` for a moving wall (e.g. cavity lid) |
| `"slip-wall"` | Identical to `"symmetry"`: zero normal velocity, no shear stress |
| `"symmetry"` | Symmetry plane: zero normal velocity, no shear stress |
| `"farfield"` | `velocity` (magnitude, m/s), `angle_of_attack` (deg, default 0), `pressure` (Pa, default 0); inflow: `turbulence` |
| `"velocity-inlet"` | `velocity = [u, v]` in m/s; inflow: `turbulence` |
| `"total-pressure-inlet"` | `total_pressure` (Pa), `flow_direction`; inflow: `turbulence` |
| `"pressure-outlet"` | `pressure` (Pa, default 0) |
| `"periodic"` | `partner` (name of the matching boundary faces; gets no entry of its own), `translation = [dx, dy]` (m); driving force: `physics.mean_pressure_gradient` |
| `"transpiration-wall"` | `normal_velocity` (m/s; + blowing, − suction), optional time dependence (D18); inflow (blowing): `turbulence` |

### Turbulence values

Used in `[initial]` and at inflow boundaries (D33). The SA working variable ν̃
is needed wherever flow enters the domain and in the initial field. Walls
(ν̃ = 0), outflow and symmetry need no input.

Exactly one of three input forms:

| Form | Keys | Units | Purpose |
|------|------|-------|---------|
| **b (primary)** | `turbulent_viscosity_ratio`, `intensity` | –, – | ν_t/ν (turbulent over molecular viscosity) and Tu as a fraction (0.001 = 0.1 %). `intensity` is required only by models that need k or Tu (SST, transition models); SA reports it as unused |
| a | `kinetic_energy`, `length_scale`, optional `c_mu` | m²/s², m, – | k and the k-ε length scale ℓ (size of the energy-carrying eddies). `c_mu` defaults to 0.09 |
| c | `nu_tilde_ratio` | – | SA only: ν̃/ν, the notation of NASA TMR |

Default, if no `turbulence` table is given: ν̃/ν = 3 (NASA TMR, SU2 default).
The resolved configuration writes it in form b as
`turbulent_viscosity_ratio = 0.2104`.

Conversions (ν = μ/ρ, U = local inflow velocity magnitude):

- Form a → ν_t: ε = C_μ^{3/4} k^{3/2} / ℓ, so ν_t = C_μ k²/ε = C_μ^{1/4} √k ℓ
- Form b → k (when needed): k = 1.5 (Tu U)²; for SST, ω = k/ν_t
- ν_t → SA's ν̃: solve ν_t/ν = χ f_v1(χ) for χ = ν̃/ν, with
  f_v1 = χ³/(χ³ + c_v1³) and c_v1 = 7.1. The left side is monotonic in χ, so
  the solution is unique.

Reported for every inflow and for the initial state, in the log and in the
resolved configuration: the equivalent ν_t/ν, ν̃/ν and, where defined, k, ℓ
and Tu. A warning is issued when ν̃/ν lies outside the NASA TMR range of 3–5
for SA.

### `[reference]`

Needed for force and moment coefficients, and for `fluid.reynolds_number`.

| Key | Type | Default | Description | # |
|-----|------|---------|-------------|---|
| `length` | float | `1.0`; req. with `fluid.reynolds_number` | Reference length (e.g. chord, channel height) in m | D31 |
| `moment_center` | [float, float] | `[0.25, 0.0]` × `length` | Moment reference point in m | |
| `velocity` | float | farfield velocity, if exactly one farfield boundary; otherwise req. with `fluid.reynolds_number` | Reference velocity in m/s | D28, D31 |
| `angle_of_attack` | float | farfield angle of attack | Defines the drag and lift directions | D28 |
| `density` | float | `fluid.density` | Reference density in kg/m³ | |
| `pressure_point` | [float, float] | domain centroid | Only without a pressure-fixing boundary: the pressure in the cell nearest this point is fixed to `initial.pressure` | D39 |

### `[numerics]`

| Key | Type | Default | Description | # |
|-----|------|---------|-------------|---|
| `convective_flux` | `"fds"` | `"fds"` | Flux-difference splitting with low-speed preconditioning | D21 |
| `spatial_order` | `1` \| `2` | `2` | 1: piecewise constant; 2: MUSCL reconstruction | D21 |
| `limiter` | `"venkatakrishnan-wang"` \| `"none"` | `"venkatakrishnan-wang"` | Slope limiter | D23 |
| `limiter_threshold` | float | `0.05` | Wang's ε′ (suggested range 0.01–0.20) | D23 |
| `limiter_freeze` | table | – | `{ after_iterations = n }` or `{ after_residual_drop = orders }` | D23 |
| `gradient` | `"weighted-least-squares"` | `"weighted-least-squares"` | Vertex-neighbour stencil | D22 |
| `gradient_weight_exponent` | float | `1.0` | q in weights 1/d^q; default to be confirmed by verification | D22 |
| `viscous_gradient` | `"alpha-damping"` \| `"edge-correction"` | `"alpha-damping"` | Face-gradient scheme | D24 |
| `alpha` | float | `4/3` | α-damping coefficient | D24 |
| `turbulence_spatial_order` | `1` \| `2` | `1` | SA convection (SU2 default: first order) | |
| `first_order_iterations` | int | `0` | Iterations at first order before switching to `spatial_order` | D36 |

### `[solver]`

| Key | Type | Default | Description | # |
|-----|------|---------|-------------|---|
| `method` | `"implicit"` \| `"explicit-rk"` | `"implicit"` | `"explicit-rk"`: reference/debugging mode | D25 |
| `max_iterations` | int | `10000` | Pseudo-time iterations (steady) | |
| `residual_drop` | float | `8.0` | Convergence: orders of magnitude of residual reduction | |
| `cfl` | table | `{ start = 10.0, max = 1.0e10, cut_factor = 0.1 }` | Pseudo-time CFL number, SER control with safeguards; values tentative | D35 |
| `newton_krylov` | table | `{ enabled = true, switch_at_residual_drop = 2.0 }` | Switch from approximate-Jacobian iterations to Newton–Krylov | D25 |
| `linear_solver` | table | `{ type = "direct" }` | `"direct"`: sparse LU (SciPy SuperLU), option `refactor_interval`. `"gmres-ilu"`: GMRES with ILU, options `tolerance`, `restart`, `ilu_drop_tolerance`, `ilu_fill_factor`. Default parameter values set from measurements in the first implicit stage | D32 |
| `turbulence_coupling` | `"segregated"` \| `"coupled"` | `"segregated"` | `"segregated"`: mean flow and turbulence equations solved one after the other per iteration; `"coupled"`: one system for all variables | D34 |

### `[time]`

| Key | Type | Default | Description | # |
|-----|------|---------|-------------|---|
| `mode` | `"steady"` \| `"unsteady"` | `"steady"` | | D9 |
| `scheme` | `"bdf2"` | `"bdf2"` | Dual time stepping; the first time step uses BDF1 | D9, D37 |
| `time_step` | float | req. if unsteady | Δt in s | D9 |
| `end_time` | float | req. if unsteady | in s | D9 |
| `inner_iterations` | int | `50` | Maximum pseudo-time iterations per time step (tentative) | D37 |
| `inner_residual_drop` | float | `3.0` | Inner convergence, orders of magnitude (tentative) | D37 |

### `[output]`

| Key | Type | Default | Description | # |
|-----|------|---------|-------------|---|
| `directory` | path | `"results"` | Relative to the case file | |
| `formats` | array | `["vtu"]` | *open: O11* | O11 |
| `interval` | int | `0` | Write every n iterations / time steps; 0 = only at the end | |
| `force_boundaries` | array of strings | all `"wall"` boundaries | Boundary faces over which forces and moments are integrated | |
| `history` | bool | `true` | Residuals and coefficients per iteration (CSV) | |


## 6. Open questions

None.
