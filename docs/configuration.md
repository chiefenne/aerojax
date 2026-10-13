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
    condition.
- **Reproducibility:** the resolved configuration, including all defaults, is
  written to the output directory. Each key carries a comment with its
  description and unit, and says whether the value was set explicitly or is a
  default.
- **Examples:** every test case ships with a commented example case file
  (D30).


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
# turbulence values: open (O9)

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
| `file` | path | req. | Mesh file (`.su2`); path relative to the case file | O11 |
| `scale` | float | `1.0` | Factor applied to all coordinates (e.g. `0.001` for mm → m) | |

Test-case meshes generated in code are passed through the Python API, not the
case file (O11).

### `[physics]`

| Key | Type | Default | Description | # |
|-----|------|---------|-------------|---|
| `model` | `"euler"` \| `"laminar"` \| `"rans"` | req. | Governing equations | D10 |
| `turbulence` | `"sa-neg"` | `"sa-neg"` if `model = "rans"` | Turbulence model; `"sst"` later | D11, L2 |

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
| turbulence | – | – | *open: O9* (same form as at boundaries) | D16, O9 |
| `from_boundary` | string | name of the farfield boundary, if exactly one exists | Uniform field equal to the state of this boundary | D28 |
| `restart` | path | – | Start from a previous solution (later) | |

If `[initial]` is missing and the case has no single farfield boundary, the
case is rejected with a message asking for `[initial]`.

### `[boundary.<name>]`

One table per set of boundary faces. `<name>` is the name in the mesh file.

| Key | Type | Default | Description | # |
|-----|------|---------|-------------|---|
| `type` | string | req. | Boundary condition type, see below | O9 |

Proposed types and parameters (*open: O9*; turbulence values for all inflow
types also open):

| `type` | Parameters (proposal) |
|--------|-----------------------|
| `"wall"` | No-slip wall. Optional `velocity = [u, v]` for a moving wall (e.g. cavity lid) |
| `"slip-wall"` | Inviscid wall, zero normal velocity |
| `"symmetry"` | Symmetry plane |
| `"farfield"` | `velocity` (magnitude, m/s), `angle_of_attack` (deg, default 0), `pressure` (Pa, default 0) |
| `"velocity-inlet"` | `velocity = [u, v]` in m/s |
| `"total-pressure-inlet"` | `total_pressure` (Pa), `flow_direction` |
| `"pressure-outlet"` | `pressure` (Pa, default 0) |
| `"periodic"` | `partner` (name of the matching boundary faces), `translation = [dx, dy]` (m) |
| `"transpiration-wall"` | `normal_velocity` (m/s; + blowing, − suction), optional time dependence (D18) |

### `[reference]`

Needed for force and moment coefficients, and for `fluid.reynolds_number`.

| Key | Type | Default | Description | # |
|-----|------|---------|-------------|---|
| `length` | float | `1.0`; req. with `fluid.reynolds_number` | Reference length (e.g. chord, channel height) in m | D31 |
| `moment_center` | [float, float] | `[0.25, 0.0]` × `length` | Moment reference point in m | |
| `velocity` | float | farfield velocity, if exactly one farfield boundary; otherwise req. with `fluid.reynolds_number` | Reference velocity in m/s | D28, D31 |
| `angle_of_attack` | float | farfield angle of attack | Defines the drag and lift directions | D28 |
| `density` | float | `fluid.density` | Reference density in kg/m³ | |

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

### `[solver]`

| Key | Type | Default | Description | # |
|-----|------|---------|-------------|---|
| `method` | `"implicit"` \| `"explicit-rk"` | `"implicit"` | `"explicit-rk"`: reference/debugging mode | D25 |
| `max_iterations` | int | `10000` | Pseudo-time iterations (steady) | |
| `residual_drop` | float | `8.0` | Convergence: orders of magnitude of residual reduction | |
| `cfl` | table | `{ start = 10.0, max = 1.0e4, adaptive = true }` | Pseudo-time CFL number | O8d |
| `newton_krylov` | table | `{ enabled = true, switch_at_residual_drop = 2.0 }` | Switch from approximate-Jacobian iterations to Newton–Krylov | D25 |
| `linear_solver` | table | `{ type = "direct" }` | `"direct"`: sparse LU (SciPy SuperLU), option `refactor_interval`. `"gmres-ilu"`: GMRES with ILU, options `tolerance`, `restart`, `ilu_drop_tolerance`, `ilu_fill_factor`. Default parameter values set from measurements in the first implicit stage | D32 |
| `turbulence_coupling` | string | – | *open: O8c* | O8c |

### `[time]`

| Key | Type | Default | Description | # |
|-----|------|---------|-------------|---|
| `mode` | `"steady"` \| `"unsteady"` | `"steady"` | | D9 |
| `scheme` | `"bdf2"` | `"bdf2"` | Dual time stepping | D9 |
| `time_step` | float | req. if unsteady | Δt in s | D9 |
| `end_time` | float | req. if unsteady | in s | D9 |
| `inner_iterations` | int | `50` | Maximum pseudo-time iterations per time step | O8d |
| `inner_residual_drop` | float | `3.0` | Inner convergence, orders of magnitude | O8d |

### `[output]`

| Key | Type | Default | Description | # |
|-----|------|---------|-------------|---|
| `directory` | path | `"results"` | Relative to the case file | |
| `formats` | array | `["vtu"]` | *open: O11* | O11 |
| `interval` | int | `0` | Write every n iterations / time steps; 0 = only at the end | |
| `force_boundaries` | array of strings | all `"wall"` boundaries | Boundary faces over which forces and moments are integrated | |
| `history` | bool | `true` | Residuals and coefficients per iteration (CSV) | |


## 6. Open questions

1. Turbulence values at inflow boundaries and in the initial state (O9).
