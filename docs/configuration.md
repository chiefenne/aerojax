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
- **Units:** SI and angles in degrees. Units are documented, not encoded in
  key names. *Open: O12 (non-dimensionalization).*
- **Defaults:** every key that has a sensible default is optional, so a
  minimal case file stays short.
- **Validation** before the run starts:
  - Unknown keys are rejected with a "did you mean …?" suggestion.
  - Types and value ranges are checked.
  - Combinations are checked for consistency, e.g. `turbulence` requires
    `model = "rans"`.
- **Reproducibility:** the resolved configuration, including all defaults, is
  written to the output directory.


## 2. Sections

| Section | Content |
|---------|---------|
| `[case]` | Name, description |
| `[mesh]` | Mesh file, scaling |
| `[physics]` | Flow model, turbulence model, transition |
| `[fluid]` | Fluid properties |
| `[freestream]` | Freestream state; also the default initial state and farfield state |
| `[reference]` | Reference values for force and moment coefficients |
| `[initial]` | Initial state, if it differs from the freestream |
| `[[boundary]]` | One entry per boundary marker |
| `[numerics]` | Spatial discretization |
| `[solver]` | Pseudo-time / nonlinear solver, convergence |
| `[time]` | Steady or unsteady, physical time stepping |
| `[output]` | Result files, monitoring |


## 3. Minimal example

```toml
# Laminar plane Poiseuille flow
[mesh]
file = "channel_quad_L2.su2"

[physics]
model = "laminar"

[fluid]
density = 1.0
viscosity = 0.01

[freestream]
velocity = 1.0

[[boundary]]
marker = "inlet"
type = "velocity-inlet"

[[boundary]]
marker = "outlet"
type = "pressure-outlet"

[[boundary]]
marker = ["upper", "lower"]
type = "wall"
```


## 4. Reference

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
| `laminar_region` | array of tables | – | Regions where SA production is switched off. Exact form (e.g. `{ marker = "airfoil", side = "upper", x_max = 0.35 }`) is settled in the transition stage | D15 |

### `[fluid]`

| Key | Type | Default | Description | # |
|-----|------|---------|-------------|---|
| `density` | float | req. | ρ in kg/m³ (constant, D5) | |
| `viscosity` | float | req. unless `model = "euler"` | Dynamic viscosity μ in Pa·s | |

### `[freestream]`

| Key | Type | Default | Description | # |
|-----|------|---------|-------------|---|
| `velocity` | float | req. | Magnitude U∞ in m/s | |
| `angle_of_attack` | float | `0.0` | α in degrees | |
| `pressure` | float | `0.0` | Gauge pressure in Pa (incompressible: only differences matter) | |
| `nu_tilde_ratio` | float | `3.0` | SA: ν̃∞/ν∞ (SU2 default 3) | D11 |

### `[reference]`

| Key | Type | Default | Description | # |
|-----|------|---------|-------------|---|
| `length` | float | `1.0` | Reference length (e.g. chord) in m | |
| `moment_center` | [float, float] | `[0.25, 0.0]` × `length` | Moment reference point in m | |
| `velocity` | float | `freestream.velocity` | Reference velocity for coefficients | |
| `density` | float | `fluid.density` | Reference density for coefficients | |

### `[initial]`

All keys default to the freestream state.

| Key | Type | Default | Description | # |
|-----|------|---------|-------------|---|
| `velocity` | [float, float] | freestream vector | (u, v) in m/s | D16 |
| `pressure` | float | `freestream.pressure` | Gauge pressure in Pa | D16 |
| `nu_tilde_ratio` | float | `freestream.nu_tilde_ratio` | SA initial value | D16 |
| `restart` | path | – | Start from a previous solution (later) | |

### `[[boundary]]`

One table per boundary type and marker set. Every mesh marker must be
assigned exactly once.

| Key | Type | Default | Description | # |
|-----|------|---------|-------------|---|
| `marker` | string or array of strings | req. | Mesh marker name(s) | |
| `type` | string | req. | Boundary type, see below | O9 |

Proposed types (*open: O9*):

| `type` | Parameters (proposal) |
|--------|-----------------------|
| `"wall"` | No-slip wall. Optional `velocity = [u, v]` for a moving wall (e.g. cavity lid) |
| `"slip-wall"` | Inviscid wall, zero normal velocity |
| `"symmetry"` | Symmetry plane |
| `"velocity-inlet"` | `velocity = [u, v]` (default: freestream), `nu_tilde_ratio` |
| `"total-pressure-inlet"` | `total_pressure`, `flow_direction`, `nu_tilde_ratio` |
| `"pressure-outlet"` | `pressure` (default: freestream) |
| `"farfield"` | Uses `[freestream]` |
| `"periodic"` | `partner` marker, `translation = [dx, dy]` |
| `"transpiration-wall"` | `normal_velocity` (+ blowing, − suction), optional time dependence; `nu_tilde_ratio` for blown fluid (D18) |

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
| `linear_solver` | table | – | *open: O8b* | O8b |
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
| `force_markers` | array of strings | all walls | Markers over which forces and moments are integrated | |
| `history` | bool | `true` | Residuals and coefficients per iteration (CSV) | |


## 5. Open questions for review

1. Section structure (§2): complete and logically split?
2. Freestream input: dimensional (velocity, density, viscosity, as above), or
   alternatively via Reynolds number and reference length? (O12)
3. `[[boundary]]` with `marker` as a string or a list: acceptable?
