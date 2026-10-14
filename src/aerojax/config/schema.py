"""Declarative schema of the case file (docs/configuration.md).

Every option has a type, a default (or ``REQUIRED``), a description and an
optional unit; the resolved configuration is written with these as comments
(D27, D30). Keys that exist in the documentation but are not implemented in
the current stage are listed in ``PLANNED`` and rejected with a message.
"""

from typing import NamedTuple

REQUIRED = object()


class Opt(NamedTuple):
    """One configuration option."""

    type: str  # "float", "int", "bool", "string", "float2", "string_list"
    default: object
    description: str
    unit: str = ""
    choices: tuple = ()
    minimum: float = None


SCHEMA = {
    "case": {
        "name": Opt("string", None, "Case name; default: file name without extension"),
        "description": Opt("string", "", "Free text"),
    },
    "mesh": {
        "file": Opt("string", REQUIRED, "Mesh file (.su2), relative to the case file"),
        "scale": Opt("float", 1.0, "Factor applied to all coordinates", minimum=0.0),
    },
    "physics": {
        "model": Opt(
            "string", REQUIRED, "Governing equations", choices=("euler", "laminar")
        ),
        "mean_pressure_gradient": Opt(
            "float2",
            None,
            "Driving force for periodic channels: mean pressure gradient "
            "(dp/dx, dp/dy), applied as the body force -grad p",
            "Pa/m",
        ),
    },
    "fluid": {
        "density": Opt("float", REQUIRED, "Density rho", "kg/m^3", minimum=0.0),
        "viscosity": Opt("float", None, "Dynamic viscosity mu", "Pa s", minimum=0.0),
        "reynolds_number": Opt(
            "float",
            None,
            "Re = rho U_ref L_ref / mu; alternative to viscosity",
            minimum=0.0,
        ),
    },
    "initial": {
        "velocity": Opt("float2", None, "Initial velocity (u, v)", "m/s"),
        "pressure": Opt("float", None, "Initial gauge pressure", "Pa"),
        "from_boundary": Opt(
            "string", None, "Uniform initial field equal to this boundary's state"
        ),
    },
    "reference": {
        "length": Opt(
            "float", 1.0, "Reference length (chord, height)", "m", minimum=0.0
        ),
        "velocity": Opt(
            "float",
            None,
            "Reference velocity; default: farfield velocity if exactly one "
            "farfield boundary exists",
            "m/s",
            minimum=0.0,
        ),
        "density": Opt(
            "float", None, "Reference density; default fluid.density", "kg/m^3"
        ),
        "pressure_point": Opt(
            "float2",
            None,
            "Without a pressure-fixing boundary: pressure is pinned in the cell "
            "nearest this point; default domain centroid",
            "m",
        ),
    },
    "numerics": {
        "convective_flux": Opt("string", "fds", "Convective flux", choices=("fds",)),
        "spatial_order": Opt(
            "int", 2, "1: piecewise constant, 2: MUSCL", choices=(1, 2)
        ),
        "limiter": Opt(
            "string",
            "venkatakrishnan-wang",
            "Slope limiter",
            choices=("venkatakrishnan-wang", "none"),
        ),
        "limiter_threshold": Opt(
            "float", 0.05, "Wang's epsilon' (suggested 0.01-0.20)", minimum=0.0
        ),
        "gradient": Opt(
            "string",
            "weighted-least-squares",
            "Gradient reconstruction",
            choices=("weighted-least-squares",),
        ),
        "gradient_weight_exponent": Opt(
            "float", 1.0, "q in the least-squares weights 1/d^q", minimum=0.0
        ),
        "viscous_gradient": Opt(
            "string",
            "alpha-damping",
            "Face-gradient scheme for viscous fluxes",
            choices=("alpha-damping", "edge-correction"),
        ),
        "alpha": Opt("float", 4.0 / 3.0, "alpha-damping coefficient", minimum=0.0),
        "first_order_iterations": Opt(
            "int",
            0,
            "Iterations at first order before switching to spatial_order",
            minimum=0,
        ),
        "beta_factor": Opt(
            "float",
            4.1,
            "Artificial compressibility: beta^2 = beta_factor * max|v|^2",
            minimum=0.0,
        ),
    },
    "solver": {
        "method": Opt(
            "string",
            "implicit",
            "Pseudo-time method",
            choices=("implicit", "explicit-rk"),
        ),
        "max_iterations": Opt(
            "int", 10000, "Maximum pseudo-time iterations", minimum=1
        ),
        "residual_drop": Opt(
            "float",
            8.0,
            "Convergence: orders of magnitude of residual reduction",
            minimum=0.0,
        ),
        "cfl": {
            "start": Opt("float", 10.0, "Initial CFL number", minimum=0.0),
            "max": Opt("float", 1.0e10, "Maximum CFL number", minimum=0.0),
            "growth_factor": Opt(
                "float", 2.0, "Minimum CFL growth per accepted iteration", minimum=1.0
            ),
            "cut_factor": Opt(
                "float", 0.1, "CFL reduction after a rejected update", minimum=0.0
            ),
            "reject_factor": Opt(
                "float",
                10.0,
                "Reject an update that raises the residual by more",
                minimum=1.0,
            ),
            "explicit": Opt(
                "float", 1.0, "CFL number of the explicit Runge-Kutta mode", minimum=0.0
            ),
        },
    },
    "time": {
        "mode": Opt("string", "steady", "Time mode", choices=("steady",)),
    },
    "output": {
        "directory": Opt(
            "string", "results", "Output directory, relative to the case file"
        ),
        "formats": Opt("string_list", ["vtu"], "Field output formats"),
        "interval": Opt(
            "int", 0, "Write fields every n iterations; 0 = only at the end", minimum=0
        ),
        "history": Opt("bool", True, "Write the convergence history (CSV and .log)"),
        "verbosity": Opt(
            "string", "normal", "Log detail", choices=("normal", "detailed")
        ),
    },
}

# Boundary-condition types and their parameters (D38).
BOUNDARY_TYPES = {
    "wall": {
        "velocity": Opt("float2", [0.0, 0.0], "Wall velocity (moving wall)", "m/s"),
    },
    "symmetry": {},
    "slip-wall": {},
    "velocity-inlet": {
        "velocity": Opt("float2", REQUIRED, "Inlet velocity (u, v)", "m/s"),
    },
    "pressure-outlet": {
        "pressure": Opt("float", 0.0, "Static gauge pressure", "Pa"),
    },
    "farfield": {
        "velocity": Opt("float", REQUIRED, "Freestream velocity magnitude", "m/s"),
        "angle_of_attack": Opt("float", 0.0, "Angle of attack", "deg"),
        "pressure": Opt("float", 0.0, "Freestream gauge pressure", "Pa"),
    },
    "periodic": {
        "partner": Opt("string", REQUIRED, "Name of the matching boundary faces"),
        "translation": Opt(
            "float2", REQUIRED, "Translation from this boundary to the partner", "m"
        ),
    },
}

# Documented keys of later stages: rejected with a pointer to the stage.
PLANNED = {
    "physics.turbulence": "stage 5 (RANS)",
    "physics.transition": "stage 5 (RANS)",
    "solver.newton_krylov": "stage 3",
    "solver.linear_solver": "stage 3",
    "solver.turbulence_coupling": "stage 5 (RANS)",
    "numerics.limiter_freeze": "stage 3",
    "numerics.turbulence_spatial_order": "stage 5 (RANS)",
    "time.scheme": "stage 4 (unsteady)",
    "time.time_step": "stage 4 (unsteady)",
    "time.end_time": "stage 4 (unsteady)",
    "time.inner_iterations": "stage 4 (unsteady)",
    "time.inner_residual_drop": "stage 4 (unsteady)",
    "output.force_boundaries": "stage 3",
    "initial.restart": "later",
    "initial.turbulence": "stage 5 (RANS)",
    "reference.moment_center": "stage 3",
    "reference.angle_of_attack": "stage 3",
}
PLANNED_BOUNDARY_TYPES = {
    "total-pressure-inlet": "stage 3",
    "transpiration-wall": "stage 3",
}
