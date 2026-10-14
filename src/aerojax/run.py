"""Run a case from a TOML file: setup, non-dimensional solve, output.

Internally the solver works with rho_ref = U_ref = L_ref = 1 (D51): the mesh
is scaled by 1 / L_ref, velocities by 1 / U_ref, pressures by
1 / (rho_ref U_ref^2), the viscosity becomes 1 / Re. Results are scaled back
before output.
"""

import math
from pathlib import Path
from typing import NamedTuple

import numpy as np

from aerojax import __version__
from aerojax.config import Resolved, load_case
from aerojax.driver import IncompressibleProblem
from aerojax.io import HistoryWriter, RunLog, write_vtu
from aerojax.mesh import Mesh, read_su2
from aerojax.physics import forces as fz
from aerojax.physics import incompressible as inc
from aerojax.solver import LinearSettings, PseudoTimeSettings


class Scaling(NamedTuple):
    """Reference values of the non-dimensionalisation (D51)."""

    density: float
    velocity: float
    length: float

    @property
    def pressure(self):
        """rho_ref U_ref^2."""
        return self.density * self.velocity**2

    @property
    def viscosity(self):
        """rho_ref U_ref L_ref."""
        return self.density * self.velocity * self.length


class RunResult(NamedTuple):
    """Outcome of :func:`run_case`.

    Attributes
    ----------
    config : Resolved
    mesh : Mesh
        The mesh in physical units.
    w : ndarray, shape (n_cells, 3)
        (p, u, v) in physical units.
    converged : bool
    history : dict
    output_dir : Path
    coefficients : Coefficients or None
        Final force coefficients over ``output.force_boundaries``.
    """

    config: Resolved
    mesh: Mesh
    w: np.ndarray
    converged: bool
    history: dict
    output_dir: Path
    coefficients: object = None


def setup(config, mesh=None):
    """Build the non-dimensional problem of a resolved configuration.

    Parameters
    ----------
    config : Resolved
    mesh : Mesh, optional
        Mesh in physical units; default: read ``mesh.file``.

    Returns
    -------
    problem : IncompressibleProblem
    scaling : Scaling
    w0 : ndarray
        Non-dimensional initial state.
    mesh : Mesh
        The physical mesh.
    """
    cfg = config.data
    if mesh is None:
        mesh = read_su2(config.resolve_path(cfg["mesh"]["file"]), cfg["mesh"]["scale"])
    fluid, ref, physics = cfg["fluid"], cfg["reference"], cfg["physics"]
    scaling = Scaling(
        float(ref["density"]), float(ref["velocity"]), float(ref["length"])
    )
    if physics["model"] == "euler":
        mu_nd = 0.0
    elif fluid["viscosity"] is not None:
        mu_nd = fluid["viscosity"] / scaling.viscosity
    else:
        mu_nd = 1.0 / fluid["reynolds_number"]
    rho_nd = fluid["density"] / scaling.density

    conditions = {}
    for name, b in cfg["boundary"].items():
        kind = b["type"]
        if kind == "wall" and physics["model"] == "euler":
            kind = "slip-wall"
        entry = {"type": kind}
        if kind == "periodic":
            entry["partner"] = b["partner"]
            entry["translation"] = tuple(t / scaling.length for t in b["translation"])
        elif kind == "farfield":
            a = math.radians(b["angle_of_attack"])
            u = b["velocity"] / scaling.velocity
            entry["velocity"] = (u * math.cos(a), u * math.sin(a))
            entry["pressure"] = b["pressure"] / scaling.pressure
        elif kind in ("wall", "velocity-inlet"):
            entry["velocity"] = tuple(v / scaling.velocity for v in b["velocity"])
        elif kind == "pressure-outlet":
            entry["pressure"] = b["pressure"] / scaling.pressure
        elif kind == "total-pressure-inlet":
            entry["total_pressure"] = b["total_pressure"] / scaling.pressure
            entry["flow_direction"] = tuple(b["flow_direction"])
        elif kind == "transpiration-wall":
            entry["normal_velocity"] = b["normal_velocity"] / scaling.velocity
        conditions[name] = entry

    num = cfg["numerics"]
    force = (0.0, 0.0)
    if physics["mean_pressure_gradient"] is not None:
        g = physics["mean_pressure_gradient"]
        force = tuple(-gi * scaling.length / scaling.pressure for gi in g)
    model = inc.IncompressibleModel(
        density=rho_nd,
        viscosity=mu_nd,
        beta_factor=num["beta_factor"],
        spatial_order=num["spatial_order"],
        limiter=num["limiter"],
        limiter_threshold=num["limiter_threshold"],
        viscous_gradient=num["viscous_gradient"],
        alpha=num["alpha"],
        body_force=force,
    )
    nd_mesh = _scaled_mesh(mesh, 1.0 / scaling.length)
    point = ref["pressure_point"]
    problem = IncompressibleProblem(
        nd_mesh,
        model,
        conditions,
        weight_exponent=num["gradient_weight_exponent"],
        pressure_point=None
        if point is None
        else tuple(p / scaling.length for p in point),
        pin_pressure=cfg["initial"]["pressure"] / scaling.pressure,
    )

    init = cfg["initial"]
    if init["from_boundary"] is not None:
        src = conditions[init["from_boundary"]]
        velocity, pressure = src["velocity"], src.get("pressure", 0.0)
    else:
        velocity = tuple(v / scaling.velocity for v in init["velocity"])
        pressure = init["pressure"] / scaling.pressure
    w0 = problem.initial_state(velocity, pressure)
    return problem, scaling, w0, mesh


def run_case(path, console=True, mesh=None):
    """Run a case file end to end and write the results.

    Parameters
    ----------
    path : str or Path
        TOML case file.
    console : bool
        Echo the log to stdout.
    mesh : Mesh, optional
        Use this mesh instead of reading ``mesh.file`` (tests).

    Returns
    -------
    RunResult
    """
    config = load_case(path)
    cfg = config.data
    out_dir = config.resolve_path(cfg["output"]["directory"])
    out_dir.mkdir(parents=True, exist_ok=True)
    name = config.name
    (out_dir / f"{name}.resolved.toml").write_text(config.to_toml())

    with RunLog(out_dir / f"{name}.log", console) as log:
        log.write(f"aerojax {__version__}: case '{name}' ({config.path})")
        problem, scaling, w0, mesh = setup(config, mesh)
        _log_setup(log, config, problem, scaling, mesh)

        names = ("p", "u", "v")
        implicit = cfg["solver"]["method"] == "implicit"
        force_boundaries = cfg["output"]["force_boundaries"]
        monitor = Monitor(config, problem, scaling) if force_boundaries else None
        coeff_names = ("CL", "CD", "CM") if monitor else ()
        history = HistoryWriter(out_dir / f"{name}.history.csv", names, coeff_names)
        extra_names = (("NK", "GMRES") if implicit else ()) + tuple(
            (c, 11) for c in coeff_names
        )
        log.iteration_header(names, extra_names)
        interval = cfg["output"]["interval"]

        timing = {"setup": None}

        def callback(it, w, residuals, cfl, accepted):
            if it == 0:
                timing["setup"] = log.elapsed
            info = callback.info
            nk, lin = info.get("newton_krylov", False), info.get("linear_iterations", 0)
            extra = ("NK" if nk else "", lin if lin else "") if implicit else ()
            coeffs = monitor.coefficients(w) if monitor else ()
            extra += tuple(f"{c:.5f}" for c in coeffs)
            log.iteration(it, residuals, cfl, accepted, extra, info.pop("note", ""))
            history.row(it, residuals, cfl, accepted, log.elapsed, nk, lin, coeffs)
            if interval and it and it % interval == 0:
                _write_fields(out_dir / f"{name}_{it:06d}.vtu", mesh, w, scaling)

        callback.info = {}

        solver = cfg["solver"]
        if solver["method"] == "implicit":
            settings = PseudoTimeSettings(
                cfl_start=solver["cfl"]["start"],
                cfl_max=solver["cfl"]["max"],
                growth_factor=solver["cfl"]["growth_factor"],
                cut_factor=solver["cfl"]["cut_factor"],
                reject_factor=solver["cfl"]["reject_factor"],
                max_iterations=solver["max_iterations"],
                residual_drop=solver["residual_drop"],
                first_order_iterations=cfg["numerics"]["first_order_iterations"],
                newton_krylov=solver["newton_krylov"]["enabled"],
                nk_switch_residual_drop=solver["newton_krylov"][
                    "switch_at_residual_drop"
                ],
                nk_linear_tolerance=solver["newton_krylov"]["linear_tolerance"],
                nk_restart=solver["newton_krylov"]["restart"],
                nk_max_krylov=solver["newton_krylov"]["max_iterations"],
                linear=LinearSettings(**solver["linear_solver"]),
            )
            freeze = cfg["numerics"]["limiter_freeze"]
            result = problem.solve(
                w0,
                settings,
                callback=callback,
                freeze_limiter_after_iterations=freeze["after_iterations"],
                freeze_limiter_after_residual_drop=freeze["after_residual_drop"],
            )
        else:
            result = problem.solve_explicit(
                w0,
                cfl=solver["cfl"]["explicit"],
                max_iterations=solver["max_iterations"],
                residual_drop=solver["residual_drop"],
                callback=callback,
            )
        history.close()

        w = result.u * np.array([scaling.pressure, scaling.velocity, scaling.velocity])
        _write_fields(out_dir / f"{name}.vtu", mesh, result.u, scaling)
        h = result.history
        log.section("Summary")
        nk_steps = int(np.count_nonzero(h["newton_krylov"]))
        rows = [
            ("converged", "yes" if result.converged else "NO"),
            ("iterations", f"{int(h['iteration'][-1])} ({nk_steps} Newton-Krylov)"),
            ("rejected updates", str(int(np.count_nonzero(~h["accepted"])))),
            ("residual drop", f"{h['residual'][-1] / h['residual'][0]:.2e}"),
            ("setup and compilation", f"{timing['setup']:.1f} s"),
            ("wall time", f"{log.elapsed:.1f} s"),
            ("fields", str(out_dir / f"{name}.vtu")),
            ("history", str(history.path)),
        ]
        coefficients = None
        if monitor:
            coefficients = monitor.final(result.u)
            c = coefficients
            rows[4:4] = [
                ("force boundaries", ", ".join(force_boundaries)),
                ("CL, CD, CM", f"{c.cl:.6f}, {c.cd:.6f}, {c.cm:.6f}"),
                ("CD pressure, viscous", f"{c.cd_pressure:.6f}, {c.cd_viscous:.6f}"),
                ("Cx, Cy", f"{c.cx:.6f}, {c.cy:.6f}"),
            ]
            for bname in force_boundaries:
                path = out_dir / f"{name}.wall_{bname}.csv"
                monitor.write_wall(result.u, bname, path)
                rows.append((f"wall '{bname}'", str(path)))
        log.table(rows)
    return RunResult(
        config, mesh, w, result.converged, result.history, out_dir, coefficients
    )


class Monitor:
    """Force coefficients of the non-dimensional solution (D45).

    With rho_ref = U_ref = L_ref = 1 internally, q = rho_nd / 2 with the
    non-dimensional reference density and the reference length is 1.
    """

    def __init__(self, config, problem, scaling):
        cfg = config.data
        self.problem = problem
        self.scaling = scaling
        self.boundaries = cfg["output"]["force_boundaries"]
        self.q = 0.5 * cfg["reference"]["density"] / scaling.density
        self.alpha = cfg["reference"]["angle_of_attack"]
        self.moment_center = tuple(
            c / scaling.length for c in cfg["reference"]["moment_center"]
        )
        farfields = [n for n, b in cfg["boundary"].items() if b["type"] == "farfield"]
        self.p_ref = (
            cfg["boundary"][farfields[0]]["pressure"] / scaling.pressure
            if len(farfields) == 1
            else 0.0
        )

    def coefficients(self, w):
        """(CL, CD, CM) of the state ``w``."""
        c = self.final(w)
        return (c.cl, c.cd, c.cm)

    def final(self, w):
        """Full :class:`~aerojax.physics.forces.Coefficients`."""
        f = self.problem.forces(w, self.boundaries, self.moment_center)
        return fz.coefficients(f, self.q, 1.0, self.alpha)

    def write_wall(self, w, boundary, path):
        """Write x, y, p, Cp, tau_x, tau_y, Cf of one boundary (SI units)."""
        loads = self.problem.wall_loads(w)
        faces = self.problem.boundary_faces([boundary])
        d = fz.wall_distribution(loads, self.problem.geom, faces, self.q, self.p_ref)
        s = self.scaling
        columns = {
            "x": d["x"] * s.length,
            "y": d["y"] * s.length,
            "p": d["p"] * s.pressure,
            "cp": d["cp"],
            "tau_x": d["tau_x"] * s.pressure,
            "tau_y": d["tau_y"] * s.pressure,
            "cf": d["cf"],
        }
        header = ",".join(columns)
        rows = np.column_stack(list(columns.values()))
        np.savetxt(path, rows, delimiter=",", header=header, comments="", fmt="%.10e")


def _scaled_mesh(mesh, factor):
    if factor == 1.0:
        return mesh
    boundary = {
        name: mesh.face_nodes[faces] for name, faces in mesh.boundary_faces.items()
    }
    cells = [mesh.cell(i) for i in range(mesh.n_cells)]
    return Mesh.from_cells(mesh.nodes * factor, cells, boundary)


def _write_fields(path, mesh, w_nd, scaling):
    p = w_nd[:, 0] * scaling.pressure
    vel = w_nd[:, 1:] * scaling.velocity
    write_vtu(
        path,
        mesh,
        cell_data={
            "pressure": p,
            "velocity": vel,
            "velocity_magnitude": np.hypot(vel[:, 0], vel[:, 1]),
        },
    )


def _log_setup(log, config, problem, scaling, mesh):
    cfg = config.data
    log.section("Configuration")
    rows = [
        ("model", cfg["physics"]["model"]),
        ("density", f"{cfg['fluid']['density']} kg/m^3"),
    ]
    if cfg["fluid"]["viscosity"] is not None:
        rows.append(("viscosity", f"{cfg['fluid']['viscosity']} Pa s"))
    if cfg["fluid"]["reynolds_number"] is not None:
        rows.append(("Reynolds number", f"{cfg['fluid']['reynolds_number']}"))
    rows += [
        (
            "reference",
            f"rho {scaling.density} kg/m^3, U {scaling.velocity} m/s, "
            f"L {scaling.length} m",
        ),
        ("viscosity (non-dim.)", f"{problem.model.viscosity:.4e} (= 1/Re)"),
        (
            "convective flux",
            f"{cfg['numerics']['convective_flux']}, order "
            f"{cfg['numerics']['spatial_order']}, limiter "
            f"{cfg['numerics']['limiter']}",
        ),
        ("viscous gradient", cfg["numerics"]["viscous_gradient"]),
        ("solver", cfg["solver"]["method"]),
        ("resolved configuration", f"{config.name}.resolved.toml"),
    ]
    log.table(rows)
    log.section("Mesh")
    for line in mesh.summary().splitlines():
        log.write("  " + line)
    log.section("Boundary conditions")
    log.table([(name, _describe_boundary(b)) for name, b in cfg["boundary"].items()])
    if problem.pinned_cell is not None:
        c = mesh.cell_centroid[problem.pinned_cell]
        config.notices.append(
            f"pressure pinned in cell {problem.pinned_cell} at ({c[0]:.4g}, {c[1]:.4g})"
        )
    if config.notices:
        log.section("Notices")
        for note in config.notices:
            log.write("  - " + note)


def _describe_boundary(b):
    parts = [b["type"]]
    for key, value in b.items():
        if key != "type":
            parts.append(f"{key} = {value}")
    return ", ".join(parts)
