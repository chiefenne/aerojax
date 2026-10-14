"""Set up and solve an incompressible flow problem (Python API, D41).

:class:`IncompressibleProblem` binds a mesh, a model and boundary conditions
and provides the residual, Jacobian and pseudo-time term for the solvers in
:mod:`aerojax.solver`.
"""

from functools import partial

import jax.numpy as jnp
import numpy as np

from aerojax.numerics import build_geometry
from aerojax.physics import forces as fz
from aerojax.physics import incompressible as inc
from aerojax.solver import (
    PseudoTimeSettings,
    block_diagonal,
    first_order_jacobian,
    replace_rows,
    solve_explicit,
    solve_steady,
    solve_unsteady,
)

PRESSURE_FIXING = {"pressure-outlet", "farfield", "total-pressure-inlet"}


class IncompressibleProblem:
    """An incompressible flow case on a mesh.

    Parameters
    ----------
    mesh : Mesh
    model : IncompressibleModel
        If ``model.pin`` is None and no boundary fixes the pressure level
        (pressure outlet or farfield), the pressure of the cell nearest
        ``pressure_point`` is pinned to ``pin_pressure`` (D39 R2).
    boundary_conditions : dict of str to dict
        See :func:`aerojax.physics.incompressible.boundary_data`. Periodic
        pairs are given as ``{"type": "periodic", "partner": name,
        "translation": (dx, dy)}`` on one of the two boundaries.
    weight_exponent : float
        Least-squares weight exponent (D22).
    pressure_point : tuple of float, optional
        Location of the pinned cell; default the domain centroid.
    pin_pressure : float
        Pinned pressure value.
    """

    def __init__(
        self,
        mesh,
        model,
        boundary_conditions,
        weight_exponent=1.0,
        pressure_point=None,
        pin_pressure=0.0,
    ):
        periodic, regular = [], {}
        for name, bc in boundary_conditions.items():
            if bc["type"] == "periodic":
                periodic.append((name, bc["partner"], np.asarray(bc["translation"])))
            else:
                regular[name] = bc
        for name_a, name_b, _ in periodic:
            if name_b in regular:
                raise ValueError(
                    f"periodic partner '{name_b}' of '{name_a}' must not have its "
                    "own boundary condition"
                )
        missing = set(mesh.boundary_faces) - set(regular) - {b for _, b, _ in periodic}
        missing -= {a for a, _, _ in periodic}
        unknown = set(regular) - set(mesh.boundary_faces)
        if missing or unknown:
            raise ValueError(
                f"boundary conditions missing for {sorted(missing)}, "
                f"unknown boundaries {sorted(unknown)}"
            )
        self.mesh = mesh
        self.disc = build_geometry(
            mesh, weight_exponent, periodic, inc.lsq_include(regular)
        )
        self.geom = self.disc.geom
        self.bc = inc.boundary_data(self.disc, regular, model.density)
        self.conditions = regular
        self.periodic = periodic
        self.check_mass_balance()

        fixes_pressure = any(bc["type"] in PRESSURE_FIXING for bc in regular.values())
        self.pinned_cell = None
        if model.pin is None and not fixes_pressure:
            point = (
                np.average(mesh.cell_centroid, axis=0, weights=mesh.cell_area)
                if pressure_point is None
                else np.asarray(pressure_point, dtype=float)
            )
            cell = int(np.argmin(np.sum((mesh.cell_centroid - point) ** 2, axis=1)))
            model = model._replace(pin=(cell, float(pin_pressure)))
            self.pinned_cell = cell
        self.model = model
        self._fi = partial(inc.interior_flux, model)
        self._fb = partial(inc.boundary_flux, model)

    # -- setup checks -------------------------------------------------------

    def check_mass_balance(self, tolerance=1e-10):
        """Reject a prescribed net inflow without a pressure boundary (D39 R3)."""
        if any(bc["type"] in PRESSURE_FIXING for bc in self.conditions.values()):
            return
        v = np.asarray(self.bc.value)[:, 1:]
        n = np.asarray(self.geom.b_normal)
        flux = np.sum(np.einsum("fk,fk->f", v, n) * np.asarray(self.geom.b_length))
        scale = np.sum(
            np.abs(np.einsum("fk,fk->f", v, n)) * np.asarray(self.geom.b_length)
        )
        if scale > 0.0 and abs(flux) > tolerance * scale:
            raise ValueError(
                f"net inflow {flux:.3e} through velocity boundaries, but no boundary "
                "fixes the pressure level (mass cannot be absorbed)"
            )

    # -- solver interface ---------------------------------------------------

    def beta2(self, w):
        """Artificial compressibility beta^2 for the current state."""
        return inc.beta_squared(self.model, w, self.bc)

    def residual(self, w, beta2, order=None, psi=None):
        """R(w) for the current beta^2, spatial order and frozen limiter."""
        order = self.model.spatial_order if order is None else order
        return np.asarray(
            inc.residual(
                self.model, self.geom, self.bc, jnp.asarray(w), order, beta2, psi
            )
        )

    def jvp(self, w, dw, beta2, psi=None):
        """Exact dR/dw @ dw of the second-order residual (``jax.jvp``)."""
        return np.asarray(
            inc.residual_jvp(
                self.model,
                self.geom,
                self.bc,
                jnp.asarray(w),
                jnp.asarray(dw),
                self.model.spatial_order,
                beta2,
                psi,
            )
        )

    def limiter(self, w):
        """Limiter values of the state ``w``, shape (n_cells, 3)."""
        return inc.limiter(self.model, self.geom, self.bc, jnp.asarray(w))

    def jacobian(self, w, beta2):
        """First-order Jacobian, with the pinned row replaced if any."""
        j = first_order_jacobian(
            self.geom,
            self._fi,
            self._fb,
            jnp.asarray(w),
            params=beta2,
            b_args=(self.bc.kind, self.bc.value),
        )
        if self.model.pin is not None:
            j = replace_rows(j, [self.model.pin[0] * 3 + inc.P])
        return j

    def time_term(self, w, beta2, cfl):
        """Block-diagonal pseudo-time matrix Gamma_i Lambda_i / CFL."""
        return block_diagonal(
            inc.time_term_blocks(self.model, self.geom, w, beta2, cfl)
        )

    def explicit_update(self, w, r, beta2, cfl):
        """Explicit increment CFL / Lambda_i * Gamma_i^-1 R_i."""
        return np.asarray(
            inc.explicit_update(
                self.model, self.geom, jnp.asarray(w), jnp.asarray(r), beta2, cfl
            )
        )

    # -- loads --------------------------------------------------------------

    def wall_loads(self, w):
        """Pressure and viscous traction on every boundary face."""
        return fz.wall_loads(self.model, self.geom, self.bc, jnp.asarray(w))

    def boundary_faces(self, names):
        """Boundary-face indices (``b_*`` order) of the named boundaries."""
        idx = [
            np.arange(
                self.disc.boundary_slices[n].start, self.disc.boundary_slices[n].stop
            )
            for n in names
        ]
        return np.concatenate(idx) if idx else np.zeros(0, dtype=np.int64)

    def forces(self, w, boundaries=None, moment_center=(0.0, 0.0), loads=None):
        """Integrated force and moment on the given boundaries (default: all walls).

        Returns
        -------
        aerojax.physics.forces.Forces
        """
        if boundaries is None:
            boundaries = [
                n
                for n, bc in self.conditions.items()
                if bc["type"] in ("wall", "transpiration-wall")
            ]
        loads = self.wall_loads(w) if loads is None else loads
        return fz.integrate(
            loads, self.geom, self.boundary_faces(boundaries), moment_center
        )

    def initial_state(self, velocity=(0.0, 0.0), pressure=0.0):
        """Uniform initial field, shape (n_cells, 3)."""
        w = np.zeros((self.mesh.n_cells, 3))
        w[:, inc.P] = pressure
        w[:, inc.U] = velocity[0]
        w[:, inc.V] = velocity[1]
        return w

    def solve(
        self,
        w0=None,
        settings=None,
        callback=None,
        freeze_limiter_after_iterations=None,
        freeze_limiter_after_residual_drop=None,
    ):
        """Implicit pseudo-time solution (D25, D35).

        beta^2 is updated from the current state before every residual
        evaluation, as in SU2. The limiter is frozen (D23) once either freeze
        condition is met; afterwards the residual, its Jacobian-vector
        products and the solution use the limiter values of that state.
        """
        w0 = self.initial_state() if w0 is None else w0
        state = {"beta2": self.beta2(w0), "psi": None, "norm0": None}
        uses_limiter = self.model.limiter != "none" and self.model.spatial_order == 2

        def residual(w):
            state["beta2"] = self.beta2(w)
            return self.residual(w, state["beta2"], psi=state["psi"])

        def residual_startup(w):
            state["beta2"] = self.beta2(w)
            return self.residual(w, state["beta2"], order=1)

        def on_iteration(it, w, residuals, cfl, accepted):
            norm = float(np.sqrt(np.mean(np.square(residuals))))
            if state["norm0"] is None:
                state["norm0"] = max(norm, np.finfo(float).tiny)
            if uses_limiter and state["psi"] is None and accepted:
                drop = np.log10(state["norm0"] / max(norm, np.finfo(float).tiny))
                if (
                    freeze_limiter_after_iterations is not None
                    and it >= freeze_limiter_after_iterations
                ) or (
                    freeze_limiter_after_residual_drop is not None
                    and drop >= freeze_limiter_after_residual_drop
                ):
                    state["psi"] = self.limiter(w)
                    if callback is not None and hasattr(callback, "info"):
                        callback.info["note"] = "limiter frozen"
            if callback is not None:
                callback(it, w, residuals, cfl, accepted)

        on_iteration.info = getattr(callback, "info", {})
        if callback is not None and not hasattr(callback, "info"):
            callback.info = on_iteration.info

        return solve_steady(
            residual,
            lambda w: self.jacobian(w, state["beta2"]),
            lambda w, cfl: self.time_term(w, state["beta2"], cfl),
            w0,
            settings or PseudoTimeSettings(),
            residual_startup=residual_startup,
            callback=on_iteration,
            jvp=lambda w, dw: self.jvp(w, dw, state["beta2"], state["psi"]),
        )

    def mass(self):
        """Diagonal of the physical-time mass matrix: rho * area on u, v."""
        area = np.asarray(self.geom.area)
        return area[:, None] * np.array([0.0, self.model.density, self.model.density])

    def solve_unsteady(self, w0, settings, callback=None, inner_callback=None):
        """Dual time stepping from ``w0`` to ``settings.end_time`` (D9, D37).

        Every time step solves the steady residual plus the BDF time term
        with the inner pseudo-time/Newton-Krylov iterations of
        ``settings.inner``; beta^2 follows the current state, the limiter is
        not frozen.
        """
        state = {"beta2": self.beta2(w0)}

        def residual(w):
            state["beta2"] = self.beta2(w)
            return self.residual(w, state["beta2"])

        return solve_unsteady(
            residual,
            lambda w: self.jacobian(w, state["beta2"]),
            lambda w, cfl: self.time_term(w, state["beta2"], cfl),
            self.mass(),
            w0,
            settings,
            callback=callback,
            inner_callback=inner_callback,
            jvp=lambda w, dw: self.jvp(w, dw, state["beta2"]),
        )

    def solve_explicit(
        self, w0=None, cfl=0.8, max_iterations=20000, residual_drop=8.0, callback=None
    ):
        """Explicit Runge-Kutta reference mode (D25)."""
        w0 = self.initial_state() if w0 is None else w0
        state = {"beta2": self.beta2(w0)}

        def residual(w):
            state["beta2"] = self.beta2(w)
            return self.residual(w, state["beta2"])

        return solve_explicit(
            residual,
            lambda w, r, cfl: self.explicit_update(w, r, state["beta2"], cfl),
            w0,
            cfl=cfl,
            max_iterations=max_iterations,
            residual_drop=residual_drop,
            callback=callback,
        )
