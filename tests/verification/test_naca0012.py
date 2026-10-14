"""R2: NACA 0012 at Re 6e6 (NASA TMR 2DN00), SA-neg on the generated O-grid."""

import numpy as np
import pytest

from aerojax.cases import naca0012
from aerojax.mesh import airfoil_points, naca0012_ogrid
from aerojax.mesh import naca0012 as thickness


def test_airfoil_geometry_and_ogrid():
    pts = airfoil_points(64)
    assert pts.shape == (64 + 3, 2)  # 64 surface nodes + 3 base nodes
    assert np.isclose(pts[:, 0].max(), 1.0) and np.isclose(pts[:, 0].min(), 0.0)
    assert thickness(0.3) == pytest.approx(0.0595, abs=5e-4)  # 12 % thick (TMR: 11.9 %)
    with pytest.raises(ValueError, match="growth ratio"):
        naca0012_ogrid(64, 32, farfield_radius=50.0)
    mesh = naca0012_ogrid(64, 32, reynolds=1.0e4, farfield_radius=50.0)
    assert mesh.n_cells == 67 * 32 and mesh.cell_area.min() > 0.0
    far = mesh.nodes[-67:]
    radius = np.linalg.norm(far - [0.5, 0.0], axis=1)
    assert 0.95 * 50.0 < radius.min() and radius.max() < 1.05 * 50.0
    assert set(mesh.boundary_faces) == {"airfoil", "farfield"}


def test_naca0012_alpha10_coarse():
    """128 x 64 O-grid, 500 chords: C_L -7 % (coarse farfield/wake resolution)."""
    r = naca0012.solve(10.0, 128, 64)
    assert r.converged and r.y_plus_max < 2.0
    assert r.cl == pytest.approx(r.reference["cl"], rel=0.10)
    assert r.cd_viscous == pytest.approx(0.0057, abs=0.001)
    assert r.cp.min() < -3.0  # suction peak


@pytest.mark.slow
@pytest.mark.parametrize("alpha", [0.0, 10.0, 15.0])
def test_naca0012_medium(alpha):
    """256 x 128 (33k cells): C_L within 2 % (-0.008, 1.076, 1.536); the drag
    is 13-45 % high, dominated by the pressure drag on the coarse O-grid wake."""
    r = naca0012.solve(alpha, 256, 128)
    # The first spacing is sized for C_f = 0.003; at the 15-degree suction
    # peak C_f is twice that and y+ reaches 2.2.
    assert r.converged and r.y_plus_max < (3.0 if alpha == 15.0 else 2.0)
    ref = r.reference
    assert r.cl == pytest.approx(ref["cl"], abs=0.015 + 0.03 * abs(ref["cl"]))
    assert r.cd_viscous == pytest.approx(
        {0.0: 0.0069, 10.0: 0.0062, 15.0: 0.0051}[alpha], rel=0.1
    )
    assert ref["cd"] < r.cd < 1.6 * ref["cd"]
