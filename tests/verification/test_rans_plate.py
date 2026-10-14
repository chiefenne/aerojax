"""R1: turbulent flat plate (NASA TMR domain), SA-neg vs the TMR values."""

import pytest

from aerojax.cases import tmr_flat_plate as tp


def test_turbulent_plate_coarse():
    """48 x 48 cells: C_f(0.97) +1.2 %, C_D +2 % vs the TMR SA values."""
    r = tp.solve(48, 12, 48)
    assert r.converged and r.y_plus_max < 1.0
    assert r.cf_097 == pytest.approx(r.reference["cf_097"], rel=0.025)
    assert r.cd == pytest.approx(r.reference["cd"], rel=0.03)
    # Fully turbulent: no laminar Cf anywhere behind the leading-edge region.
    assert r.cf[r.x > 0.2].min() > 0.002


def test_turbulent_plate_coupled_matches_segregated():
    seg = tp.solve(24, 6, 32)
    cpl = tp.solve(24, 6, 32, coupling="coupled")
    assert cpl.converged
    assert cpl.cf_097 == pytest.approx(seg.cf_097, rel=1e-4)


@pytest.mark.slow
@pytest.mark.parametrize(("nxp", "nxf", "ny"), [(96, 24, 64), (192, 48, 96)])
def test_turbulent_plate_refined(nxp, nxf, ny):
    """7.7k / 23k cells: C_f(0.97) +1.0 / +0.9 %, C_D +1.7 / +1.4 %."""
    r = tp.solve(nxp, nxf, ny)
    assert r.converged and r.y_plus_max < 1.0
    assert r.cf_097 == pytest.approx(r.reference["cf_097"], rel=0.015)
    assert r.cd == pytest.approx(r.reference["cd"], rel=0.025)
