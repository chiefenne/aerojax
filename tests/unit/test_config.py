"""Case-file validation and resolution (D27-D31, D38, D39)."""

import tomllib

import pytest

from aerojax.config import ConfigError, load_case, resolve

MINIMAL = {
    "mesh": {"file": "m.su2"},
    "physics": {"model": "laminar"},
    "fluid": {"density": 1.0, "viscosity": 0.1},
    "reference": {"velocity": 1.0},
    "initial": {"velocity": [0.0, 0.0]},
    "boundary": {
        "bottom": {"type": "wall"},
        "top": {"type": "wall", "velocity": [1.0, 0.0]},
        "left": {"type": "periodic", "partner": "right", "translation": [2.0, 0.0]},
    },
}


def test_defaults_and_explicit_tracking():
    cfg = resolve(MINIMAL, "couette.toml")
    assert cfg.name == "couette"
    assert cfg["numerics"]["limiter"] == "venkatakrishnan-wang"
    assert cfg["solver"]["cfl"]["start"] == 10.0
    assert "fluid.viscosity" in cfg.explicit
    assert "numerics.limiter" not in cfg.explicit
    assert cfg["boundary"]["bottom"]["velocity"] == [0.0, 0.0]
    assert any("pressure pinned" in n for n in cfg.notices)


def test_resolved_toml_round_trip():
    cfg = resolve(MINIMAL, "couette.toml")
    again = resolve(tomllib.loads(cfg.to_toml()), "couette.toml")
    assert again.data == cfg.data


def problems(raw):
    with pytest.raises(ConfigError) as err:
        resolve(raw)
    return str(err.value)


def test_unknown_key_suggestion():
    raw = {**MINIMAL, "numerics": {"limitr": "none"}}
    msg = problems(raw)
    assert "unknown key 'numerics.limitr'" in msg and "numerics.limiter" in msg


def test_unknown_section_and_boundary_type():
    msg = problems({**MINIMAL, "numerix": {}})
    assert "unknown section 'numerix'" in msg and "numerics" in msg
    raw = {**MINIMAL, "boundary": {**MINIMAL["boundary"], "top": {"type": "wal"}}}
    assert "did you mean 'wall'" in problems(raw)


def test_planned_keys_are_named():
    msg = problems({**MINIMAL, "solver": {"newton_krylov": {"enabled": True}}})
    assert "planned for stage 3" in msg
    raw = {
        **MINIMAL,
        "boundary": {**MINIMAL["boundary"], "top": {"type": "transpiration-wall"}},
    }
    assert "planned for stage 3" in problems(raw)


def test_required_and_types():
    raw = {k: v for k, v in MINIMAL.items() if k != "mesh"}
    assert "mesh.file is required" in problems(raw)
    raw = {**MINIMAL, "fluid": {"density": "one", "viscosity": 0.1}}
    assert "fluid.density: expected float" in problems(raw)
    raw = {**MINIMAL, "numerics": {"spatial_order": 3}}
    assert "numerics.spatial_order: 3 is not one of [1, 2]" in problems(raw)
    raw = {**MINIMAL, "initial": {"velocity": [1.0]}}
    assert "initial.velocity: expected array of two numbers" in problems(raw)


def test_viscosity_xor_reynolds():
    raw = {**MINIMAL, "fluid": {"density": 1.0}}
    assert "exactly one of viscosity and reynolds_number" in problems(raw)
    raw = {**MINIMAL, "fluid": {"density": 1.0, "reynolds_number": 100.0}}
    assert "reference.length is required" in problems(raw)
    raw["reference"] = {"velocity": 1.0, "length": 1.0}
    assert resolve(raw)["fluid"]["reynolds_number"] == 100.0


def test_periodic_partner_rules():
    raw = {**MINIMAL, "boundary": {**MINIMAL["boundary"], "right": {"type": "wall"}}}
    assert "must not have its own entry" in problems(raw)


def test_farfield_supplies_reference_and_initial_state():
    raw = {
        "mesh": {"file": "m.su2"},
        "physics": {"model": "laminar"},
        "fluid": {"density": 1.225, "viscosity": 1.8e-5},
        "boundary": {
            "airfoil": {"type": "wall"},
            "outer": {"type": "farfield", "velocity": 10.0, "angle_of_attack": 4.0},
        },
    }
    cfg = resolve(raw)
    assert cfg["reference"]["velocity"] == 10.0
    assert cfg["initial"]["from_boundary"] == "outer"
    assert cfg["reference"]["density"] == 1.225
    assert not any("pinned" in n for n in cfg.notices)


def test_load_case_reports_toml_syntax_errors(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text("[mesh\nfile = 'x'\n")
    with pytest.raises(ConfigError, match="bad.toml"):
        load_case(path)
