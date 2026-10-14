"""Reading, validating and resolving a TOML case file (D27-D33, D38, D39)."""

import difflib
import tomllib
from pathlib import Path

import numpy as np

from aerojax.config.schema import (
    BOUNDARY_TYPES,
    PLANNED,
    PLANNED_BOUNDARY_TYPES,
    REQUIRED,
    SCHEMA,
    Opt,
)

PRESSURE_FIXING = {"pressure-outlet", "farfield"}


class ConfigError(ValueError):
    """Invalid case configuration; the message lists all problems found."""


class Resolved:
    """A validated configuration with all defaults filled in.

    Attributes
    ----------
    data : dict
        Nested values, ``data[section][key]`` and ``data["boundary"][name][key]``.
    explicit : set of str
        Dotted paths of the keys the user set.
    path : Path
        Case file (for relative paths).
    notices : list of str
        Informational messages produced while resolving.
    """

    def __init__(self, data, explicit, path, notices):
        self.data = data
        self.explicit = explicit
        self.path = Path(path)
        self.notices = notices

    def __getitem__(self, section):
        """Return the values of a section."""
        return self.data[section]

    @property
    def name(self):
        """Case name."""
        return self.data["case"]["name"]

    def resolve_path(self, relative):
        """Return an absolute path for one given relative to the case file."""
        return (self.path.parent / relative).resolve()

    def to_toml(self):
        """Serialise with a comment per key: description, unit, set/default."""
        lines = [f"# aerojax resolved configuration of {self.path.name}", ""]
        for section, options in SCHEMA.items():
            lines.append(f"[{section}]")
            lines += self._table_lines(section, options, self.data[section])
            lines.append("")
        for name, values in self.data["boundary"].items():
            lines.append(f'[boundary."{name}"]')
            opts = {"type": Opt("string", REQUIRED, "Boundary-condition type")}
            opts.update(BOUNDARY_TYPES[values["type"]])
            lines += self._table_lines(f"boundary.{name}", opts, values)
            lines.append("")
        return "\n".join(lines)

    def _table_lines(self, prefix, options, values):
        lines = []
        for key, opt in options.items():
            if isinstance(opt, dict):
                lines.append("")
                lines.append(f"[{prefix}.{key}]")
                lines += self._table_lines(f"{prefix}.{key}", opt, values[key])
                continue
            value = values.get(key)
            if value is None:
                continue
            origin = "set" if f"{prefix}.{key}" in self.explicit else "default"
            unit = f" [{opt.unit}]" if opt.unit else ""
            lines.append(
                f"{key} = {_toml_value(value)}  # {opt.description}{unit} ({origin})"
            )
        return lines


def load_case(path):
    """Read and validate a case file.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    Resolved

    Raises
    ------
    ConfigError
        With every problem found, one per line.
    """
    path = Path(path)
    try:
        raw = tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: {exc}") from exc
    return resolve(raw, path)


def resolve(raw, path="case.toml"):
    """Validate a parsed case dictionary and fill in defaults."""
    problems, notices, explicit = [], [], set()
    data = {}
    known_sections = set(SCHEMA) | {"boundary"}
    for section in raw:
        if section not in known_sections:
            problems.append(_unknown("section", section, known_sections))
    for section, options in SCHEMA.items():
        values = raw.get(section, {})
        if not isinstance(values, dict):
            problems.append(f"[{section}] must be a table")
            values = {}
        data[section] = _resolve_table(section, options, values, explicit, problems)

    data["boundary"] = {}
    boundaries = raw.get("boundary", {})
    if not isinstance(boundaries, dict) or any(
        not isinstance(v, dict) for v in boundaries.values()
    ):
        problems.append("[boundary.<name>] tables expected")
        boundaries = {}
    for name, values in boundaries.items():
        data["boundary"][name] = _resolve_boundary(name, values, explicit, problems)

    if problems:
        raise ConfigError("invalid case configuration:\n  " + "\n  ".join(problems))
    _consistency(data, explicit, problems, notices, path)
    if problems:
        raise ConfigError("invalid case configuration:\n  " + "\n  ".join(problems))
    if data["case"]["name"] is None:
        data["case"]["name"] = Path(path).stem
    return Resolved(data, explicit, path, notices)


def _resolve_table(prefix, options, values, explicit, problems):
    out = {}
    for key in values:
        if key not in options:
            dotted = f"{prefix}.{key}"
            if dotted in PLANNED:
                problems.append(
                    f"{dotted}: not available yet (planned for {PLANNED[dotted]})"
                )
            else:
                problems.append(
                    _unknown("key", dotted, [f"{prefix}.{k}" for k in options])
                )
    for key, opt in options.items():
        dotted = f"{prefix}.{key}"
        if isinstance(opt, dict):
            sub = values.get(key, {})
            if not isinstance(sub, dict):
                problems.append(f"{dotted} must be a table")
                sub = {}
            out[key] = _resolve_table(dotted, opt, sub, explicit, problems)
            continue
        if key in values:
            explicit.add(dotted)
            out[key] = _check_value(dotted, opt, values[key], problems)
        elif opt.default is REQUIRED:
            problems.append(f"{dotted} is required")
            out[key] = None
        else:
            out[key] = (
                list(opt.default) if isinstance(opt.default, list) else opt.default
            )
    return out


def _resolve_boundary(name, values, explicit, problems):
    prefix = f"boundary.{name}"
    kind = values.get("type")
    if kind is None:
        problems.append(f"{prefix}.type is required")
        return {"type": None}
    if kind in PLANNED_BOUNDARY_TYPES:
        problems.append(
            f"{prefix}.type = {kind!r}: not available yet "
            f"(planned for {PLANNED_BOUNDARY_TYPES[kind]})"
        )
        return {"type": None}
    if kind not in BOUNDARY_TYPES:
        problems.append(_unknown(f"{prefix}.type", kind, BOUNDARY_TYPES))
        return {"type": None}
    explicit.add(f"{prefix}.type")
    options = BOUNDARY_TYPES[kind]
    rest = {k: v for k, v in values.items() if k != "type"}
    out = _resolve_table(prefix, options, rest, explicit, problems)
    out = {"type": kind, **out}
    return out


def _check_value(dotted, opt, value, problems):
    ok = True
    if opt.type == "float":
        ok = isinstance(value, int | float) and not isinstance(value, bool)
        value = float(value) if ok else value
    elif opt.type == "int":
        ok = isinstance(value, int) and not isinstance(value, bool)
    elif opt.type == "bool":
        ok = isinstance(value, bool)
    elif opt.type == "string":
        ok = isinstance(value, str)
    elif opt.type == "float2":
        ok = (
            isinstance(value, list)
            and len(value) == 2
            and all(
                isinstance(v, int | float) and not isinstance(v, bool) for v in value
            )
        )
        value = [float(v) for v in value] if ok else value
    elif opt.type == "string_list":
        ok = isinstance(value, list) and all(isinstance(v, str) for v in value)
    if not ok:
        kind = {"float2": "array of two numbers", "string_list": "array of strings"}
        problems.append(
            f"{dotted}: expected {kind.get(opt.type, opt.type)}, got {value!r}"
        )
        return value
    if opt.choices and value not in opt.choices:
        problems.append(f"{dotted}: {value!r} is not one of {list(opt.choices)}")
    if opt.minimum is not None and (
        value < opt.minimum if opt.type != "float2" else min(value) < opt.minimum
    ):
        problems.append(f"{dotted}: must be >= {opt.minimum}, got {value!r}")
    return value


def _consistency(data, explicit, problems, notices, path):
    """Cross-key rules (D31, D28, D39 R1/R5/R6, configuration.md)."""
    fluid, physics, ref = data["fluid"], data["physics"], data["reference"]
    boundaries = data["boundary"]
    types = {name: b["type"] for name, b in boundaries.items()}
    farfields = [n for n, t in types.items() if t == "farfield"]

    # Periodic pairs: partner exists and has no own condition.
    for name, b in boundaries.items():
        if b["type"] == "periodic":
            partner = b["partner"]
            if partner in boundaries:
                problems.append(
                    f"boundary.{partner}: is the periodic partner of '{name}' and "
                    "must not have its own entry"
                )
            if partner == name:
                problems.append(f"boundary.{name}: periodic partner is itself")

    # Viscosity: exactly one of viscosity / reynolds_number, unless Euler.
    has_mu, has_re = (
        fluid["viscosity"] is not None,
        fluid["reynolds_number"] is not None,
    )
    if physics["model"] == "euler":
        if has_mu or has_re:
            notices.append("physics.model = euler: fluid viscosity ignored")
    elif has_mu == has_re:
        problems.append(
            "fluid: exactly one of viscosity and reynolds_number is required"
        )

    # Reference velocity: from the farfield if there is exactly one.
    if ref["velocity"] is None:
        if len(farfields) == 1:
            ref["velocity"] = boundaries[farfields[0]]["velocity"]
            notices.append(
                f"reference.velocity taken from farfield boundary '{farfields[0]}'"
            )
        elif has_re or data["initial"]["from_boundary"] is None and not farfields:
            problems.append(
                "reference.velocity is required (no single farfield boundary to "
                "take it from)"
            )
    if has_re and "reference.length" not in explicit:
        problems.append("reference.length is required with fluid.reynolds_number")
    if ref["density"] is None:
        ref["density"] = fluid["density"]

    # Initial state: explicit values or from_boundary (default: the farfield).
    init = data["initial"]
    if init["from_boundary"] is not None and (
        init["velocity"] is not None or init["pressure"] is not None
    ):
        problems.append(
            "initial: give either from_boundary or explicit values, not both"
        )
    if init["from_boundary"] is None and init["velocity"] is None:
        if len(farfields) == 1:
            init["from_boundary"] = farfields[0]
            notices.append(
                f"initial state taken from farfield boundary '{farfields[0]}'"
            )
        else:
            problems.append(
                "[initial] is required: give velocity (and pressure) or from_boundary"
            )
    if init["from_boundary"] is not None:
        src = init["from_boundary"]
        if src not in boundaries:
            problems.append(_unknown("initial.from_boundary", src, boundaries))
        elif boundaries[src]["type"] not in ("farfield", "velocity-inlet"):
            problems.append(
                f"initial.from_boundary = '{src}': must be a farfield or velocity "
                "inlet boundary"
            )
    if init["pressure"] is None:
        init["pressure"] = 0.0

    # Pressure level and periodic driving force.
    if not any(t in PRESSURE_FIXING for t in types.values()):
        notices.append(
            "no boundary fixes the pressure level: pressure pinned to "
            "initial.pressure in the cell nearest reference.pressure_point"
        )
    if (
        physics["mean_pressure_gradient"] is not None
        and "periodic" not in types.values()
    ):
        notices.append("physics.mean_pressure_gradient set without periodic boundaries")

    # Euler: walls become slip walls (D39 R5).
    if physics["model"] == "euler":
        for name, t in types.items():
            if t == "wall":
                notices.append(
                    f"physics.model = euler: wall '{name}' treated as slip-wall"
                )

    if data["output"]["formats"] and set(data["output"]["formats"]) - {"vtu"}:
        problems.append("output.formats: only 'vtu' is available")


def _unknown(what, name, candidates):
    close = difflib.get_close_matches(str(name), [str(c) for c in candidates], n=1)
    hint = f" (did you mean '{close[0]}'?)" if close else ""
    return f"unknown {what} '{name}'{hint}"


def _toml_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return repr(float(value)) if isinstance(value, float) else str(value)
    if isinstance(value, str):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(value, list | tuple | np.ndarray):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    raise TypeError(f"cannot serialise {value!r}")
