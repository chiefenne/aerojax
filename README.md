# aerojax

2D unstructured cell-centred finite-volume Navier–Stokes/RANS solver in
Python/JAX with SU2-based numerics, developed as a companion to
[PyAero](https://github.com/chiefenne/PyAero).

**Status:** stage 0 of the [stage plan](docs/requirements.md#7-stage-plan-d53),
which covers the mesh foundations. aerojax does not solve flows yet.

## Documentation

- [Requirements](docs/requirements.md): decisions, planned features, stage plan
- [Case configuration](docs/configuration.md): TOML case-file reference (draft)

## Installation

aerojax needs a conda environment from conda-forge. On Intel Macs this is the
only source of `jaxlib`.

```
conda env create -f environment.yml
conda activate aerojax
pip install -e . --no-deps
```

## Tests and example

```
pytest                         # fast tests
pytest -m slow                 # long benchmark cases (later stages)
python examples/stage0_meshes.py
```

The example writes VTU files (for ParaView) and a mesh plot to
`examples/output/`.

## License

MIT, see [LICENSE](LICENSE).
