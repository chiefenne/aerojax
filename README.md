# aerojax

2D unstructured cell-centred finite-volume Navier–Stokes/RANS solver in
Python/JAX with SU2-based numerics, developed as a companion to
[PyAero](https://github.com/chiefenne/PyAero).

**Status:** stage 1 of the [stage plan](docs/requirements.md#7-stage-plan-d53):
mesh foundations, finite-volume operators and the implicit solver machinery,
verified on a scalar convection–diffusion model problem. aerojax does not solve
flows yet.

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
pytest -m slow                 # order-of-accuracy studies, benchmarks
python examples/stage0_meshes.py
python examples/stage1_convection_diffusion.py
```

The examples write VTU files (for ParaView) and plots to `examples/output/`.

## License

MIT, see [LICENSE](LICENSE).
