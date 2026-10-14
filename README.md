# aerojax

2D unstructured cell-centred finite-volume Navier–Stokes/RANS solver in
Python/JAX, developed as a companion to
[PyAero](https://github.com/chiefenne/PyAero).

**Status:** stage 3 of the [stage plan](docs/requirements.md#7-stage-plan-d53):
steady laminar and inviscid incompressible flow with the SU2 INC
formulation and a Newton–Krylov solver, verified on Couette and Poiseuille
flow, the lid-driven cavity (Ghia et al.), the inviscid cylinder, the
Blasius flat plate and the DFG cylinder-in-channel benchmark. Unsteady flow
(stage 4) and turbulence (stage 5) follow.

## Documentation

- [Requirements](docs/requirements.md): decisions, planned features, stage plan
- [Case configuration](docs/configuration.md): TOML case-file reference

## Installation

aerojax needs a conda environment from conda-forge. On Intel Macs this is the
only source of `jaxlib`.

```
conda env create -f environment.yml
conda activate aerojax
pip install -e . --no-deps
```

## Running a case

```
aerojax check case.toml        # validate, print the resolved configuration
aerojax run case.toml          # solve; writes <name>.log, .history.csv, .vtu
```

Example case files are in `examples/cases/` (generate their meshes first with
`python examples/cases/make_meshes.py`). The case-file reference is
[docs/configuration.md](docs/configuration.md); the same setup is available
from Python through `aerojax.driver.IncompressibleProblem`.

## Tests and examples

```
pytest                         # fast tests
pytest -m slow                 # order-of-accuracy studies, benchmarks
python examples/stage0_meshes.py
python examples/stage1_convection_diffusion.py
python examples/stage2_laminar.py [--re1000]
python examples/stage3_benchmarks.py
```

The examples write VTU files (for ParaView) and plots to `examples/output/`.

## License

MIT, see [LICENSE](LICENSE).
