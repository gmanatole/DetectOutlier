# OutlierDetect

OutlierDetect is a profile quality-control and reconstruction package for sparse CTD-SRDL-style ocean profiles. The documentation in this tree is the source of truth for the codebase. It is written in MyST Markdown and built with Sphinx so the rationale, physics, and API reference live in one place.

The package is organized around three things:

1. local quality control on sparse temperature/salinity profiles,
2. a physical correction model for nuisance bias and density consistency,
3. a reconstruction path that is constrained by the same ocean physics used during inference.

The key design choice is that the code does not treat the ocean as a generic tabular problem. It uses TEOS-10 density, local pressure geometry, and a constrained nuisance model because those are the quantities that matter for static stability and for the local correction problem the model is trying to solve.

## Build

Install the docs dependencies and build the HTML tree with:

```bash
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
```

The runtime package itself installs with:

```bash
pip install -e .
```

## Read This First

```{toctree}
:maxdepth: 2
:caption: Guide

configuration
physics
api/index
```

## Package Layout

The implementation is split into small modules so the physical assumptions remain visible:

- `outlierdetect.density`: TEOS-10 density, static stability, and salinity-only projection.
- `outlierdetect.corrections`: Gaussian nuisance prior/posterior for local T/S bias.
- `outlierdetect.features`: per-level features built for the neural and heuristic paths.
- `outlierdetect.tool`: prediction wrappers that combine point flags, nuisance correction, and reconstruction.
- `outlierdetect.training`: synthetic data generation, losses, and training utilities.

## Why This Structure

The project started as a compact, local QC tool and then grew a Bayesian nuisance-correction layer and a neural model around the same physical constraints. Keeping the code modular makes that history explicit: density lives in one module, nuisance priors in another, and the learning code only consumes the derived quantities it needs.
