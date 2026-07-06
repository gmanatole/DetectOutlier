# OutlierDetect

OutlierDetect is a Python package for local CTD-SRDL-style quality control, outlier detection, and constrained reconstruction of sparse ocean profiles.

It combines:

- TEOS-10 / GSW density calculations for the actual inference path,
- a Gaussian nuisance prior/posterior for local T/S correction,
- a small transformer model for profile-level and point-level QC,
- synthetic training data built from clean Argo or EN4 profiles,
- reconstruction plots and JSON sidecars for inspection.

The full documentation lives in Sphinx/MyST under [docs/](docs/index.md).

## Install

```bash
pip install -e .
```

For training, prediction, Argo NetCDF I/O, and docs:

```bash
pip install -e ".[train,io,docs]"
```

## Quickstart

Run the heuristic predictor on a profile you already have in memory:

```python
import numpy as np
from outlierdetect import Heuristic, ProfileInput

profile = ProfileInput(
    pressure=np.array([5, 20, 45, 80, 130, 210, 330, 500], dtype=float),
    temperature=np.array([5.6, 5.4, 5.1, 4.8, 4.4, 3.8, 3.0, 2.3]),
    salinity=np.array([34.10, 34.12, 34.15, 34.20, 34.28, 34.40, 34.56, 34.70]),
    residual_t=np.array([0.12, 0.10, 0.11, 0.09, 0.12, 0.08, 0.10, 0.09]),
    residual_s=np.array([0.08, 0.08, 0.09, 0.08, 0.09, 0.08, 0.09, 0.08]),
    sigma_t=np.full(8, 0.25),
    sigma_s=np.full(8, 0.04),
    sigma_vert=np.full(8, 40.0),
    day_of_year=220,
    profile_id="example_profile",
)

result = Heuristic().predict(profile)
print(result.summary())
```

Train from Argo NetCDF data:

```bash
outlierdetect-train --config outlierdetect.toml --train-root C:\data\argo --test-root C:\data\argo_test
```

If you omit `--test-root`, the command uses `--val-fraction` to split the training root into train and validation subsets.
When you do pass `--test-root`, the held-out side is built from raw `TEMP`/`PSAL` values where available and bypasses synthetic augmentation unless you also pass `--test-augment`.
With `--test-augment`, the held-out side uses adjusted/corrected values and the same synthetic corruption pipeline as training.

Train from EN4 monthly NetCDF data:

```bash
outlierdetect-train --data-source en4 --train-root C:\data\en4 --config outlierdetect.toml
```

Predict on a dataset:

```bash
outlierdetect-predict --config outlierdetect.toml
```

Export raw Argo profiles to parquet:

```bash
outlier-detect --raw-to-parquet --input C:\data\argo --output C:\data\argo.parquet
```

## Documentation

The Sphinx tree is the authoritative reference for the codebase:

- [Project overview](docs/index.md)
- [Physical model](docs/physics.md)
- [API reference](docs/api/index.md)

Build it locally with:

```bash
sphinx-build -b html docs docs/_build/html
```

## Configuration

The starter config file is [outlierdetect.toml](outlierdetect.toml). It covers data roots, checkpoint paths, sigma-vert loading for heave lookup, and the main training and prediction toggles.

CLI flags override TOML values, and each run writes the resolved configuration into its output directory.

You can manage a user-owned config file with:

```bash
outlierdetect config init
outlierdetect config show
outlierdetect config validate --config path\to\outlierdetect.toml
```
