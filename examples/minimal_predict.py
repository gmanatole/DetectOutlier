"""Minimal profile QC inference example.

Run from the repository root after installation:

    python examples/minimal_predict.py
"""

import json

import numpy as np

from outlierdetect import Heuristic, ProfileInput

pressure = np.array([5, 20, 45, 80, 130, 210, 330, 500], dtype=float)
temperature = np.array([5.6, 5.4, 5.1, 4.8, 4.4, 3.8, 3.0, 2.3])
salinity = np.array([34.10, 34.12, 34.15, 34.20, 34.28, 34.40, 34.56, 34.70])

# Example residuals against GLORYS or a local reference profile.
residual_t = np.array([0.12, 0.10, 0.11, 0.09, 0.12, 0.08, 0.10, 0.09])
residual_s = np.array([0.08, 0.08, 0.09, 0.08, 0.09, 0.08, 0.09, 0.08])

profile = ProfileInput(
    pressure=pressure,
    temperature=temperature,
    salinity=salinity,
    residual_t=residual_t,
    residual_s=residual_s,
    sigma_t=np.full_like(pressure, 0.25),
    sigma_s=np.full_like(pressure, 0.04),
    sigma_vert=np.full_like(pressure, 40.0),
    day_of_year=220,
    profile_id="example_profile",
)

result = Heuristic().predict(profile)
print(json.dumps(result.summary(), indent=2, sort_keys=True))
