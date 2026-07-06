"""Synthetic degradation utilities for Tool 1.

These functions convert trusted high-resolution profiles into sparse CTD-SRDL-like
training examples with known labels. The aim is to train Tool 1 to reject spikes
and unstable artifacts while remaining tolerant of large coherent T/S biases.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.random import Generator
from numpy.typing import ArrayLike, NDArray

from outlierdetect.climatology import sample_climatology_reference
from outlierdetect.data import ProfileInput
from outlierdetect.density import inversion_metrics
from outlierdetect.features import pressure_normalized
from outlierdetect.training.dataset import Example, Labels

FloatArray = NDArray[np.float64]


@dataclass(slots=True)
class SyntheticExample:
    """Synthetic profile plus true corruption parameters."""

    example: Example
    true_bias: dict[str, float]
    truth_pressure_grid: FloatArray
    truth_t: FloatArray
    truth_s: FloatArray


def degrade_highres_profile(
    pressure_hr: ArrayLike,
    temperature_hr: ArrayLike,
    salinity_hr: ArrayLike,
    *,
    rng: Generator | None = None,
    n_levels: int = 20,
    pressure_levels: ArrayLike | None = None,
    grid_size: int = 80,
    obs_noise_t: float = 0.01,
    obs_noise_s: float = 0.005,
    ref_noise_t: float = 0.08,
    ref_noise_s: float = 0.015,
    sigma_ocean_t: float = 0.25,
    sigma_ocean_s: float = 0.04,
    sigma_vert_range: tuple[float, float] = (15.0, 80.0),
    spike_probability: float = 0.08,
    pressure_error_probability: float = 0.03,
    global_bad_probability: float = 0.04,
    large_bias_probability: float = 0.50,
    latitude: float | None = None,
    longitude: float | None = None,
    day_of_year: float | None = None,
    profile_id: str | None = None,
    reference_source: bool | str | Path | None = None,
) -> SyntheticExample:
    """Create one sparse synthetic CTD-SRDL training example.

    The high-resolution input is treated as truth for the synthetic corruption
    process, but the profile residuals are scored against the ECCO monthly
    climatology when that reference file is available.
    """
    rng = np.random.default_rng() if rng is None else rng
    p_hr = np.asarray(pressure_hr, dtype=float)
    t_hr = np.asarray(temperature_hr, dtype=float)
    s_hr = np.asarray(salinity_hr, dtype=float)
    # Argo files occasionally contain negative pressure placeholders or bad levels.
    # Drop them up front so the synthetic training target stays physically valid.
    valid = np.isfinite(p_hr) & np.isfinite(t_hr) & np.isfinite(s_hr) & (p_hr >= 0.0)
    p_hr = p_hr[valid]
    t_hr = t_hr[valid]
    s_hr = s_hr[valid]
    order = np.argsort(p_hr)
    p_hr = p_hr[order]
    t_hr = t_hr[order]
    s_hr = s_hr[order]
    p_hr = _strictly_increasing_pressure(p_hr)
    if p_hr.size < 5:
        raise ValueError("High-resolution profile is too short for requested degradation.")

    if pressure_levels is not None:
        p_sparse = np.asarray(pressure_levels, dtype=float)
        p_sparse = p_sparse[np.isfinite(p_sparse)]
        if p_sparse.size < 2:
            raise ValueError("pressure_levels must contain at least two finite values.")
        p_sparse = np.clip(p_sparse, p_hr[0], p_hr[-1])
        p_sparse = np.unique(np.sort(p_sparse))
        n_levels = int(p_sparse.size)
    else:
        n_levels = min(n_levels, p_hr.size)
        # A rough CTD-SRDL-like sampling pattern: more levels in the upper ocean
        # but still covering most of the observed range.
        u = np.sort(rng.random(n_levels) ** 1.7)
        p_sparse = p_hr[0] + u * (p_hr[-1] - p_hr[0])
        p_sparse[0] = p_hr[0]
        p_sparse[-1] = p_hr[-1]
    t_truth = np.interp(p_sparse, p_hr, t_hr)
    s_truth = np.interp(p_sparse, p_hr, s_hr)

    p_norm = pressure_normalized(p_sparse)
    if rng.random() < large_bias_probability:
        a_t = rng.normal(0.0, 0.06)
        b_t = rng.normal(0.0, 0.04)
        a_s = rng.normal(0.0, 0.08)
        b_s = rng.normal(0.0, 0.04)
    else:
        a_t = rng.normal(0.0, 0.015)
        b_t = rng.normal(0.0, 0.01)
        a_s = rng.normal(0.0, 0.015)
        b_s = rng.normal(0.0, 0.008)

    t_obs = t_truth + a_t + b_t * p_norm + rng.normal(0.0, obs_noise_t, n_levels)
    s_obs = s_truth + a_s + b_s * p_norm + rng.normal(0.0, obs_noise_s, n_levels)

    point_bad_t = np.zeros(n_levels, dtype=float)
    point_bad_s = np.zeros(n_levels, dtype=float)

    for i in range(n_levels):
        if rng.random() < spike_probability:
            if rng.random() < 0.5:
                t_obs[i] += rng.normal(0.0, 1.0)
                point_bad_t[i] = 1.0
            else:
                s_obs[i] += rng.normal(0.0, 0.25)
                point_bad_s[i] = 1.0

    if rng.random() < pressure_error_probability and n_levels > 4:
        j = int(rng.integers(1, n_levels - 1))
        p_sparse[j] += rng.normal(0.0, 30.0)
        p_sparse = _strictly_increasing_pressure(p_sparse)
        point_bad_t[j] = 1.0
        point_bad_s[j] = 1.0

    profile_bad = 0.0
    if rng.random() < global_bad_probability:
        profile_bad = 1.0
        if rng.random() < 0.5:
            s_obs += rng.normal(0.0, 0.2, n_levels)
            point_bad_s[:] = np.maximum(point_bad_s, 0.5)
        else:
            t_obs += rng.normal(0.0, 1.5, n_levels)
            point_bad_t[:] = np.maximum(point_bad_t, 0.5)

    sigma_vert_value = rng.uniform(*sigma_vert_range)
    sigma_vert = np.full(n_levels, sigma_vert_value, dtype=float)

    if day_of_year is None:
        day_of_year = float(rng.uniform(0, 365))

    attrs = {"source": "synthetic_degradation"}
    if latitude is not None and np.isfinite(latitude):
        attrs["latitude"] = float(latitude)
    if longitude is not None and np.isfinite(longitude):
        attrs["longitude"] = float(longitude)

    reference_profile = ProfileInput(
        pressure=p_sparse,
        temperature=t_obs,
        salinity=s_obs,
        sigma_vert=sigma_vert,
        day_of_year=day_of_year,
        profile_id=profile_id,
        attrs={**attrs},
    )
    reference_sample = sample_climatology_reference(
        reference_profile,
        source=reference_source,
        sigma_vert=sigma_vert,
    )

    if reference_sample is not None:
        t_ref = reference_sample.reference_temperature
        s_ref = reference_sample.reference_salinity
        residual_t = reference_sample.reference_residual_t
        residual_s = reference_sample.reference_residual_s
        sigma_heave_t = reference_sample.reference_sigma_heave_t
        sigma_heave_s = reference_sample.reference_sigma_heave_s
        attrs.update(
            {
                "reference_source": str(reference_sample.source_path),
                "reference_month": int(reference_sample.month),
                "reference_latitude": float(reference_sample.latitude),
                "reference_longitude": float(reference_sample.longitude),
                "reference_latitude_index": int(reference_sample.latitude_index),
                "reference_longitude_index": int(reference_sample.longitude_index),
            }
        )
    else:
        # Legacy fallback: use the truth profile displaced vertically and add a small reference perturbation.
        heave = rng.normal(0.0, sigma_vert_value)
        p_ref_sample = np.clip(p_sparse + heave, p_hr[0], p_hr[-1])
        t_ref = np.interp(p_ref_sample, p_hr, t_hr) + rng.normal(0.0, ref_noise_t, n_levels)
        s_ref = np.interp(p_ref_sample, p_hr, s_hr) + rng.normal(0.0, ref_noise_s, n_levels)
        residual_t = t_obs - t_ref
        residual_s = s_obs - s_ref

        # Heave-induced uncertainty based on local truth gradients.
        dtdp_hr = np.gradient(t_hr, p_hr, edge_order=1)
        dsdp_hr = np.gradient(s_hr, p_hr, edge_order=1)
        sigma_heave_t = np.abs(np.interp(p_sparse, p_hr, dtdp_hr)) * sigma_vert_value
        sigma_heave_s = np.abs(np.interp(p_sparse, p_hr, dsdp_hr)) * sigma_vert_value

    if sigma_heave_t is None:
        sigma_heave_t = np.zeros(n_levels, dtype=float)
    if sigma_heave_s is None:
        sigma_heave_s = np.zeros(n_levels, dtype=float)

    sigma_t = np.sqrt(obs_noise_t**2 + ref_noise_t**2 + sigma_ocean_t**2 + sigma_heave_t**2)
    sigma_s = np.sqrt(obs_noise_s**2 + ref_noise_s**2 + sigma_ocean_s**2 + sigma_heave_s**2)

    lon = 0.0 if longitude is None or not np.isfinite(longitude) else float(longitude)
    lat = 0.0 if latitude is None or not np.isfinite(latitude) else float(latitude)
    inv = inversion_metrics(p_sparse, t_obs, s_obs, lon=lon, lat=lat)
    point_density = (np.asarray(inv["level_inversion_magnitude"], dtype=float) > 0.02).astype(float)

    profile = ProfileInput(
        pressure=p_sparse,
        temperature=t_obs,
        salinity=s_obs,
        residual_t=residual_t,
        residual_s=residual_s,
        sigma_t=sigma_t,
        sigma_s=sigma_s,
        sigma_vert=sigma_vert,
        sigma_heave_t=sigma_heave_t,
        sigma_heave_s=sigma_heave_s,
        reference_temperature=t_ref if reference_sample is not None else None,
        reference_salinity=s_ref if reference_sample is not None else None,
        reference_residual_t=residual_t if reference_sample is not None else None,
        reference_residual_s=residual_s if reference_sample is not None else None,
        reference_sigma_heave_t=sigma_heave_t if reference_sample is not None else None,
        reference_sigma_heave_s=sigma_heave_s if reference_sample is not None else None,
        rho_ts=np.full(n_levels, 0.6, dtype=float),
        day_of_year=day_of_year,
        profile_id=profile_id,
        attrs=attrs,
    )

    grid = np.linspace(p_hr[0], p_hr[-1], grid_size)
    truth_t_grid = np.interp(grid, p_hr, t_hr)
    truth_s_grid = np.interp(grid, p_hr, s_hr)
    labels = Labels(
        profile_bad=profile_bad,
        point_bad_t=point_bad_t,
        point_bad_s=point_bad_s,
        point_density_inconsistent=point_density,
        nuisance_mean=np.array([a_t, b_t, a_s, b_s], dtype=float),
        pressure_grid=grid,
        truth_t=truth_t_grid,
        truth_s=truth_s_grid,
    )
    return SyntheticExample(
        example=Example(profile=profile, labels=labels),
        true_bias={"a_t": a_t, "b_t": b_t, "a_s": a_s, "b_s": b_s},
        truth_pressure_grid=grid,
        truth_t=truth_t_grid,
        truth_s=truth_s_grid,
    )


def _strictly_increasing_pressure(pressure: ArrayLike, min_step: float = 1e-6) -> FloatArray:
    """Return a monotonically increasing copy of pressure.

    Sparse CTD profiles should already be monotonic, but synthetic pressure
    corruption can create duplicates. We nudge those levels by an epsilon so
    downstream gradient calculations stay numerically safe.
    """
    p = np.asarray(pressure, dtype=float).copy()
    if p.size < 2:
        return p
    step = float(min_step)
    if not np.isfinite(step) or step <= 0:
        step = 1e-6
    for i in range(1, p.size):
        if not np.isfinite(p[i - 1]):
            continue
        if not np.isfinite(p[i]) or p[i] <= p[i - 1]:
            p[i] = p[i - 1] + step
    return p
