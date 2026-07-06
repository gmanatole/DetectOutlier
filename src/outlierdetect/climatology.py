"""Monthly climatology reference lookup for sparse profiles.

The reference file built under ``data/`` stores monthly ECCO temperature and
salinity on a regular 0.5-degree grid. This module loads that reference and
extracts the closest month/latitude/longitude profile for an observed station.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from netCDF4 import Dataset

from .data import ProfileInput
from .density import profile_location_from_attrs

FloatArray = NDArray[np.float64]

DEFAULT_CLIMATOLOGY_PATH = Path(__file__).resolve().parents[2] / "data" / "ecco_monthly_climatology_0p5deg.nc"


@dataclass(slots=True)
class ClimatologyReference:
    """Cached metadata for the ECCO climatology file."""

    path: Path
    depth: FloatArray
    latitude: FloatArray
    longitude: FloatArray

    def sample(
        self,
        profile: ProfileInput,
        *,
        sigma_vert: ArrayLike | None = None,
    ) -> "ClimatologySample | None":
        month = _profile_month(profile)
        lat_obs, lon_obs = profile_location_from_attrs(profile.attrs)
        lat_idx = _nearest_index(self.latitude, lat_obs)
        lon_idx = _nearest_longitude_index(self.longitude, lon_obs)

        with Dataset(self.path, "r") as ds:
            theta = _read_column(ds.variables["THETA"], month - 1, lat_idx, lon_idx)
            salt = _read_column(ds.variables["SALT"], month - 1, lat_idx, lon_idx)

        if theta is None or salt is None:
            return None
        if not np.any(np.isfinite(theta)) or not np.any(np.isfinite(salt)):
            return None

        theta = _fill_missing_profile_values(theta, self.depth)
        salt = _fill_missing_profile_values(salt, self.depth)

        reference_t = np.interp(profile.pressure, self.depth, theta)
        reference_s = np.interp(profile.pressure, self.depth, salt)
        residual_t = np.asarray(profile.temperature, dtype=float) - reference_t
        residual_s = np.asarray(profile.salinity, dtype=float) - reference_s

        sigma_heave_t = None
        sigma_heave_s = None
        sigma_arr = _coerce_sigma_array(sigma_vert, profile.n_levels)
        if sigma_arr is not None:
            grad_t = _safe_gradient(reference_t, profile.pressure)
            grad_s = _safe_gradient(reference_s, profile.pressure)
            sigma_heave_t = np.abs(grad_t) * sigma_arr
            sigma_heave_s = np.abs(grad_s) * sigma_arr

        return ClimatologySample(
            reference_temperature=reference_t.astype(float),
            reference_salinity=reference_s.astype(float),
            reference_residual_t=residual_t.astype(float),
            reference_residual_s=residual_s.astype(float),
            reference_sigma_heave_t=None if sigma_heave_t is None else sigma_heave_t.astype(float),
            reference_sigma_heave_s=None if sigma_heave_s is None else sigma_heave_s.astype(float),
            month=month,
            latitude_index=lat_idx,
            longitude_index=lon_idx,
            latitude=float(self.latitude[lat_idx]),
            longitude=float(self.longitude[lon_idx]),
            source_path=self.path,
        )


@dataclass(slots=True)
class ClimatologySample:
    """Profile-sized reference values extracted from the climatology."""

    reference_temperature: FloatArray
    reference_salinity: FloatArray
    reference_residual_t: FloatArray
    reference_residual_s: FloatArray
    reference_sigma_heave_t: FloatArray | None
    reference_sigma_heave_s: FloatArray | None
    month: int
    latitude_index: int
    longitude_index: int
    latitude: float
    longitude: float
    source_path: Path


def resolve_climatology_source(source: bool | str | Path | None = None) -> Path | None:
    """Return a usable climatology path or ``None`` when the lookup is disabled."""

    if source is False:
        return None
    if source is True or source is None:
        path = DEFAULT_CLIMATOLOGY_PATH
    else:
        path = Path(source).expanduser()
        if not path.is_absolute():
            path = path.resolve()
    if not path.exists():
        return None
    return path


@lru_cache(maxsize=2)
def load_climatology_reference(path: str | Path | None = None) -> ClimatologyReference | None:
    """Load climatology metadata from disk and cache the coordinate arrays."""

    resolved = resolve_climatology_source(path)
    if resolved is None:
        return None

    with Dataset(resolved, "r") as ds:
        depth = np.asarray(ds.variables["Z"][:], dtype=float)
        latitude = np.asarray(ds.variables["latitude"][:], dtype=float)
        longitude = np.asarray(ds.variables["longitude"][:], dtype=float)

    return ClimatologyReference(
        path=resolved,
        depth=depth,
        latitude=latitude,
        longitude=longitude,
    )


def sample_climatology_reference(
    profile: ProfileInput,
    *,
    source: bool | str | Path | None = None,
    sigma_vert: ArrayLike | None = None,
) -> ClimatologySample | None:
    """Return reference values for *profile* if the climatology file is available."""

    reference = load_climatology_reference(source)
    if reference is None:
        return None
    return reference.sample(profile, sigma_vert=sigma_vert)


def _profile_month(profile: ProfileInput) -> int:
    juld = profile.attrs.get("juld")
    if juld is not None:
        try:
            month = (datetime(1950, 1, 1) + timedelta(days=float(juld))).month
            return int(np.clip(month, 1, 12))
        except Exception:
            pass
    if profile.day_of_year is not None:
        try:
            month = (datetime(2001, 1, 1) + timedelta(days=float(profile.day_of_year) % 365.2425)).month
            return int(np.clip(month, 1, 12))
        except Exception:
            pass
    return 1


def _nearest_index(values: FloatArray, target: float) -> int:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0
    finite = np.isfinite(arr)
    if not np.any(finite):
        return 0
    idx = np.argmin(np.abs(arr[finite] - float(target)))
    return int(np.flatnonzero(finite)[idx])


def _nearest_longitude_index(values: FloatArray, target: float) -> int:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0
    finite = np.isfinite(arr)
    if not np.any(finite):
        return 0
    lon = float(target)
    diff = np.abs(((arr[finite] - lon + 180.0) % 360.0) - 180.0)
    idx = np.argmin(diff)
    return int(np.flatnonzero(finite)[idx])


def _read_column(var: Any, month_index: int, lat_index: int, lon_index: int) -> FloatArray | None:
    try:
        raw = var[month_index, :, lat_index, lon_index]
    except Exception:
        return None
    data = np.asarray(np.ma.filled(raw, np.nan), dtype=float)
    fill_value = None
    for attr in ("_FillValue", "missing_value", "fill_value"):
        try:
            candidate = getattr(var, attr, None)
            if candidate is not None:
                fill_value = float(candidate)
                break
        except Exception:
            continue
    if fill_value is not None and np.isfinite(fill_value):
        data[data == fill_value] = np.nan
    return data


def _fill_missing_profile_values(values: ArrayLike, pressure: ArrayLike) -> FloatArray:
    x = np.asarray(values, dtype=float).copy()
    p = np.asarray(pressure, dtype=float)
    finite = np.isfinite(x) & np.isfinite(p)
    if int(np.sum(finite)) == 0:
        return np.zeros_like(x, dtype=float)
    if int(np.sum(finite)) == 1:
        x[:] = x[finite][0]
        return x
    order = np.argsort(p[finite])
    p_valid = np.asarray(p[finite], dtype=float)[order]
    x_valid = np.asarray(x[finite], dtype=float)[order]
    x[~finite] = np.interp(p[~finite], p_valid, x_valid)
    return x


def _coerce_sigma_array(value: ArrayLike | None, n_levels: int) -> FloatArray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 0:
        return None
    if arr.size == 1:
        return np.full(n_levels, float(arr[0]), dtype=float)
    if arr.size == n_levels:
        return arr.astype(float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None
    return np.full(n_levels, float(np.nanmean(finite)), dtype=float)


def _safe_gradient(values: ArrayLike, pressure: ArrayLike) -> FloatArray:
    x = np.asarray(values, dtype=float)
    p = np.asarray(pressure, dtype=float)
    finite = np.isfinite(x) & np.isfinite(p)
    if int(np.sum(finite)) < 2:
        return np.zeros_like(x, dtype=float)
    filled = _fill_missing_profile_values(x, p)
    try:
        grad = np.gradient(filled, p, edge_order=1)
    except Exception:
        grad = np.zeros_like(filled, dtype=float)
    return np.nan_to_num(np.asarray(grad, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
