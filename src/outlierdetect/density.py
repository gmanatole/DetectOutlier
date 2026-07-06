"""TEOS-10/GSW-backed density and stability diagnostics.

The historical ``density_proxy`` name is preserved for compatibility, but the
implementation now uses the GSW equation of state instead of the old linear
T/S proxy. Static stability is assessed with sigma0, which keeps the logic
physical while remaining practical for sparse CTD profiles.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def profile_location_from_attrs(attrs: Mapping[str, Any] | None) -> tuple[float, float]:
    """Extract longitude/latitude metadata with safe fallbacks."""
    if attrs is None:
        return 0.0, 0.0
    lon = _coerce_optional_float(_lookup_attr(attrs, ("longitude", "lon", "LONGITUDE", "LON")))
    lat = _coerce_optional_float(_lookup_attr(attrs, ("latitude", "lat", "LATITUDE", "LAT")))
    return 0.0 if lon is None else lon, 0.0 if lat is None else lat


def density_proxy(
    temperature: ArrayLike,
    salinity: ArrayLike,
    pressure: ArrayLike | None = None,
    *,
    lon: float = 0.0,
    lat: float = 0.0,
) -> FloatArray:
    """Return TEOS-10 sigma0 for a profile."""
    return _sigma0_from_profile(pressure, temperature, salinity, lon=lon, lat=lat)


def sigma0_from_ts(
    salinity: ArrayLike,
    temperature: ArrayLike,
    *,
    lon: float = 0.0,
    lat: float = 0.0,
) -> FloatArray:
    """Compute sigma0 contours from T/S using GSW.

    This is intended for T-S plots. Plot code does not always carry station
    coordinates, so lon/lat default to zero as a pragmatic fallback.
    """
    sp, temp = np.broadcast_arrays(
        np.asarray(salinity, dtype=float),
        np.asarray(temperature, dtype=float),
    )
    pressure = np.zeros_like(sp, dtype=float)
    return _sigma0_from_profile(pressure, temp, sp, lon=lon, lat=lat)


def inversion_metrics(
    pressure: ArrayLike,
    temperature: ArrayLike,
    salinity: ArrayLike,
    tolerance: float = 1e-4,
    *,
    lon: float = 0.0,
    lat: float = 0.0,
) -> dict[str, FloatArray | float]:
    """Compute density-inversion diagnostics for a sparse profile.

    Returns
    -------
    dict
        ``density_proxy`` and ``density_sigma0``: sigma0 at levels.
        ``drho_dp``: finite-difference density gradient between levels.
        ``level_inversion_magnitude``: per-level positive magnitude where
        density decreases with pressure.
        ``max_inversion``: maximum pairwise inversion magnitude.
    """
    p = np.asarray(pressure, dtype=float)
    rho = density_proxy(temperature, salinity, p, lon=lon, lat=lat)
    n = p.size
    if n < 2:
        return {
            "density_proxy": rho,
            "density_sigma0": rho,
            "drho_dp": np.full(0, np.nan),
            "level_inversion_magnitude": np.full(n, np.nan),
            "max_inversion": np.nan,
        }

    dp = np.diff(p)
    dp = np.where(dp <= 0, np.nan, dp)
    drho = np.diff(rho)
    drho_dp = drho / dp
    pair_inversion = np.maximum(-(drho - tolerance), 0.0)
    level_mag = np.zeros(n, dtype=float)
    finite = np.isfinite(pair_inversion)
    for j, mag in enumerate(pair_inversion):
        if finite[j]:
            level_mag[j] = max(level_mag[j], mag)
            level_mag[j + 1] = max(level_mag[j + 1], mag)
    max_inv = float(np.nanmax(level_mag)) if np.any(np.isfinite(level_mag)) else np.nan
    return {
        "density_proxy": rho,
        "density_sigma0": rho,
        "drho_dp": drho_dp,
        "level_inversion_magnitude": level_mag,
        "max_inversion": max_inv,
    }


def stable_project_salinity_only(
    pressure: ArrayLike,
    temperature: ArrayLike,
    salinity: ArrayLike,
    min_density_step: float = 1e-5,
    *,
    lon: float = 0.0,
    lat: float = 0.0,
) -> tuple[FloatArray, FloatArray]:
    """Project a profile onto a statically stable sigma0 sequence.

    The temperature profile is left fixed. Salinity is increased just enough to
    make the sigma0 profile non-decreasing with pressure, using a local TEOS-10
    sensitivity estimate at each level.
    """
    p = np.asarray(pressure, dtype=float)
    t = np.asarray(temperature, dtype=float).copy()
    s = np.asarray(salinity, dtype=float).copy()
    if p.size != t.size or p.size != s.size:
        raise ValueError("pressure, temperature, and salinity must have the same length.")
    if p.size < 2:
        return t, s

    order = np.argsort(p)
    inv_order = np.argsort(order)
    p_sorted = _strictly_increasing_pressure(p[order])
    t_sorted = _fill_missing_profile_values(t[order], p_sorted)
    s_sorted = _fill_missing_profile_values(s[order], p_sorted)

    try:
        import gsw  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - dependency should be present
        raise ImportError(
            "GSW is required for density projection. Install with: pip install gsw "
            "or pip install -e '.[train]'."
        ) from exc

    for _ in range(2):
        sa = np.asarray(gsw.SA_from_SP(s_sorted, p_sorted, lon, lat), dtype=float)
        ct = np.asarray(gsw.CT_from_t(sa, t_sorted, p_sorted), dtype=float)
        sigma0 = np.asarray(gsw.sigma0(sa, ct), dtype=float)
        rho, _, beta = gsw.rho_alpha_beta(sa, ct, p_sorted)
        rho = np.asarray(rho, dtype=float)
        beta = np.asarray(beta, dtype=float)

        sigma0_mono = _pava_non_decreasing(sigma0)
        sigma0_mono = np.maximum.accumulate(
            sigma0_mono + min_density_step * np.arange(sigma0_mono.size, dtype=float)
        )

        drho_dsa = rho * beta
        finite = np.isfinite(drho_dsa) & (drho_dsa > 0)
        fallback = float(np.nanmedian(drho_dsa[finite])) if np.any(finite) else 0.75
        drho_dsa = np.where(finite, drho_dsa, fallback)

        delta_sa = np.maximum((sigma0_mono - sigma0) / np.maximum(drho_dsa, 1e-12), 0.0)
        sa_projected = sa + delta_sa
        s_next = np.asarray(gsw.SP_from_SA(sa_projected, p_sorted, lon, lat), dtype=float)
        if np.allclose(s_next, s_sorted, rtol=1e-6, atol=1e-8, equal_nan=True):
            s_sorted = s_next
            break
        s_sorted = s_next

    return t[inv_order], s_sorted[inv_order]


def _sigma0_from_profile(
    pressure: ArrayLike | None,
    temperature: ArrayLike,
    salinity: ArrayLike,
    *,
    lon: float = 0.0,
    lat: float = 0.0,
) -> FloatArray:
    gsw = _require_gsw()
    t = np.asarray(temperature, dtype=float)
    s = np.asarray(salinity, dtype=float)
    p = np.zeros_like(t, dtype=float) if pressure is None else np.asarray(pressure, dtype=float)
    p, t, s = np.broadcast_arrays(p, t, s)
    p = np.asarray(p, dtype=float)
    t = _fill_missing_profile_values(np.asarray(t, dtype=float), p)
    s = _fill_missing_profile_values(np.asarray(s, dtype=float), p)
    sa = np.asarray(gsw.SA_from_SP(s, p, lon, lat), dtype=float)
    ct = np.asarray(gsw.CT_from_t(sa, t, p), dtype=float)
    return np.asarray(gsw.sigma0(sa, ct), dtype=float)


def _fill_missing_profile_values(values: FloatArray, pressure: FloatArray | None) -> FloatArray:
    x = np.asarray(values, dtype=float).copy()
    if x.size == 0:
        return x
    if pressure is None:
        p = np.arange(x.size, dtype=float)
    else:
        p = np.asarray(pressure, dtype=float)
        if p.shape != x.shape:
            p = np.broadcast_to(p, x.shape).astype(float, copy=False)

    finite = np.isfinite(x) & np.isfinite(p)
    if int(np.sum(finite)) == 0:
        return np.zeros_like(x, dtype=float)
    if int(np.sum(finite)) == 1:
        x[:] = x[finite][0]
        return x

    order = np.argsort(p[finite])
    p_valid = _strictly_increasing_pressure(np.asarray(p[finite], dtype=float)[order])
    x_valid = np.asarray(x[finite], dtype=float)[order]
    out = x.copy()
    missing = ~finite
    if np.any(missing):
        out[missing] = np.interp(p[missing], p_valid, x_valid)
    return out


def _require_gsw():
    try:
        import gsw  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - dependency should be present
        raise ImportError(
            "GSW is required for density calculations. Install with: pip install gsw "
            "or pip install -e '.[train]'."
        ) from exc
    return gsw


def _lookup_attr(attrs: Mapping[str, Any], names: tuple[str, ...]) -> Any | None:
    for name in names:
        if name in attrs:
            return attrs[name]
    return None


def _coerce_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        return None
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None
    return float(finite[0])


def _strictly_increasing_pressure(pressure: FloatArray, *, min_step: float = 1e-6) -> FloatArray:
    """Return a monotonically increasing copy of pressure."""
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


def _pava_non_decreasing(y: FloatArray) -> FloatArray:
    """Pool-adjacent-violators algorithm for non-decreasing sequence."""
    y = np.asarray(y, dtype=float)
    n = y.size
    values: list[float] = []
    weights: list[float] = []
    starts: list[int] = []
    ends: list[int] = []

    for i, yi in enumerate(y):
        if not np.isfinite(yi):
            yi = values[-1] if values else 0.0
        values.append(float(yi))
        weights.append(1.0)
        starts.append(i)
        ends.append(i + 1)
        while len(values) >= 2 and values[-2] > values[-1]:
            w = weights[-2] + weights[-1]
            v = (values[-2] * weights[-2] + values[-1] * weights[-1]) / w
            values[-2] = v
            weights[-2] = w
            ends[-2] = ends[-1]
            values.pop()
            weights.pop()
            starts.pop()
            ends.pop()

    out = np.empty(n, dtype=float)
    for v, start, end in zip(values, starts, ends, strict=False):
        out[start:end] = v
    return out
