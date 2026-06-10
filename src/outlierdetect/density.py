"""Lightweight density and stability diagnostics.

This MVP keeps a simple proxy for feature engineering and stability heuristics,
but plotting helpers can also request GSW-backed sigma0 contours.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def density_proxy(temperature: ArrayLike, salinity: ArrayLike) -> FloatArray:
    """Return a simple potential-density anomaly proxy.

    The scale is approximately kg m-3, but only relative differences should be
    interpreted. The sign convention is correct: density increases with salinity
    and decreases with temperature.
    """
    temp = np.asarray(temperature, dtype=float)
    sal = np.asarray(salinity, dtype=float)
    alpha = 0.20  # kg m-3 degC-1, rough thermal expansion effect
    beta = 0.78  # kg m-3 psu-1, rough haline contraction effect
    return -alpha * (temp - 10.0) + beta * (sal - 35.0)


def sigma0_from_ts(
    salinity: ArrayLike,
    temperature: ArrayLike,
    *,
    lon: float = 0.0,
    lat: float = 0.0,
) -> FloatArray:
    """Compute sigma0 contours from T/S using the GSW package.

    This is intended for T-S plots. The plot code does not carry station
    coordinates, so the lon/lat defaults are a pragmatic approximation.
    """
    try:
        import gsw  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "GSW is required for density contours. Install with: pip install gsw "
            "or pip install -e '.[train]'."
        ) from exc

    sp, temp = np.broadcast_arrays(
        np.asarray(salinity, dtype=float),
        np.asarray(temperature, dtype=float),
    )
    pressure = np.zeros_like(sp, dtype=float)
    sa = gsw.SA_from_SP(sp, pressure, lon, lat)
    ct = gsw.CT_from_t(sa, temp, pressure)
    return np.asarray(gsw.sigma0(sa, ct), dtype=float)


def inversion_metrics(
    pressure: ArrayLike,
    temperature: ArrayLike,
    salinity: ArrayLike,
    tolerance: float = 1e-4,
) -> dict[str, FloatArray | float]:
    """Compute simple density-inversion diagnostics for a sparse profile.

    Returns
    -------
    dict
        ``density_proxy``: proxy density at levels.
        ``drho_dp``: finite-difference density gradient between levels.
        ``level_inversion_magnitude``: per-level positive magnitude where density
        decreases with pressure.
        ``max_inversion``: maximum pairwise inversion magnitude.
    """
    p = np.asarray(pressure, dtype=float)
    rho = density_proxy(temperature, salinity)
    n = p.size
    if n < 2:
        return {
            "density_proxy": rho,
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
        "drho_dp": drho_dp,
        "level_inversion_magnitude": level_mag,
        "max_inversion": max_inv,
    }


def stable_project_salinity_only(
    pressure: ArrayLike,
    temperature: ArrayLike,
    salinity: ArrayLike,
    min_density_step: float = 1e-5,
) -> tuple[FloatArray, FloatArray]:
    """MVP static-stability projection by minimally increasing salinity.

    This is a crude placeholder. It keeps temperature fixed, computes a monotone
    density proxy using a pool-adjacent-violators algorithm, then adjusts salinity
    to match that monotone proxy. Production use should replace this with a
    TEOS-10-aware constrained optimization over both T and S.
    """
    p = np.asarray(pressure, dtype=float)
    t = np.asarray(temperature, dtype=float).copy()
    s = np.asarray(salinity, dtype=float).copy()
    if p.size != t.size or p.size != s.size:
        raise ValueError("pressure, temperature, and salinity must have the same length.")
    if p.size < 2:
        return t, s

    # Sort internally for projection and restore original order.
    order = np.argsort(p)
    inv_order = np.argsort(order)
    rho = density_proxy(t[order], s[order])

    rho_mono = _pava_non_decreasing(rho)
    rho_mono = np.maximum.accumulate(rho_mono + min_density_step * np.arange(rho_mono.size))

    beta = 0.78
    # rho = -alpha*(T-10)+beta*(S-35); only salinity is adjusted here.
    delta_s = np.maximum((rho_mono - rho) / beta, 0.0)
    s_sorted = s[order] + delta_s
    return t, s_sorted[inv_order]


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
