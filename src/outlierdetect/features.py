"""Feature construction for Tool 1.

The feature builder is the central input entry point for the neural network.
It keeps the model mostly local by exposing residuals, normalized residuals,
variability scales, vertical-heave scales, day-of-year phase, and sparse-profile
geometry. Latitude/longitude are not exposed as separate features, but they can
be used internally for TEOS-10 density diagnostics when present in metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .corrections import CorrectionPrior, estimate_correction_posterior
from .data import NormalizationStats, ProfileInput
from .density import density_proxy, inversion_metrics, profile_location_from_attrs

FloatArray = NDArray[np.float64]


@dataclass(slots=True)
class DetrendResult:
    intercept: float
    slope: float
    trend: FloatArray
    residual: FloatArray
    sigma_intercept: float = np.nan
    sigma_slope: float = np.nan

    def as_dict(self) -> dict[str, Any]:
        return {
            "intercept": float(self.intercept),
            "slope": float(self.slope),
            "trend": np.asarray(self.trend, dtype=float).tolist(),
            "residual": np.asarray(self.residual, dtype=float).tolist(),
            "sigma_intercept": float(self.sigma_intercept),
            "sigma_slope": float(self.sigma_slope),
        }


@dataclass(slots=True)
class FeatureBatch:
    """Per-level features and diagnostics for one sparse profile."""

    level_features: FloatArray
    feature_names: list[str]
    mask: NDArray[np.bool_]
    diagnostics: dict[str, Any]

    @property
    def n_levels(self) -> int:
        return int(self.level_features.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.level_features.shape[1])

    def column(self, name: str) -> FloatArray:
        idx = self.feature_names.index(name)
        return self.level_features[:, idx]


def pressure_normalized(pressure: ArrayLike) -> FloatArray:
    """Return pressure normalized to roughly [0, 1]."""
    p = np.asarray(pressure, dtype=float)
    p0 = float(np.nanmin(p))
    span = float(np.nanmax(p) - p0)
    if not np.isfinite(span) or span <= 0:
        span = max(float(np.nanmax(np.abs(p))), 1.0)
        p0 = 0.0
    return (p - p0) / span


def linear_detrend(
    pressure: ArrayLike,
    residual: ArrayLike | None,
    sigma: ArrayLike | None = None,
    mask: ArrayLike | None = None,
) -> DetrendResult:
    """Fit residual = intercept + slope * normalized_pressure.

    The fit is weighted by 1 / sigma^2 when sigma is provided. NaNs are ignored.
    If fewer than two valid points are available, the trend is set to the valid
    median residual and the slope is zero.
    """
    p = np.asarray(pressure, dtype=float)
    n = p.size
    if residual is None:
        nan = np.full(n, np.nan)
        return DetrendResult(np.nan, np.nan, nan, nan)
    r = np.asarray(residual, dtype=float)
    if r.size != n:
        raise ValueError("pressure and residual must have the same length.")
    x = pressure_normalized(p)
    valid = np.isfinite(p) & np.isfinite(r)
    if sigma is not None:
        s = np.asarray(sigma, dtype=float)
        if s.size != n:
            raise ValueError("sigma must have the same length as residual.")
        valid &= np.isfinite(s) & (s > 0)
    else:
        s = np.ones(n, dtype=float)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)

    trend = np.full(n, np.nan, dtype=float)
    detrended = np.full(n, np.nan, dtype=float)
    if int(np.sum(valid)) == 0:
        return DetrendResult(np.nan, np.nan, trend, detrended)
    if int(np.sum(valid)) == 1:
        intercept = float(r[valid][0])
        slope = 0.0
        trend[:] = intercept
        detrended = r - trend
        return DetrendResult(intercept, slope, trend, detrended)

    xv = x[valid]
    rv = r[valid]
    sv = s[valid]
    weights = 1.0 / np.maximum(sv, 1e-12) ** 2
    a = np.column_stack([np.ones_like(xv), xv])
    aw = a * np.sqrt(weights)[:, None]
    bw = rv * np.sqrt(weights)
    beta, *_ = np.linalg.lstsq(aw, bw, rcond=None)
    intercept, slope = float(beta[0]), float(beta[1])
    trend = intercept + slope * x
    detrended = r - trend

    sigma_intercept = np.nan
    sigma_slope = np.nan
    dof = int(np.sum(valid)) - 2
    if dof > 0:
        resid = rv - (intercept + slope * xv)
        weighted_var = float(np.sum(weights * resid**2) / dof)
        try:
            cov = weighted_var * np.linalg.inv(aw.T @ aw)
            sigma_intercept = float(np.sqrt(max(cov[0, 0], 0.0)))
            sigma_slope = float(np.sqrt(max(cov[1, 1], 0.0)))
        except np.linalg.LinAlgError:
            pass
    return DetrendResult(intercept, slope, trend, detrended, sigma_intercept, sigma_slope)


def robust_linear_detrend(
    pressure: ArrayLike,
    residual: ArrayLike | None,
    sigma: ArrayLike | None = None,
    mask: ArrayLike | None = None,
    n_iter: int = 4,
    huber_k: float = 2.5,
) -> DetrendResult:
    """Robust linear detrend with simple Huber reweighting."""
    if residual is None:
        return linear_detrend(pressure, residual, sigma=sigma, mask=mask)
    p = np.asarray(pressure, dtype=float)
    r = np.asarray(residual, dtype=float)
    base_sigma = np.ones_like(r) if sigma is None else np.asarray(sigma, dtype=float).copy()
    base_sigma = np.where(np.isfinite(base_sigma) & (base_sigma > 0), base_sigma, np.nanmedian(base_sigma[np.isfinite(base_sigma) & (base_sigma > 0)]) if np.any(np.isfinite(base_sigma) & (base_sigma > 0)) else 1.0)
    active = np.isfinite(r) & np.isfinite(p)
    if mask is not None:
        active &= np.asarray(mask, dtype=bool)

    effective_sigma = base_sigma.copy()
    result = linear_detrend(p, r, sigma=effective_sigma, mask=active)
    for _ in range(max(n_iter, 1)):
        scaled = np.abs(result.residual) / np.maximum(base_sigma, 1e-12)
        huber_weight = np.ones_like(scaled)
        too_large = scaled > huber_k
        huber_weight[too_large] = huber_k / np.maximum(scaled[too_large], 1e-12)
        effective_sigma = base_sigma / np.sqrt(np.maximum(huber_weight, 1e-6))
        result = linear_detrend(p, r, sigma=effective_sigma, mask=active)
    return result


def build_level_features(
    profile: ProfileInput,
    *,
    robust: bool = True,
    normalization: NormalizationStats | dict[str, float] | None = None,
    correction_prior: CorrectionPrior | None = None,
) -> FeatureBatch:
    """Build per-level features for one sparse CTD-SRDL profile.

    This function is the main data entry point for NN inference and training.
    """
    norm = NormalizationStats.from_mapping(normalization)
    p = profile.pressure.astype(float)
    t_raw = profile.temperature.astype(float)
    s_raw = profile.salinity.astype(float)
    residual_t_raw = None if profile.residual_t is None else np.asarray(profile.residual_t, dtype=float)
    residual_s_raw = None if profile.residual_s is None else np.asarray(profile.residual_s, dtype=float)
    n = p.size
    mask = np.isfinite(p) & (np.isfinite(t_raw) | np.isfinite(s_raw))

    if norm is None:
        t = t_raw
        s = s_raw
        residual_t = _nan_default(residual_t_raw, default=0.0, n=n)
        residual_s = _nan_default(residual_s_raw, default=0.0, n=n)
        sigma_t = _effective_sigma(profile.sigma_t, default=0.5, extra=profile.sigma_heave_t, n=n)
        sigma_s = _effective_sigma(profile.sigma_s, default=0.05, extra=profile.sigma_heave_s, n=n)
        sigma_vert = _nan_default(profile.sigma_vert, default=0.0, n=n)
        sigma_heave_t = _nan_default(profile.sigma_heave_t, default=0.0, n=n)
        sigma_heave_s = _nan_default(profile.sigma_heave_s, default=0.0, n=n)
    else:
        t = norm.normalize_temperature(t_raw)
        s = norm.normalize_salinity(s_raw)
        residual_t = _nan_default(residual_t_raw, default=0.0, n=n) / norm.temperature_scale
        residual_s = _nan_default(residual_s_raw, default=0.0, n=n) / norm.salinity_scale
        sigma_t = _effective_sigma(profile.sigma_t, default=0.5, extra=profile.sigma_heave_t, n=n)
        sigma_t = sigma_t / norm.temperature_scale
        sigma_s = _effective_sigma(profile.sigma_s, default=0.05, extra=profile.sigma_heave_s, n=n)
        sigma_s = sigma_s / norm.salinity_scale
        sigma_vert = _nan_default(profile.sigma_vert, default=0.0, n=n)
        sigma_heave_t = _nan_default(profile.sigma_heave_t, default=0.0, n=n) / norm.temperature_scale
        sigma_heave_s = _nan_default(profile.sigma_heave_s, default=0.0, n=n) / norm.salinity_scale

    p_norm = pressure_normalized(p)
    gap_above, gap_below = _neighbor_gaps(p)
    p_span = max(float(np.nanmax(p) - np.nanmin(p)), 1.0)
    gap_above_norm = gap_above / p_span
    gap_below_norm = gap_below / p_span
    edge_top = np.zeros(n, dtype=float)
    edge_bottom = np.zeros(n, dtype=float)
    edge_top[0] = 1.0
    edge_bottom[-1] = 1.0

    dtdp = _safe_gradient(t, p)
    dsdp = _safe_gradient(s, p)
    d2tdp2 = _safe_gradient(dtdp, p)
    d2sdp2 = _safe_gradient(dsdp, p)

    lon, lat = profile_location_from_attrs(profile.attrs)
    dens = density_proxy(t_raw, s_raw, p, lon=lon, lat=lat)
    inv = inversion_metrics(p, t_raw, s_raw, lon=lon, lat=lat)
    inv_mag = np.asarray(inv["level_inversion_magnitude"], dtype=float)
    rho_ts = np.clip(_nan_default(profile.rho_ts, default=0.0, n=n), -0.999, 0.999)
    has_residual_t = np.full(n, float(profile.residual_t is not None), dtype=float)
    has_residual_s = np.full(n, float(profile.residual_s is not None), dtype=float)
    has_sigma_t = np.full(n, float(profile.sigma_t is not None), dtype=float)
    has_sigma_s = np.full(n, float(profile.sigma_s is not None), dtype=float)

    z_t = residual_t / np.maximum(sigma_t, 1e-12)
    z_s = residual_s / np.maximum(sigma_s, 1e-12)

    detrend = robust_linear_detrend if robust else linear_detrend
    residual_t_for_fit = None if residual_t_raw is None else residual_t_raw / (norm.temperature_scale if norm is not None else 1.0)
    residual_s_for_fit = None if residual_s_raw is None else residual_s_raw / (norm.salinity_scale if norm is not None else 1.0)
    dt_res = detrend(p, residual_t_for_fit, sigma=sigma_t)
    ds_res = detrend(p, residual_s_for_fit, sigma=sigma_s)
    detrended_t = _nan_default(dt_res.residual, default=0.0, n=n)
    detrended_s = _nan_default(ds_res.residual, default=0.0, n=n)
    detrended_z_t = detrended_t / np.maximum(sigma_t, 1e-12)
    detrended_z_s = detrended_s / np.maximum(sigma_s, 1e-12)

    active_prior = correction_prior or CorrectionPrior.default()
    correction_post = estimate_correction_posterior(profile, active_prior)
    posterior_delta_t = _nan_default(correction_post.delta_t, default=0.0, n=n)
    posterior_delta_s = _nan_default(correction_post.delta_s, default=0.0, n=n)
    posterior_debiased_t = _nan_default(correction_post.debiased_residual_t, default=0.0, n=n)
    posterior_debiased_s = _nan_default(correction_post.debiased_residual_s, default=0.0, n=n)
    if norm is not None:
        posterior_delta_t = posterior_delta_t / norm.temperature_scale
        posterior_delta_s = posterior_delta_s / norm.salinity_scale
        posterior_debiased_t = posterior_debiased_t / norm.temperature_scale
        posterior_debiased_s = posterior_debiased_s / norm.salinity_scale
    posterior_debiased_z_t = posterior_debiased_t / np.maximum(sigma_t, 1e-12)
    posterior_debiased_z_s = posterior_debiased_s / np.maximum(sigma_s, 1e-12)
    correction_prior_tension = np.full(n, float(correction_post.prior_tension), dtype=float)
    correction_information_gain = np.full(n, float(correction_post.information_gain), dtype=float)
    if correction_post.constraint_strength is None:
        correction_constraint = np.zeros(4, dtype=float)
    else:
        correction_constraint = np.asarray(correction_post.constraint_strength, dtype=float)
    correction_constraint_a_t = np.full(n, float(correction_constraint[0]), dtype=float)
    correction_constraint_b_t = np.full(n, float(correction_constraint[1]), dtype=float)
    correction_constraint_a_s = np.full(n, float(correction_constraint[2]), dtype=float)
    correction_constraint_b_s = np.full(n, float(correction_constraint[3]), dtype=float)

    if profile.day_of_year is None:
        doy_sin = np.zeros(n, dtype=float)
        doy_cos = np.zeros(n, dtype=float)
        has_doy = np.zeros(n, dtype=float)
    else:
        phase = 2.0 * np.pi * float(profile.day_of_year) / 365.0
        doy_sin = np.full(n, np.sin(phase), dtype=float)
        doy_cos = np.full(n, np.cos(phase), dtype=float)
        has_doy = np.ones(n, dtype=float)

    names = [
        "p_norm",
        "gap_above_norm",
        "gap_below_norm",
        "edge_top",
        "edge_bottom",
        "temperature",
        "salinity",
        "dtdp",
        "dsdp",
        "d2tdp2",
        "d2sdp2",
        "density_proxy",
        "density_inversion_magnitude",
        "residual_t",
        "residual_s",
        "z_t",
        "z_s",
        "detrended_residual_t",
        "detrended_residual_s",
        "detrended_z_t",
        "detrended_z_s",
        "posterior_delta_t",
        "posterior_delta_s",
        "posterior_debiased_residual_t",
        "posterior_debiased_residual_s",
        "posterior_debiased_z_t",
        "posterior_debiased_z_s",
        "correction_prior_tension",
        "correction_information_gain",
        "correction_constraint_a_t",
        "correction_constraint_b_t",
        "correction_constraint_a_s",
        "correction_constraint_b_s",
        "sigma_t",
        "sigma_s",
        "sigma_vert",
        "sigma_heave_t",
        "sigma_heave_s",
        "rho_ts",
        "doy_sin",
        "doy_cos",
        "has_doy",
        "has_residual_t",
        "has_residual_s",
        "has_sigma_t",
        "has_sigma_s",
    ]
    cols = [
        p_norm,
        gap_above_norm,
        gap_below_norm,
        edge_top,
        edge_bottom,
        t,
        s,
        dtdp,
        dsdp,
        d2tdp2,
        d2sdp2,
        dens,
        inv_mag,
        residual_t,
        residual_s,
        z_t,
        z_s,
        detrended_t,
        detrended_s,
        detrended_z_t,
        detrended_z_s,
        posterior_delta_t,
        posterior_delta_s,
        posterior_debiased_t,
        posterior_debiased_s,
        posterior_debiased_z_t,
        posterior_debiased_z_s,
        correction_prior_tension,
        correction_information_gain,
        correction_constraint_a_t,
        correction_constraint_b_t,
        correction_constraint_a_s,
        correction_constraint_b_s,
        sigma_t,
        sigma_s,
        sigma_vert,
        sigma_heave_t,
        sigma_heave_s,
        rho_ts,
        doy_sin,
        doy_cos,
        has_doy,
        has_residual_t,
        has_residual_s,
        has_sigma_t,
        has_sigma_s,
    ]
    features = np.column_stack(cols).astype(float)
    features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)

    diagnostics = {
        "t_detrend": dt_res,
        "s_detrend": ds_res,
        "correction_posterior": correction_post,
        "max_density_inversion": inv["max_inversion"],
        "feature_version": "outlierdetect_v2_correction_prior_norm"
        if norm is not None
        else "outlierdetect_v3_gsw_sigma0_correction_prior",
    }
    if norm is not None:
        diagnostics["normalization"] = norm.as_dict()
    return FeatureBatch(features, names, mask=mask, diagnostics=diagnostics)


def _neighbor_gaps(p: FloatArray) -> tuple[FloatArray, FloatArray]:
    n = p.size
    above = np.zeros(n, dtype=float)
    below = np.zeros(n, dtype=float)
    if n == 1:
        return above, below
    dp = np.diff(p)
    dp = np.where(np.isfinite(dp) & (dp > 0), dp, np.nan)
    above[1:] = dp
    above[0] = dp[0]
    below[:-1] = dp
    below[-1] = dp[-1]
    fallback = np.nanmedian(dp) if np.any(np.isfinite(dp)) else 0.0
    above = np.where(np.isfinite(above), above, fallback)
    below = np.where(np.isfinite(below), below, fallback)
    return above, below


def _safe_gradient(values: FloatArray, pressure: FloatArray) -> FloatArray:
    values = np.asarray(values, dtype=float)
    pressure = _strictly_increasing_pressure(np.asarray(pressure, dtype=float))
    out = np.full_like(values, np.nan, dtype=float)
    valid = np.isfinite(values) & np.isfinite(pressure)
    if int(np.sum(valid)) < 2:
        return np.nan_to_num(out, nan=0.0)
    try:
        out[valid] = np.gradient(values[valid], pressure[valid], edge_order=1)
    except Exception:
        out[valid] = 0.0
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _nan_default(arr: FloatArray | None, *, default: float, n: int) -> FloatArray:
    if arr is None:
        return np.full(n, default, dtype=float)
    out = np.asarray(arr, dtype=float).copy()
    out = np.where(np.isfinite(out), out, default)
    return out


def _strictly_increasing_pressure(pressure: FloatArray, *, min_step: float = 1e-6) -> FloatArray:
    """Return a monotonically increasing copy of pressure.

    Small non-increasing segments can appear after synthetic pressure corruption
    or in sparse/rounded input files. We nudge them by an epsilon so gradient
    operations stay numerically safe without materially changing the profile.
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


def _effective_sigma(
    sigma: FloatArray | None,
    *,
    default: float,
    extra: FloatArray | None = None,
    n: int,
) -> FloatArray:
    base = _nan_default(sigma, default=default, n=n)
    base = np.maximum(base, default * 1e-3)
    if extra is not None:
        ex = _nan_default(extra, default=0.0, n=n)
        base = np.sqrt(base**2 + np.maximum(ex, 0.0) ** 2)
    return base
