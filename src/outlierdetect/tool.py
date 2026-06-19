"""Predictors.

The MVP exposes a no-training heuristic baseline and a neural wrapper. Both
return the same Result object, making downstream code stable while the NN
matures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .corrections import CorrectionPrior, estimate_correction_posterior
from .data import NormalizationStats, NuisanceBias, ProfileInput, Result
from .density import profile_location_from_attrs, stable_project_salinity_only
from .features import FeatureBatch, build_level_features

FloatArray = NDArray[np.float64]


@dataclass(slots=True)
class Config:
    """Configuration for inference."""

    point_z_threshold: float = 3.0
    curvature_threshold: float = 2.5
    density_inversion_scale: float = 0.02
    profile_flag_threshold: float = 0.5
    point_accept_threshold: float = 0.5
    do_reconstruction: bool = True
    reconstruction_grid_size: int = 80
    standard_pressure_grid: FloatArray | None = None
    min_reconstruction_points: int = 3
    default_sigma_t: float = 0.5
    default_sigma_s: float = 0.05
    correction_prior: CorrectionPrior | None = None
    recompute_correction_with_point_weights: bool = True


class Heuristic:
    """No-training MVP baseline.

    This class is intentionally transparent. It is not meant to be the final
    product, but it provides a usable API, test baseline, and reference behavior
    for the future neural model.
    """

    def __init__(self, config: Config | None = None):
        self.config = config or Config()

    def predict(
        self,
        profile: ProfileInput,
        *,
        correction_prior: CorrectionPrior | None = None,
    ) -> Result:
        active_prior = correction_prior or self.config.correction_prior or CorrectionPrior.default()
        features = build_level_features(profile, correction_prior=active_prior)
        correction_post = features.diagnostics.get("correction_posterior")
        lon, lat = profile_location_from_attrs(profile.attrs)
        point_t, point_s, point_rho = self._point_probabilities(features, profile)
        if self.config.recompute_correction_with_point_weights:
            point_weights_t = 1.0 - np.maximum(point_t, point_rho)
            point_weights_s = 1.0 - np.maximum(point_s, point_rho)
            correction_post = estimate_correction_posterior(
                profile,
                active_prior,
                point_weights_t=point_weights_t,
                point_weights_s=point_weights_s,
                default_sigma_t=self.config.default_sigma_t,
                default_sigma_s=self.config.default_sigma_s,
            )
        profile_prob = self._profile_probability(point_t, point_s, point_rho, features, correction_post)
        nuisance = (
            correction_post.as_nuisance_bias()
            if correction_post is not None
            else self._nuisance_bias(features)
        )

        if self.config.do_reconstruction:
            grid, trec, srec, sig_t, sig_s = self._reconstruct(
                profile,
                point_t,
                point_s,
                point_rho,
                features,
                lon=lon,
                lat=lat,
            )
        else:
            grid = trec = srec = sig_t = sig_s = None

        diagnostics = dict(features.diagnostics)
        diagnostics.update(
            {
                "method": "heuristic_mvp",
                "warning": "Heuristic baseline. Use for API tests and bootstrapping, not final QC.",
                "max_abs_detrended_z_t": float(np.nanmax(np.abs(features.column("detrended_z_t")))),
                "max_abs_detrended_z_s": float(np.nanmax(np.abs(features.column("detrended_z_s")))),
                "profile_flag_threshold": self.config.profile_flag_threshold,
                "correction_status": None if correction_post is None else getattr(correction_post, "status", None),
                "correction_prior_tension": None
                if correction_post is None
                else float(getattr(correction_post, "prior_tension", np.nan)),
                "correction_information_gain": None
                if correction_post is None
                else float(getattr(correction_post, "information_gain", np.nan)),
            }
        )
        return Result(
            profile_bad_probability=float(profile_prob),
            point_bad_t=point_t,
            point_bad_s=point_s,
            point_density_inconsistent=point_rho,
            nuisance_bias=nuisance,
            correction_posterior=correction_post,
            pressure_grid=grid,
            temperature_reconstructed=trec,
            salinity_reconstructed=srec,
            sigma_temperature=sig_t,
            sigma_salinity=sig_s,
            feature_names=features.feature_names,
            diagnostics=diagnostics,
            profile_id=profile.profile_id,
        )

    def _point_probabilities(
        self, features: FeatureBatch, profile: ProfileInput
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        c = self.config
        dz_t = _feature_or(features, "posterior_debiased_z_t", "detrended_z_t")
        dz_s = _feature_or(features, "posterior_debiased_z_s", "detrended_z_s")
        sigma_t = np.maximum(features.column("sigma_t"), c.default_sigma_t * 1e-3)
        sigma_s = np.maximum(features.column("sigma_s"), c.default_sigma_s * 1e-3)
        gap = 0.5 * (features.column("gap_above_norm") + features.column("gap_below_norm"))

        # Curvature score is dimensionless: curvature * local_gap^2 / local_sigma.
        d2t = np.abs(features.column("d2tdp2"))
        d2s = np.abs(features.column("d2sdp2"))
        p_span = max(float(np.nanmax(profile.pressure) - np.nanmin(profile.pressure)), 1.0)
        local_gap_dbar = np.maximum(gap * p_span, 1.0)
        rough_t = d2t * local_gap_dbar**2 / sigma_t
        rough_s = d2s * local_gap_dbar**2 / sigma_s

        score_t = 1.15 * (np.abs(dz_t) - c.point_z_threshold) + 0.6 * (
            rough_t - c.curvature_threshold
        )
        score_s = 1.15 * (np.abs(dz_s) - c.point_z_threshold) + 0.6 * (
            rough_s - c.curvature_threshold
        )

        inv_mag = np.maximum(features.column("density_inversion_magnitude"), 0.0)
        score_rho = inv_mag / max(c.density_inversion_scale, 1e-12) - 1.0

        point_t = _sigmoid(score_t)
        point_s = _sigmoid(score_s)
        point_rho = _sigmoid(score_rho)

        # Missing raw values are bad for that variable.
        point_t = np.where(np.isfinite(profile.temperature), point_t, 1.0)
        point_s = np.where(np.isfinite(profile.salinity), point_s, 1.0)
        point_rho = np.where(np.isfinite(profile.temperature) & np.isfinite(profile.salinity), point_rho, 1.0)

        # If residuals are absent, avoid making the residual score look informative.
        if profile.residual_t is None:
            point_t = np.maximum(0.05, 0.4 * _sigmoid(rough_t - c.curvature_threshold))
        if profile.residual_s is None:
            point_s = np.maximum(0.05, 0.4 * _sigmoid(rough_s - c.curvature_threshold))

        return point_t.astype(float), point_s.astype(float), point_rho.astype(float)

    def _profile_probability(
        self,
        point_t: FloatArray,
        point_s: FloatArray,
        point_rho: FloatArray,
        features: FeatureBatch,
        correction_post: Any | None = None,
    ) -> float:
        frac_bad = float(
            np.mean((point_t > 0.5) | (point_s > 0.5) | (point_rho > 0.5))
        )
        max_point = float(np.nanmax(np.maximum.reduce([point_t, point_s, point_rho])))
        zt = np.abs(_feature_or(features, "posterior_debiased_z_t", "detrended_z_t"))
        zs = np.abs(_feature_or(features, "posterior_debiased_z_s", "detrended_z_s"))
        coherent_bad = max(float(np.nanmedian(zt)), float(np.nanmedian(zs)))
        prior_tension = 0.0 if correction_post is None else float(getattr(correction_post, "prior_tension", 0.0))
        unusual_penalty = 0.0
        if correction_post is not None and getattr(correction_post, "status", None) in {
            "requires_unusual_correction",
            "slope_soft_bound_exceeded",
        }:
            unusual_penalty = 0.45
        tension_penalty = 0.08 * max(prior_tension - 9.5, 0.0)
        score = (
            3.0 * frac_bad
            + 1.25 * (max_point - 0.7)
            + 0.35 * (coherent_bad - 2.5)
            + tension_penalty
            + unusual_penalty
            - 1.0
        )
        return float(_sigmoid(score))

    def _nuisance_bias(self, features: FeatureBatch) -> NuisanceBias:
        t_det = features.diagnostics.get("t_detrend")
        s_det = features.diagnostics.get("s_detrend")
        return NuisanceBias(
            a_t=float(getattr(t_det, "intercept", np.nan)),
            b_t=float(getattr(t_det, "slope", np.nan)),
            a_s=float(getattr(s_det, "intercept", np.nan)),
            b_s=float(getattr(s_det, "slope", np.nan)),
            uncertainty={
                "a_t": float(getattr(t_det, "sigma_intercept", np.nan)),
                "b_t": float(getattr(t_det, "sigma_slope", np.nan)),
                "a_s": float(getattr(s_det, "sigma_intercept", np.nan)),
                "b_s": float(getattr(s_det, "sigma_slope", np.nan)),
                "note": "Local nuisance fit only; not a final tag adjustment.",
            },
        )

    def _reconstruct(
        self,
        profile: ProfileInput,
        point_t: FloatArray,
        point_s: FloatArray,
        point_rho: FloatArray,
        features: FeatureBatch,
        *,
        lon: float = 0.0,
        lat: float = 0.0,
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
        c = self.config
        grid = _pressure_grid(profile, c)
        accepted = (
            np.isfinite(profile.pressure)
            & np.isfinite(profile.temperature)
            & np.isfinite(profile.salinity)
            & (point_t < c.point_accept_threshold)
            & (point_s < c.point_accept_threshold)
            & (point_rho < c.point_accept_threshold)
        )
        if int(np.sum(accepted)) < c.min_reconstruction_points:
            accepted = np.isfinite(profile.pressure) & np.isfinite(profile.temperature) & np.isfinite(profile.salinity)

        if int(np.sum(accepted)) < 2:
            nan = np.full_like(grid, np.nan, dtype=float)
            large_t = np.full_like(grid, 10.0, dtype=float)
            large_s = np.full_like(grid, 1.0, dtype=float)
            return grid, nan, nan, large_t, large_s

        p = profile.pressure[accepted]
        t = profile.temperature[accepted]
        s = profile.salinity[accepted]
        order = np.argsort(p)
        p = p[order]
        t = t[order]
        s = s[order]

        trec = np.interp(grid, p, t)
        srec = np.interp(grid, p, s)
        trec, srec = stable_project_salinity_only(grid, trec, srec, lon=lon, lat=lat)

        sigma_t_levels = features.column("sigma_t")[accepted][order]
        sigma_s_levels = features.column("sigma_s")[accepted][order]
        sig_t = np.interp(grid, p, sigma_t_levels)
        sig_s = np.interp(grid, p, sigma_s_levels)
        nearest_dist = _nearest_distance(grid, p)
        sigma_vert = features.column("sigma_vert")[accepted][order]
        sigma_vert_grid = np.maximum(np.interp(grid, p, np.maximum(sigma_vert, 0.0)), 10.0)
        gap_factor = 1.0 + nearest_dist / sigma_vert_grid
        sig_t = sig_t * gap_factor
        sig_s = sig_s * gap_factor
        return grid, trec, srec, sig_t, sig_s


class Neural:
    """Wrapper around a trained PyTorch neural net.

    The wrapper keeps the same input/output API as Heuristic. It expects the
    neural model to have been trained with the same feature order returned by
    build_level_features().
    """

    def __init__(
        self,
        model: Any,
        config: Config | None = None,
        device: str | None = None,
        normalization: NormalizationStats | dict[str, float] | None = None,
    ):
        self.model = model
        self.config = config or Config()
        self.device = device
        self.normalization = NormalizationStats.from_mapping(normalization)
        if hasattr(self.model, "eval"):
            self.model.eval()
        if device is not None and hasattr(self.model, "to"):
            self.model.to(device)

    def predict(
        self,
        profile: ProfileInput,
        *,
        correction_prior: CorrectionPrior | None = None,
    ) -> Result:
        try:
            import torch
        except Exception as exc:  # pragma: no cover
            raise ImportError("Neural requires PyTorch.") from exc

        active_prior = correction_prior or self.config.correction_prior or CorrectionPrior.default()
        features = build_level_features(
            profile,
            normalization=self.normalization,
            correction_prior=active_prior,
        )
        lon, lat = profile_location_from_attrs(profile.attrs)
        x = torch.as_tensor(features.level_features[None, :, :], dtype=torch.float32)
        mask = torch.as_tensor(features.mask[None, :], dtype=torch.bool)
        grid = _pressure_grid(profile, self.config)
        pressure_grid = torch.as_tensor(grid[None, :], dtype=torch.float32)
        if self.device is not None:
            x = x.to(self.device)
            mask = mask.to(self.device)
            pressure_grid = pressure_grid.to(self.device)
        with torch.no_grad():
            out = self.model(x, mask=mask, recon_pressure=pressure_grid)
        point = torch.sigmoid(out["point_logits"])[0].detach().cpu().numpy()
        profile_prob = float(torch.sigmoid(out["profile_logit"])[0].detach().cpu().numpy())
        nuisance_mean = out["nuisance_mean"][0].detach().cpu().numpy()
        nuisance_log_std = out["nuisance_log_std"][0].detach().cpu().numpy()
        recon_mean = out["recon_mean"][0].detach().cpu().numpy()
        recon_std = np.exp(out["recon_log_std"][0].detach().cpu().numpy())

        trec = recon_mean[:, 0]
        srec = recon_mean[:, 1]
        if self.normalization is not None:
            trec = self.normalization.denormalize_temperature(trec)
            srec = self.normalization.denormalize_salinity(srec)
            recon_std = np.column_stack(
                [
                    recon_std[:, 0] * self.normalization.temperature_scale,
                    recon_std[:, 1] * self.normalization.salinity_scale,
                ]
            )
        trec, srec = stable_project_salinity_only(grid, trec, srec, lon=lon, lat=lat)
        correction_post = estimate_correction_posterior(
            profile,
            active_prior,
            point_weights_t=1.0 - np.maximum(point[:, 0], point[:, 2]),
            point_weights_s=1.0 - np.maximum(point[:, 1], point[:, 2]),
            default_sigma_t=self.config.default_sigma_t,
            default_sigma_s=self.config.default_sigma_s,
        )
        nuisance = correction_post.as_nuisance_bias()
        return Result(
            profile_bad_probability=profile_prob,
            point_bad_t=point[:, 0],
            point_bad_s=point[:, 1],
            point_density_inconsistent=point[:, 2],
            nuisance_bias=nuisance,
            correction_posterior=correction_post,
            pressure_grid=grid,
            temperature_reconstructed=trec,
            salinity_reconstructed=srec,
            sigma_temperature=recon_std[:, 0],
            sigma_salinity=recon_std[:, 1],
            feature_names=features.feature_names,
            diagnostics={
                "method": "neural",
                "feature_version": features.diagnostics.get("feature_version"),
                "nuisance_head_mean": nuisance_mean,
                "nuisance_head_std": np.exp(nuisance_log_std),
                "correction_status": getattr(correction_post, "status", None),
                "correction_prior_tension": float(getattr(correction_post, "prior_tension", np.nan)),
                "correction_information_gain": float(getattr(correction_post, "information_gain", np.nan)),
            },
            profile_id=profile.profile_id,
        )


def _feature_or(features: FeatureBatch, preferred: str, fallback: str) -> FloatArray:
    if preferred in features.feature_names:
        return features.column(preferred)
    return features.column(fallback)

def _sigmoid(x: FloatArray | float) -> FloatArray | float:
    return 1.0 / (1.0 + np.exp(-np.asarray(x)))


def _pressure_grid(profile: ProfileInput, config: Config, grid_size: int | None = None) -> FloatArray:
    if config.standard_pressure_grid is not None:
        return np.asarray(config.standard_pressure_grid, dtype=float)
    n = int(grid_size or config.reconstruction_grid_size)
    return np.linspace(profile.pmin, profile.pmax, n, dtype=float)


def _nearest_distance(grid: FloatArray, points: FloatArray) -> FloatArray:
    grid = np.asarray(grid, dtype=float)
    points = np.asarray(points, dtype=float)
    return np.min(np.abs(grid[:, None] - points[None, :]), axis=1)
