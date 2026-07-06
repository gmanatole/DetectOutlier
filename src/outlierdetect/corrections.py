"""Gaussian nuisance-correction prior/posterior for local T/S bias estimation.

The correction model treats the sparse-profile nuisance terms as a 4D latent
variable ``[a_t, b_t, a_s, b_s]`` with linear-in-pressure corrections expressed
per km. The prior is intentionally physical rather than purely statistical:

- salinity offsets are allowed to be larger than temperature offsets,
- temperature and salinity corrections can be correlated,
- slopes are softly bounded near 0.05 per km,
- the posterior can be refined with point weights from the QC heads.

This is an explicit Bayesian layer around the nuisance correction, not a final
tag-adjustment model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .data import NuisanceBias, ProfileInput

FloatArray = NDArray[np.float64]

PARAMETER_NAMES: tuple[str, str, str, str] = ("a_t", "b_t", "a_s", "b_s")


@dataclass(slots=True)
class CorrectionPrior:
    """Gaussian prior for local linear-in-pressure T/S nuisance corrections."""

    mean: FloatArray
    covariance: FloatArray
    parameter_names: tuple[str, str, str, str] = PARAMETER_NAMES
    slope_soft_bounds: Mapping[str, float] = field(
        default_factory=lambda: {"b_t": 0.05, "b_s": 0.05}
    )
    units: Mapping[str, str] = field(
        default_factory=lambda: {
            "a_t": "degree_C",
            "b_t": "degree_C_per_km",
            "a_s": "salinity",
            "b_s": "salinity_per_km",
        }
    )
    source: str = "default_outlierdetect_prior"
    note: str = (
        "Prior over local nuisance corrections; not a final tag adjustment. "
        "Slopes are expressed per km using z=(p-p_ref)/1000 dbar."
    )

    def __post_init__(self) -> None:
        self.mean = np.asarray(self.mean, dtype=float).reshape(4)  # type: ignore[assignment]
        self.covariance = np.asarray(self.covariance, dtype=float)  # type: ignore[assignment]
        if self.covariance.shape != (4, 4):
            raise ValueError("correction covariance must have shape (4, 4).")
        if not np.all(np.isfinite(self.mean)):
            raise ValueError("correction prior mean must be finite.")
        if not np.all(np.isfinite(self.covariance)):
            raise ValueError("correction prior covariance must be finite.")
        self.covariance = _nearest_positive_definite(self.covariance)  # type: ignore[assignment]

    @classmethod
    def default(
        cls,
        *,
        sigma_a_t: float = 0.04,
        sigma_b_t: float = 0.03,
        sigma_a_s: float = 0.12,
        sigma_b_s: float = 0.03,
        corr_a_t_a_s: float = 0.30,
        corr_b_t_b_s: float = 0.25,
        corr_a_s_b_s: float = 0.10,
    ) -> "CorrectionPrior":
        """Create the default physical prior.

        The default keeps temperature and slope corrections tight, allows larger
        salinity offsets, and injects mild covariance between the T and S terms.
        """

        std = np.array([sigma_a_t, sigma_b_t, sigma_a_s, sigma_b_s], dtype=float)
        corr = np.eye(4, dtype=float)
        corr[0, 2] = corr[2, 0] = float(corr_a_t_a_s)
        corr[1, 3] = corr[3, 1] = float(corr_b_t_b_s)
        corr[2, 3] = corr[3, 2] = float(corr_a_s_b_s)
        cov = corr * np.outer(std, std)
        return cls(mean=np.zeros(4, dtype=float), covariance=cov)

    @classmethod
    def from_estimate(
        cls,
        mean: ArrayLike,
        covariance: ArrayLike | None = None,
        std: ArrayLike | None = None,
        *,
        source: str = "external_correction_estimate",
        slope_soft_bounds: Mapping[str, float] | None = None,
    ) -> "CorrectionPrior":
        """Build a prior from an external estimate or calibration summary."""

        mean_arr = np.asarray(mean, dtype=float).reshape(4)
        if covariance is None:
            if std is None:
                raise ValueError("Provide either covariance or std.")
            std_arr = np.asarray(std, dtype=float).reshape(4)
            covariance_arr = np.diag(std_arr**2)
        else:
            covariance_arr = np.asarray(covariance, dtype=float)
        return cls(
            mean=mean_arr,
            covariance=covariance_arr,
            source=source,
            slope_soft_bounds=slope_soft_bounds
            if slope_soft_bounds is not None
            else {"b_t": 0.05, "b_s": 0.05},
        )

    def combined_with(self, other: "CorrectionPrior", *, source: str = "combined_prior") -> "CorrectionPrior":
        """Combine two independent Gaussian priors."""

        cov_a = _nearest_positive_definite(self.covariance)
        cov_b = _nearest_positive_definite(other.covariance)
        prec = _safe_inv(cov_a) + _safe_inv(cov_b)
        cov = _safe_inv(prec)
        mean = cov @ (_safe_inv(cov_a) @ self.mean + _safe_inv(cov_b) @ other.mean)
        return CorrectionPrior(mean=mean, covariance=cov, source=source)

    @property
    def std(self) -> FloatArray:
        return np.sqrt(np.maximum(np.diag(self.covariance), 0.0))

    @property
    def correlation(self) -> FloatArray:
        return covariance_to_correlation(self.covariance)

    def as_dict(self) -> dict[str, Any]:
        return {
            "parameter_names": list(self.parameter_names),
            "mean": self.mean.tolist(),
            "covariance": self.covariance.tolist(),
            "std": self.std.tolist(),
            "correlation": self.correlation.tolist(),
            "slope_soft_bounds": dict(self.slope_soft_bounds),
            "units": dict(self.units),
            "source": self.source,
            "note": self.note,
        }


@dataclass(slots=True)
class CorrectionPosterior:
    """Posterior nuisance correction diagnostics for one profile."""

    mean: FloatArray
    covariance: FloatArray
    prior: CorrectionPrior
    p_ref_dbar: float
    debiased_residual_t: FloatArray | None = None
    debiased_residual_s: FloatArray | None = None
    delta_t: FloatArray | None = None
    delta_s: FloatArray | None = None
    prior_tension: float = np.nan
    information_gain: float = np.nan
    constraint_strength: FloatArray | None = None
    status: str = "unknown"
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.mean = np.asarray(self.mean, dtype=float).reshape(4)  # type: ignore[assignment]
        self.covariance = np.asarray(self.covariance, dtype=float)  # type: ignore[assignment]
        if self.covariance.shape != (4, 4):
            raise ValueError("posterior covariance must have shape (4, 4).")
        self.covariance = _nearest_positive_definite(self.covariance)  # type: ignore[assignment]
        if self.constraint_strength is not None:
            self.constraint_strength = np.asarray(self.constraint_strength, dtype=float).reshape(4)  # type: ignore[assignment]

    @property
    def parameter_names(self) -> tuple[str, str, str, str]:
        return self.prior.parameter_names

    @property
    def std(self) -> FloatArray:
        return np.sqrt(np.maximum(np.diag(self.covariance), 0.0))

    @property
    def correlation(self) -> FloatArray:
        return covariance_to_correlation(self.covariance)

    @property
    def credible_interval_95(self) -> FloatArray:
        half = 1.96 * self.std
        return np.column_stack([self.mean - half, self.mean + half])

    def as_nuisance_bias(self) -> NuisanceBias:
        """Return a backwards-compatible summary object."""

        std = self.std
        ci = self.credible_interval_95
        return NuisanceBias(
            a_t=float(self.mean[0]),
            b_t=float(self.mean[1]),
            a_s=float(self.mean[2]),
            b_s=float(self.mean[3]),
            uncertainty={
                "a_t": float(std[0]),
                "b_t": float(std[1]),
                "a_s": float(std[2]),
                "b_s": float(std[3]),
                "credible_interval_95": {
                    name: [float(ci[i, 0]), float(ci[i, 1])]
                    for i, name in enumerate(self.parameter_names)
                },
                "covariance": self.covariance.tolist(),
                "correlation": self.correlation.tolist(),
                "prior_tension": float(self.prior_tension),
                "information_gain": float(self.information_gain),
                "constraint_strength": None
                if self.constraint_strength is None
                else self.constraint_strength.tolist(),
                "status": self.status,
                "p_ref_dbar": float(self.p_ref_dbar),
                "note": "Local nuisance correction posterior only; not a final tag adjustment.",
            },
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "parameter_names": list(self.parameter_names),
            "mean": self.mean.tolist(),
            "covariance": self.covariance.tolist(),
            "std": self.std.tolist(),
            "correlation": self.correlation.tolist(),
            "credible_interval_95": self.credible_interval_95.tolist(),
            "prior": self.prior.as_dict(),
            "p_ref_dbar": float(self.p_ref_dbar),
            "debiased_residual_t": None
            if self.debiased_residual_t is None
            else np.asarray(self.debiased_residual_t, dtype=float).tolist(),
            "debiased_residual_s": None
            if self.debiased_residual_s is None
            else np.asarray(self.debiased_residual_s, dtype=float).tolist(),
            "delta_t": None if self.delta_t is None else np.asarray(self.delta_t, dtype=float).tolist(),
            "delta_s": None if self.delta_s is None else np.asarray(self.delta_s, dtype=float).tolist(),
            "prior_tension": float(self.prior_tension),
            "information_gain": float(self.information_gain),
            "constraint_strength": None
            if self.constraint_strength is None
            else np.asarray(self.constraint_strength, dtype=float).tolist(),
            "status": self.status,
            "diagnostics": _jsonify(self.diagnostics),
        }


def estimate_correction_posterior(
    profile: ProfileInput,
    prior: CorrectionPrior | None = None,
    *,
    point_weights_t: ArrayLike | None = None,
    point_weights_s: ArrayLike | None = None,
    default_sigma_t: float = 0.5,
    default_sigma_s: float = 0.05,
    p_ref_dbar: float | None = None,
    min_weight: float = 1e-3,
) -> CorrectionPosterior:
    """Estimate the Gaussian posterior for local nuisance corrections."""

    prior = prior or CorrectionPrior.default()
    p = np.asarray(profile.pressure, dtype=float)
    n = p.size
    if p_ref_dbar is None:
        p_ref_dbar = float(np.nanmean(p)) if np.any(np.isfinite(p)) else 0.0
    z = pressure_km_coordinate(p, p_ref_dbar)

    rt = _optional_array(profile.effective_residual_t(), n=n)
    rs = _optional_array(profile.effective_residual_s(), n=n)
    sigma_t = _effective_sigma(profile.sigma_t, profile.effective_sigma_heave_t(), default_sigma_t, n)
    sigma_s = _effective_sigma(profile.sigma_s, profile.effective_sigma_heave_s(), default_sigma_s, n)
    rho = _optional_array(profile.rho_ts, n=n, default=0.0)
    rho = np.clip(np.nan_to_num(rho, nan=0.0), -0.95, 0.95)
    wt = _weights(point_weights_t, n=n, min_weight=min_weight)
    ws = _weights(point_weights_s, n=n, min_weight=min_weight)

    rows: list[FloatArray] = []
    d_values: list[float] = []
    row_meta: list[tuple[int, str]] = []
    cov_blocks: list[FloatArray] = []

    for i in range(n):
        valid_t = bool(np.isfinite(rt[i]) and np.isfinite(sigma_t[i]) and sigma_t[i] > 0 and wt[i] > 0)
        valid_s = bool(np.isfinite(rs[i]) and np.isfinite(sigma_s[i]) and sigma_s[i] > 0 and ws[i] > 0)
        if not valid_t and not valid_s:
            continue
        local_rows: list[FloatArray] = []
        local_values: list[float] = []
        local_meta: list[tuple[int, str]] = []
        st_eff = float(sigma_t[i] / np.sqrt(max(wt[i], min_weight)))
        ss_eff = float(sigma_s[i] / np.sqrt(max(ws[i], min_weight)))
        if valid_t:
            local_rows.append(np.array([1.0, z[i], 0.0, 0.0], dtype=float))
            local_values.append(float(rt[i]))
            local_meta.append((i, "t"))
        if valid_s:
            local_rows.append(np.array([0.0, 0.0, 1.0, z[i]], dtype=float))
            local_values.append(float(rs[i]))
            local_meta.append((i, "s"))
        if valid_t and valid_s:
            block = np.array(
                [[st_eff**2, float(rho[i]) * st_eff * ss_eff], [float(rho[i]) * st_eff * ss_eff, ss_eff**2]],
                dtype=float,
            )
        elif valid_t:
            block = np.array([[st_eff**2]], dtype=float)
        else:
            block = np.array([[ss_eff**2]], dtype=float)
        rows.extend(local_rows)
        d_values.extend(local_values)
        row_meta.extend(local_meta)
        cov_blocks.append(block)

    if not rows:
        delta_t, delta_s, deb_t, deb_s = correction_profile_values(prior.mean, p, p_ref_dbar, rt, rs)
        return CorrectionPosterior(
            mean=prior.mean.copy(),
            covariance=prior.covariance.copy(),
            prior=prior,
            p_ref_dbar=float(p_ref_dbar),
            debiased_residual_t=deb_t,
            debiased_residual_s=deb_s,
            delta_t=delta_t,
            delta_s=delta_s,
            prior_tension=0.0,
            information_gain=0.0,
            constraint_strength=np.zeros(4, dtype=float),
            status="no_residuals",
            diagnostics={"n_observations": 0},
        )

    a = np.vstack(rows)
    d = np.asarray(d_values, dtype=float)
    r_cov = _nearest_positive_definite(_block_diag(cov_blocks))

    prior_prec = _safe_inv(prior.covariance)
    r_prec = _safe_inv(r_cov)
    post_prec = prior_prec + a.T @ r_prec @ a
    post_cov = _safe_inv(post_prec)
    rhs = prior_prec @ prior.mean + a.T @ r_prec @ d
    post_mean = post_cov @ rhs

    diff = post_mean - prior.mean
    prior_tension = float(diff.T @ prior_prec @ diff)
    info_gain = gaussian_kl(post_mean, post_cov, prior.mean, prior.covariance)
    std_prior = prior.std
    std_post = np.sqrt(np.maximum(np.diag(post_cov), 0.0))
    constraint_strength = np.clip(1.0 - std_post / np.maximum(std_prior, 1e-12), 0.0, 1.0)

    delta_t, delta_s, deb_t, deb_s = correction_profile_values(post_mean, p, p_ref_dbar, rt, rs)
    status = correction_status(
        prior_tension=prior_tension,
        constraint_strength=constraint_strength,
        mean=post_mean,
        prior=prior,
        pressure_span_dbar=float(np.nanmax(p) - np.nanmin(p)) if n else 0.0,
        n_observations=len(rows),
    )
    diagnostics = {
        "n_observations": int(len(rows)),
        "n_temperature_residuals": int(sum(var == "t" for _, var in row_meta)),
        "n_salinity_residuals": int(sum(var == "s" for _, var in row_meta)),
        "pressure_span_dbar": float(np.nanmax(p) - np.nanmin(p)) if n else 0.0,
        "slope_soft_bound_exceeded": slope_bound_exceeded(post_mean, prior),
    }
    return CorrectionPosterior(
        mean=post_mean,
        covariance=post_cov,
        prior=prior,
        p_ref_dbar=float(p_ref_dbar),
        debiased_residual_t=deb_t,
        debiased_residual_s=deb_s,
        delta_t=delta_t,
        delta_s=delta_s,
        prior_tension=prior_tension,
        information_gain=info_gain,
        constraint_strength=constraint_strength,
        status=status,
        diagnostics=diagnostics,
    )


def pressure_km_coordinate(pressure: ArrayLike, p_ref_dbar: float) -> FloatArray:
    """Pressure coordinate in km-like units, z=(p-p_ref)/1000."""

    return (np.asarray(pressure, dtype=float) - float(p_ref_dbar)) / 1000.0


def correction_profile_values(
    theta: ArrayLike,
    pressure: ArrayLike,
    p_ref_dbar: float,
    residual_t: ArrayLike | None = None,
    residual_s: ArrayLike | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray | None, FloatArray | None]:
    """Return Delta_T, Delta_S and optionally debiased residual arrays."""

    th = np.asarray(theta, dtype=float).reshape(4)
    p = np.asarray(pressure, dtype=float)
    z = pressure_km_coordinate(p, p_ref_dbar)
    delta_t = th[0] + th[1] * z
    delta_s = th[2] + th[3] * z
    deb_t = None
    deb_s = None
    if residual_t is not None:
        rt = np.asarray(residual_t, dtype=float)
        deb_t = rt - delta_t
    if residual_s is not None:
        rs = np.asarray(residual_s, dtype=float)
        deb_s = rs - delta_s
    return delta_t.astype(float), delta_s.astype(float), deb_t, deb_s


def correction_status(
    *,
    prior_tension: float,
    constraint_strength: FloatArray,
    mean: FloatArray,
    prior: CorrectionPrior,
    pressure_span_dbar: float,
    n_observations: int,
) -> str:
    """Classify posterior interpretability for logs and QC summaries."""

    if n_observations < 4:
        return "weakly_constrained"
    if pressure_span_dbar < 100.0:
        return "weakly_constrained_shallow_profile"
    if prior_tension > 13.3:
        return "requires_unusual_correction"
    if slope_bound_exceeded(mean, prior):
        return "slope_soft_bound_exceeded"
    max_strength = float(np.nanmax(constraint_strength)) if constraint_strength.size else 0.0
    if max_strength < 0.1:
        return "prior_dominated"
    if max_strength < 0.3:
        return "weakly_constrained"
    return "well_constrained"


def slope_bound_exceeded(theta: ArrayLike, prior: CorrectionPrior) -> bool:
    th = np.asarray(theta, dtype=float).reshape(4)
    bounds = prior.slope_soft_bounds
    b_t = float(abs(th[1]))
    b_s = float(abs(th[3]))
    return bool(
        ("b_t" in bounds and b_t > float(bounds["b_t"]))
        or ("b_s" in bounds and b_s > float(bounds["b_s"]))
    )


def gaussian_kl(mean_q: FloatArray, cov_q: FloatArray, mean_p: FloatArray, cov_p: FloatArray) -> float:
    """KL[N_q || N_p] for two Gaussian distributions."""

    k = mean_q.size
    cov_q = _nearest_positive_definite(cov_q)
    cov_p = _nearest_positive_definite(cov_p)
    prec_p = _safe_inv(cov_p)
    diff = mean_p - mean_q
    sign_p, logdet_p = np.linalg.slogdet(cov_p)
    sign_q, logdet_q = np.linalg.slogdet(cov_q)
    if sign_p <= 0 or sign_q <= 0:
        return float("nan")
    value = 0.5 * (np.trace(prec_p @ cov_q) + diff.T @ prec_p @ diff - k + logdet_p - logdet_q)
    return float(max(value, 0.0))


def covariance_to_correlation(covariance: ArrayLike) -> FloatArray:
    cov = np.asarray(covariance, dtype=float)
    std = np.sqrt(np.maximum(np.diag(cov), 0.0))
    denom = np.outer(std, std)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = cov / denom
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)
    return np.clip(corr, -1.0, 1.0)


def _effective_sigma(
    sigma: FloatArray | None,
    sigma_heave: FloatArray | None,
    default: float,
    n: int,
) -> FloatArray:
    base = _optional_array(sigma, n=n, default=default)
    base = np.where(np.isfinite(base) & (base > 0), base, default)
    if sigma_heave is not None:
        heave = _optional_array(sigma_heave, n=n, default=0.0)
        base = np.sqrt(base**2 + np.maximum(heave, 0.0) ** 2)
    return np.maximum(base, default * 1e-3)


def _optional_array(values: ArrayLike | None, *, n: int, default: float = np.nan) -> FloatArray:
    if values is None:
        return np.full(n, default, dtype=float)
    arr = np.asarray(values, dtype=float)
    if arr.shape != (n,):
        raise ValueError(f"Expected array of shape ({n},), got {arr.shape}.")
    return arr.astype(float, copy=False)


def _weights(values: ArrayLike | None, *, n: int, min_weight: float) -> FloatArray:
    if values is None:
        return np.ones(n, dtype=float)
    w = np.asarray(values, dtype=float)
    if w.shape != (n,):
        raise ValueError(f"point weights must have shape ({n},), got {w.shape}.")
    w = np.nan_to_num(w, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(w, min_weight, 1.0)


def _block_diag(blocks: Iterable[FloatArray]) -> FloatArray:
    blocks = list(blocks)
    size = int(sum(block.shape[0] for block in blocks))
    out = np.zeros((size, size), dtype=float)
    offset = 0
    for block in blocks:
        n = block.shape[0]
        out[offset : offset + n, offset : offset + n] = block
        offset += n
    return out


def _safe_inv(matrix: ArrayLike) -> FloatArray:
    mat = _nearest_positive_definite(np.asarray(matrix, dtype=float))
    try:
        return np.linalg.inv(mat)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(mat)


def _nearest_positive_definite(matrix: ArrayLike, jitter: float = 1e-10) -> FloatArray:
    """Symmetrize and add jitter until Cholesky succeeds."""

    mat = np.asarray(matrix, dtype=float)
    mat = 0.5 * (mat + mat.T)
    if mat.size == 0:
        return mat
    scale = max(float(np.nanmax(np.abs(np.diag(mat)))) if mat.ndim == 2 else 1.0, 1.0)
    eye = np.eye(mat.shape[0], dtype=float)
    for k in range(8):
        candidate = mat + eye * jitter * scale * (10.0**k)
        try:
            np.linalg.cholesky(candidate)
            return candidate
        except np.linalg.LinAlgError:
            continue
    eigvals, eigvecs = np.linalg.eigh(mat)
    eigvals = np.maximum(eigvals, jitter * scale)
    return (eigvecs * eigvals) @ eigvecs.T


def _jsonify(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    return value
