"""Input and output data objects for the profile QC model.

The package deliberately keeps I/O simple: create a :class:`ProfileInput` from
arrays or a dataframe, pass it to a predictor, and receive a :class:`Result`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(slots=True)
class NormalizationStats:
    """Per-training-run standardization stats for temperature and salinity."""

    temperature_mean: float
    temperature_std: float
    salinity_mean: float
    salinity_std: float

    @classmethod
    def from_profiles(cls, profiles: Any) -> "NormalizationStats":
        temps: list[np.ndarray] = []
        sals: list[np.ndarray] = []
        for profile in profiles:
            if profile is None:
                continue
            temp_values = getattr(profile, "temperature", None)
            sal_values = getattr(profile, "salinity", None)
            if temp_values is not None:
                temp = np.asarray(temp_values, dtype=float).ravel()
                temp = temp[np.isfinite(temp)]
                if temp.size:
                    temps.append(temp)
            if sal_values is not None:
                sal = np.asarray(sal_values, dtype=float).ravel()
                sal = sal[np.isfinite(sal)]
                if sal.size:
                    sals.append(sal)
        if not temps or not sals:
            raise ValueError("Cannot compute normalization stats from empty profiles.")

        temp_all = np.concatenate(temps)
        sal_all = np.concatenate(sals)
        t_mean = float(np.mean(temp_all))
        t_std = float(np.std(temp_all))
        s_mean = float(np.mean(sal_all))
        s_std = float(np.std(sal_all))
        if not np.isfinite(t_std) or t_std <= 0:
            t_std = 1.0
        if not np.isfinite(s_std) or s_std <= 0:
            s_std = 1.0
        return cls(t_mean, t_std, s_mean, s_std)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | "NormalizationStats" | None) -> "NormalizationStats | None":
        if mapping is None:
            return None
        if isinstance(mapping, cls):
            return mapping

        def pick(*keys: str) -> float:
            for key in keys:
                if key in mapping:
                    return float(mapping[key])
            raise KeyError(f"Normalization mapping is missing keys {keys!r}.")

        return cls(
            temperature_mean=pick("T_mean", "temperature_mean"),
            temperature_std=pick("T_std", "temperature_std"),
            salinity_mean=pick("S_mean", "salinity_mean"),
            salinity_std=pick("S_std", "salinity_std"),
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "T_mean": float(self.temperature_mean),
            "T_std": float(self.temperature_std),
            "S_mean": float(self.salinity_mean),
            "S_std": float(self.salinity_std),
        }

    @property
    def temperature_scale(self) -> float:
        return max(float(self.temperature_std), 1e-12)

    @property
    def salinity_scale(self) -> float:
        return max(float(self.salinity_std), 1e-12)

    def normalize_temperature(self, values: ArrayLike) -> FloatArray:
        return (np.asarray(values, dtype=float) - self.temperature_mean) / self.temperature_scale

    def normalize_salinity(self, values: ArrayLike) -> FloatArray:
        return (np.asarray(values, dtype=float) - self.salinity_mean) / self.salinity_scale

    def denormalize_temperature(self, values: ArrayLike) -> FloatArray:
        return np.asarray(values, dtype=float) * self.temperature_scale + self.temperature_mean

    def denormalize_salinity(self, values: ArrayLike) -> FloatArray:
        return np.asarray(values, dtype=float) * self.salinity_scale + self.salinity_mean


def _to_1d_float(name: str, values: ArrayLike | None, n: int | None = None) -> FloatArray | None:
    if values is None:
        return None
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {arr.shape}.")
    if n is not None and arr.size != n:
        raise ValueError(f"{name} must have length {n}, got {arr.size}.")
    return arr.astype(float, copy=False)


@dataclass(slots=True)
class ProfileInput:
    """Sparse profile input for Tool 1.

    Parameters
    ----------
    pressure:
        Pressure levels in dbar. Must be one-dimensional.
    temperature, salinity:
        Observed sparse CTD-SRDL values.
    residual_t, residual_s:
        Residuals against a reference, e.g. GLORYS or a local profile composite.
        The model sees residuals, not the full reference profile.
    sigma_t, sigma_s:
        Local uncertainty/variability scales for the residuals. These should
        include observation error, reference error, and unresolved ocean variability.
    sigma_vert:
        RMS vertical heave scale in m or dbar. It describes expected vertical
        displacement of interfaces/stratification.
    sigma_heave_t, sigma_heave_s:
        Optional derived uncertainty from vertical heave: |dTref/dp| sigma_p and
        |dSref/dp| sigma_p. These can be supplied without exposing Tref/Sref.
    rho_ts:
        Optional local T-S residual correlation coefficient.
    day_of_year:
        Optional day of year. Encoded cyclically in the feature builder.
    profile_id:
        Optional identifier carried into outputs.
    attrs:
        Free metadata that is not directly used by the model.
    """

    pressure: FloatArray
    temperature: FloatArray
    salinity: FloatArray
    residual_t: FloatArray | None = None
    residual_s: FloatArray | None = None
    sigma_t: FloatArray | None = None
    sigma_s: FloatArray | None = None
    sigma_vert: FloatArray | None = None
    sigma_heave_t: FloatArray | None = None
    sigma_heave_s: FloatArray | None = None
    rho_ts: FloatArray | None = None
    day_of_year: float | None = None
    profile_id: str | None = None
    attrs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.pressure = _to_1d_float("pressure", self.pressure)  # type: ignore[assignment]
        n = self.pressure.size
        self.temperature = _to_1d_float("temperature", self.temperature, n)  # type: ignore[assignment]
        self.salinity = _to_1d_float("salinity", self.salinity, n)  # type: ignore[assignment]
        self.residual_t = _to_1d_float("residual_t", self.residual_t, n)
        self.residual_s = _to_1d_float("residual_s", self.residual_s, n)
        self.sigma_t = _to_1d_float("sigma_t", self.sigma_t, n)
        self.sigma_s = _to_1d_float("sigma_s", self.sigma_s, n)
        self.sigma_vert = _to_1d_float("sigma_vert", self.sigma_vert, n)
        self.sigma_heave_t = _to_1d_float("sigma_heave_t", self.sigma_heave_t, n)
        self.sigma_heave_s = _to_1d_float("sigma_heave_s", self.sigma_heave_s, n)
        self.rho_ts = _to_1d_float("rho_ts", self.rho_ts, n)
        self.validate()

    @property
    def n_levels(self) -> int:
        return int(self.pressure.size)

    @property
    def pmax(self) -> float:
        return float(np.nanmax(self.pressure))

    @property
    def pmin(self) -> float:
        return float(np.nanmin(self.pressure))

    def validate(self) -> None:
        if self.pressure.size < 2:
            raise ValueError("A profile needs at least two pressure levels.")
        if not np.any(np.isfinite(self.temperature)):
            raise ValueError("temperature contains no finite values.")
        if not np.any(np.isfinite(self.salinity)):
            raise ValueError("salinity contains no finite values.")
        if np.any(~np.isfinite(self.pressure)):
            raise ValueError("pressure must be finite.")
        if np.any(self.pressure < 0):
            raise ValueError("pressure must be non-negative.")
        if np.any(np.diff(self.pressure) < 0):
            raise ValueError("pressure must be sorted in increasing order. Use sorted_copy().")
        if self.day_of_year is not None and not (0.0 <= float(self.day_of_year) <= 366.0):
            raise ValueError("day_of_year must be between 0 and 366 when provided.")

    def sorted_copy(self) -> "ProfileInput":
        """Return a copy sorted by increasing pressure."""
        order = np.argsort(self.pressure)

        def take(arr: FloatArray | None) -> FloatArray | None:
            return None if arr is None else arr[order]

        return ProfileInput(
            pressure=self.pressure[order],
            temperature=self.temperature[order],
            salinity=self.salinity[order],
            residual_t=take(self.residual_t),
            residual_s=take(self.residual_s),
            sigma_t=take(self.sigma_t),
            sigma_s=take(self.sigma_s),
            sigma_vert=take(self.sigma_vert),
            sigma_heave_t=take(self.sigma_heave_t),
            sigma_heave_s=take(self.sigma_heave_s),
            rho_ts=take(self.rho_ts),
            day_of_year=self.day_of_year,
            profile_id=self.profile_id,
            attrs=dict(self.attrs),
        )

    @classmethod
    def from_dataframe(
        cls,
        df: Any,
        columns: Mapping[str, str] | None = None,
        **kwargs: Any,
    ) -> "ProfileInput":
        """Create a profile from a pandas-like dataframe.

        Parameters
        ----------
        df:
            Dataframe with one row per observed level.
        columns:
            Optional mapping from canonical names to dataframe columns. Canonical
            names are ``pressure``, ``temperature``, ``salinity``, ``residual_t``,
            ``residual_s``, ``sigma_t``, ``sigma_s``, ``sigma_vert``,
            ``sigma_heave_t``, ``sigma_heave_s``, and ``rho_ts``.
        kwargs:
            Additional ProfileInput fields, e.g. ``day_of_year`` or ``profile_id``.
        """
        default_cols = {
            "pressure": "pressure",
            "temperature": "temperature",
            "salinity": "salinity",
            "residual_t": "residual_t",
            "residual_s": "residual_s",
            "sigma_t": "sigma_t",
            "sigma_s": "sigma_s",
            "sigma_vert": "sigma_vert",
            "sigma_heave_t": "sigma_heave_t",
            "sigma_heave_s": "sigma_heave_s",
            "rho_ts": "rho_ts",
        }
        if columns is not None:
            default_cols.update(columns)

        values: dict[str, Any] = {}
        for key, col in default_cols.items():
            if col in df:
                values[key] = np.asarray(df[col], dtype=float)
        values.update(kwargs)
        return cls(**values)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dictionary."""
        out: dict[str, Any] = {}
        for key in (
            "pressure",
            "temperature",
            "salinity",
            "residual_t",
            "residual_s",
            "sigma_t",
            "sigma_s",
            "sigma_vert",
            "sigma_heave_t",
            "sigma_heave_s",
            "rho_ts",
        ):
            value = getattr(self, key)
            out[key] = None if value is None else value.tolist()
        out["day_of_year"] = self.day_of_year
        out["profile_id"] = self.profile_id
        out["attrs"] = dict(self.attrs)
        return out


@dataclass(slots=True)
class NuisanceBias:
    """Local linear-in-pressure nuisance fit.

    These parameters are used to make Tool 1 bias-insensitive. They are not the
    final tag calibration estimate.
    """

    a_t: float = np.nan
    b_t: float = np.nan
    a_s: float = np.nan
    b_s: float = np.nan
    uncertainty: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Result:
    """Structured output object"""

    profile_bad_probability: float
    point_bad_t: FloatArray
    point_bad_s: FloatArray
    point_density_inconsistent: FloatArray
    nuisance_bias: NuisanceBias
    correction_posterior: Any | None = None
    pressure_grid: FloatArray | None = None
    temperature_reconstructed: FloatArray | None = None
    salinity_reconstructed: FloatArray | None = None
    sigma_temperature: FloatArray | None = None
    sigma_salinity: FloatArray | None = None
    feature_names: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    profile_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dictionary."""

        def conv(value: Any) -> Any:
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, NuisanceBias):
                return value.as_dict()
            if hasattr(value, "as_dict") and callable(value.as_dict):
                return value.as_dict()
            if isinstance(value, dict):
                return {k: conv(v) for k, v in value.items()}
            if isinstance(value, list):
                return [conv(v) for v in value]
            if isinstance(value, np.generic):
                return value.item()
            return value

        return {key: conv(getattr(self, key)) for key in self.__dataclass_fields__}

    def summary(self) -> dict[str, Any]:
        """Compact summary useful for logs and dataframe rows."""
        return {
            "profile_id": self.profile_id,
            "profile_bad_probability": float(self.profile_bad_probability),
            "max_point_bad_t": float(np.nanmax(self.point_bad_t)),
            "max_point_bad_s": float(np.nanmax(self.point_bad_s)),
            "max_density_inconsistent": float(np.nanmax(self.point_density_inconsistent)),
            "n_points_flagged_t": int(np.sum(self.point_bad_t >= 0.5)),
            "n_points_flagged_s": int(np.sum(self.point_bad_s >= 0.5)),
            "a_t_local": float(self.nuisance_bias.a_t),
            "b_t_local": float(self.nuisance_bias.b_t),
            "a_s_local": float(self.nuisance_bias.a_s),
            "b_s_local": float(self.nuisance_bias.b_s),
            "correction_status": None
            if self.correction_posterior is None
            else getattr(self.correction_posterior, "status", None),
            "correction_prior_tension": None
            if self.correction_posterior is None
            else float(getattr(self.correction_posterior, "prior_tension", np.nan)),
            "correction_information_gain": None
            if self.correction_posterior is None
            else float(getattr(self.correction_posterior, "information_gain", np.nan)),
        }

    def probability_dict(self) -> dict[str, Any]:
        """Return the profile and point outlier probabilities as JSON-friendly data."""
        return probability_payload(
            profile_id=self.profile_id,
            profile_bad_probability=self.profile_bad_probability,
            point_bad_t=self.point_bad_t,
            point_bad_s=self.point_bad_s,
            point_density_inconsistent=self.point_density_inconsistent,
        )


def probability_payload(
    *,
    profile_id: str | None,
    profile_bad_probability: float,
    point_bad_t: ArrayLike,
    point_bad_s: ArrayLike,
    point_density_inconsistent: ArrayLike | None = None,
    plot_file: str | None = None,
    epoch: int | None = None,
    rank: int | None = None,
) -> dict[str, Any]:
    """Build a JSON-friendly payload for saved profile probabilities."""

    payload: dict[str, Any] = {
        "profile_id": profile_id,
        "profile_bad_probability": float(profile_bad_probability),
        "point_bad_t": np.asarray(point_bad_t, dtype=float).tolist(),
        "point_bad_s": np.asarray(point_bad_s, dtype=float).tolist(),
    }
    if point_density_inconsistent is not None:
        payload["point_density_inconsistent"] = np.asarray(point_density_inconsistent, dtype=float).tolist()
    if plot_file is not None:
        payload["plot_file"] = str(plot_file)
    if epoch is not None:
        payload["epoch"] = int(epoch)
    if rank is not None:
        payload["rank"] = int(rank)
    return payload
