"""Argo float reader and processor for training.

This module owns the Argo-specific profile structure, file traversal, and
pressure subsampling logic. NetCDF opening and array normalization live in
``outlierdetect.netcdf_backend`` so EN4 and Argo can share the same low-level
I/O backend without the training stack traces looking Argo-specific.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from numpy.typing import NDArray

from .netcdf_backend import _get_qc, _get_var, _has_var, _open_nc, _profile_meta_value

FloatArray = NDArray[np.float64]

# Public data structures
# QC flags considered "good" for Argo (ADMT standard)
GOOD_QC_FLAGS: frozenset[int] = frozenset({1, 2})


@dataclass
class ArgoProfile:
    """One vertical profile from an Argo float file."""

    profile_id: str          # e.g. "5904989_042" (float_wmo_cycle)
    pressure: FloatArray     # dbar, increasing, NaN stripped
    temperature: FloatArray  # °C, TEMP_ADJUSTED
    salinity: FloatArray     # PSU, PSAL_ADJUSTED
    n_levels: int = field(init=False)
    cycle_number: int | None = None
    float_wmo: str | None = None
    juld: float | None = None   # Julian days since 1950-01-01
    latitude: float | None = None
    longitude: float | None = None

    def __post_init__(self) -> None:
        self.n_levels = int(self.pressure.size)

    def is_valid(self, min_levels: int = 5) -> bool:
        """True if the profile has enough finite data points."""
        ok = (
            np.isfinite(self.pressure)
            & np.isfinite(self.temperature)
            & np.isfinite(self.salinity)
        )
        return int(ok.sum()) >= min_levels

    def to_profile_input(
        self,
        *,
        day_of_year: float | None = None,
        attrs: dict[str, Any] | None = None,
    ) -> "ProfileInput":
        """Convert this Argo profile into a :class: outlierdetect.data.ProfileInput.

        The inference path uses this to turn raw Argo or parquet-backed profiles
        into the sparse profile object consumed by the model.
        """
        from .data import ProfileInput

        pressure = np.asarray(self.pressure, dtype=float)
        temperature = np.asarray(self.temperature, dtype=float)
        salinity = np.asarray(self.salinity, dtype=float)

        # Check profile validity
        valid = np.isfinite(pressure) & np.isfinite(temperature) & np.isfinite(salinity) & (pressure >= 0.0)
        if int(np.sum(valid)) < 2:
            raise ValueError("Profile must contain at least two finite non-negative levels.")

        # Get doy array
        if day_of_year is None and self.juld is not None and np.isfinite(self.juld):
            day_of_year = float(np.mod(self.juld, 365.2425))

        profile_attrs: dict[str, Any] = {
            "float_wmo": self.float_wmo,
            "cycle_number": self.cycle_number,
            "juld": self.juld,
            "latitude": self.latitude,
            "longitude": self.longitude,
        }
        if attrs:
            profile_attrs.update(attrs)
        profile_attrs = {key: value for key, value in profile_attrs.items() if value is not None}

        return ProfileInput(
            pressure=pressure[valid],
            temperature=temperature[valid],
            salinity=salinity[valid],
            day_of_year=day_of_year,
            profile_id=self.profile_id,
            attrs=profile_attrs,
        )



# Core reader

def read_argo_file(
    path: str | Path,
    good_qc_only: bool = True,
    good_qc_flags: frozenset[int] = GOOD_QC_FLAGS,
    min_levels: int = 5,
    use_raw_values: bool = False,
) -> list[ArgoProfile]:
    """Read all profiles from a single Argo NetCDF file.

    Parameters
    ----------
    path:
        Path to an Argo NetCDF file (core profile, e.g. ``SD5904989_Sprof.nc``
        or ``5904989_042.nc``).
    good_qc_only:
        If True (default), set data to NaN where QC flags are not in
        ``good_qc_flags``.  Disabling this passes raw adjusted values through
        without flag-based masking.
    use_raw_values:
        If True, read ``TEMP`` and ``PSAL`` directly and skip QC masking.
    good_qc_flags:
        Set of Argo QC flag values considered acceptable (default: {1, 2}).
    min_levels:
        Discard profiles with fewer than this many finite data points after QC
        filtering (default: 5).

    Returns
    -------
    list[ArgoProfile]
        One entry per profile in the file; may be empty.
    """
    path = Path(path)
    ds = _open_nc(path)
    try:
        profiles = _parse_profiles(
            ds,
            path.stem,
            good_qc_only and not use_raw_values,
            good_qc_flags,
            min_levels,
            use_raw_values=use_raw_values,
        )
    finally:
        try:
            ds.close()
        except Exception:
            pass
    return profiles


def _parse_profiles(
    ds,
    stem: str,
    good_qc_only: bool,
    good_qc_flags: frozenset[int],
    min_levels: int,
    *,
    use_raw_values: bool = False,
) -> list[ArgoProfile]:
    """Internal: extract profiles from an open dataset handle."""

    # ------------------------------------------------------------------ dims
    # Argo files are either:
    #   (N_PROF, N_LEVELS) – multi-profile "Sprof" or "Rprof"
    #   (N_LEVELS,)        – single-profile mono files
    pres_raw = _get_var(ds, "PRES")
    if use_raw_values:
        temp_raw = _get_var(ds, "TEMP")
        psal_raw = _get_var(ds, "PSAL")
    else:
        temp_raw = _get_var(ds, "TEMP_ADJUSTED")
        psal_raw = _get_var(ds, "PSAL_ADJUSTED")

    two_d = pres_raw.ndim == 2
    if pres_raw.ndim == 1:
        pres_raw = pres_raw[np.newaxis, :]
        temp_raw = temp_raw[np.newaxis, :]
        psal_raw = psal_raw[np.newaxis, :]

    n_prof, _ = pres_raw.shape

    # ------------------------------------------------------------------ QC
    if use_raw_values:
        pres_qc = None
        temp_qc = None
        psal_qc = None
    else:
        pres_qc = _get_qc(ds, "PRES_QC")
        temp_qc = _get_qc(ds, "TEMP_ADJUSTED_QC")
        psal_qc = _get_qc(ds, "PSAL_ADJUSTED_QC")

    if pres_qc is not None and pres_qc.ndim == 1 and two_d:
        pres_qc = pres_qc[np.newaxis, :]
    if temp_qc is not None and temp_qc.ndim == 1 and two_d:
        temp_qc = temp_qc[np.newaxis, :]
    if psal_qc is not None and psal_qc.ndim == 1 and two_d:
        psal_qc = psal_qc[np.newaxis, :]

    # ------------------------------------------------------------------ meta
    cycle_numbers: list[int | None] = [None] * n_prof
    juldays: list[float | None] = [None] * n_prof
    latitudes: list[float | None] = [None] * n_prof
    longitudes: list[float | None] = [None] * n_prof
    float_wmo: str | None = None

    if _has_var(ds, "CYCLE_NUMBER"):
        cn = np.asarray(ds["CYCLE_NUMBER"][:]).flatten()
        for k in range(min(n_prof, len(cn))):
            try:
                cycle_numbers[k] = int(cn[k])
            except Exception:
                pass

    if _has_var(ds, "JULD"):
        jd = _get_var(ds, "JULD").flatten()
        for k in range(min(n_prof, len(jd))):
            if np.isfinite(jd[k]):
                juldays[k] = float(jd[k])

    if _has_var(ds, "LATITUDE"):
        lat_raw = np.asarray(ds["LATITUDE"][:], dtype=float)
        for k in range(n_prof):
            latitudes[k] = _profile_meta_value(lat_raw, k)

    if _has_var(ds, "LONGITUDE"):
        lon_raw = np.asarray(ds["LONGITUDE"][:], dtype=float)
        for k in range(n_prof):
            longitudes[k] = _profile_meta_value(lon_raw, k)

    for attr in ("PLATFORM_NUMBER", "PLATFORM_CODE"):
        if _has_var(ds, attr):
            raw = ds[attr]
            try:
                v = np.asarray(raw[:])
                if v.ndim >= 1:
                    v = v[0]
                float_wmo = "".join(chr(c) if isinstance(c, int) else str(c, "ascii", "ignore")
                                    for c in np.asarray(v).flatten()).strip()
                if float_wmo:
                    break
            except Exception:
                pass
    if not float_wmo:
        float_wmo = stem

    # build
    profiles: list[ArgoProfile] = []
    for k in range(n_prof):
        p = pres_raw[k].copy()
        t = temp_raw[k].copy()
        s = psal_raw[k].copy()

        if good_qc_only:
            if pres_qc is not None:
                bad = ~np.isin(pres_qc[k].astype(int), list(good_qc_flags))
                p[bad] = np.nan
            if temp_qc is not None:
                bad = ~np.isin(temp_qc[k].astype(int), list(good_qc_flags))
                t[bad] = np.nan
            if psal_qc is not None:
                bad = ~np.isin(psal_qc[k].astype(int), list(good_qc_flags))
                s[bad] = np.nan

        # Keep only levels where all three are finite
        ok = np.isfinite(p) & np.isfinite(t) & np.isfinite(s)
        p, t, s = p[ok], t[ok], s[ok]

        # Sort by increasing pressure
        order = np.argsort(p)
        p, t, s = p[order], t[order], s[order]

        cycle = cycle_numbers[k]
        cycle_str = f"{cycle:03d}" if cycle is not None else f"{k:04d}"
        pid = f"{float_wmo}_{cycle_str}"

        prof = ArgoProfile(
            profile_id=pid,
            pressure=p,
            temperature=t,
            salinity=s,
            cycle_number=cycle,
            float_wmo=float_wmo,
            juld=juldays[k],
            latitude=latitudes[k],
            longitude=longitudes[k],
        )
        if prof.is_valid(min_levels):
            profiles.append(prof)

    return profiles


def _has_var(ds, name: str) -> bool:
    """Return True when *name* is present as a data variable.

    Different NetCDF backends expose slightly different container semantics.
    Checking ``ds.variables`` is safe for netCDF4, scipy, and h5py-backed
    datasets, while falling back to mapping membership keeps plain dict-like
    test doubles working.
    """
    variables = getattr(ds, "variables", None)
    if variables is not None:
        try:
            return name in variables
        except Exception:
            pass
    try:
        return name in ds
    except Exception:
        return False


def _profile_meta_value(raw: np.ndarray, index: int) -> float | None:
    """Return a profile-level metadata value from a potentially multi-dimensional array."""
    arr = np.asarray(raw, dtype=float)
    if arr.ndim == 0:
        return float(arr) if np.isfinite(arr) else None
    if index >= arr.shape[0]:
        return None
    sample = np.asarray(arr[index], dtype=float).reshape(-1)
    finite = sample[np.isfinite(sample)]
    if finite.size == 0:
        return None
    return float(finite[0])


def iter_argo_files(
    root: str | Path,
    pattern: str = "**/*.nc",
    good_qc_only: bool = True,
    min_levels: int = 5,
    use_raw_values: bool = False,
) -> Iterator[ArgoProfile]:
    """Recursively yield ArgoProfile objects from all .nc files under *root*.

    Parameters
    ----------
    root:
        Top-level directory to search (or a single .nc file path).
    pattern:
        Glob pattern relative to *root* (default: ``**/*.nc``).
    good_qc_only, min_levels:
        Forwarded to :func:`read_argo_file`.
    use_raw_values:
        If True, prefer raw ``TEMP``/``PSAL`` values and skip QC masking.

    Yields
    ------
    ArgoProfile
        One profile at a time (lazy; does not load all files into memory).
    """
    root = Path(root)
    if root.is_file():
        files = [root]
    else:
        files = sorted(root.glob(pattern))

    for nc_path in files:
        try:
            for prof in read_argo_file(
                nc_path,
                good_qc_only=good_qc_only,
                min_levels=min_levels,
                use_raw_values=use_raw_values,
            ):
                yield prof
        except Exception as exc:
            import warnings
            warnings.warn(f"Skipping {nc_path}: {exc}", stacklevel=2)


def subsample_profile(
    pressure: FloatArray,
    temperature: FloatArray,
    salinity: FloatArray,
    n_levels: int = 20,
    rng: np.random.Generator | None = None,
    upper_ocean_bias: float = 1.7,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Randomly subsample a dense profile to ``n_levels`` sparse levels.

    Replicates the CTD-SRDL-like sampling pattern used in
    ``degrade_highres_profile``:  levels are drawn with a power-law bias
    toward the upper ocean, always including the shallowest and deepest levels.

    This is also the correct call to make **at inference time** when you have
    a full-resolution Argo (or CTD) profile and want to test the trained model
    on a sparse version — it gives a profile with the same level-count
    distribution the model was trained on.

    Parameters
    ----------
    pressure, temperature, salinity:
        Arrays of length ≥ ``n_levels``, already sorted by increasing pressure
        and NaN-free.
    n_levels:
        Target number of levels (default: 20). Clamped to ``len(pressure)``
        if the profile is shorter.
    rng:
        NumPy random generator; created freshly if None.
    upper_ocean_bias:
        Exponent for the power-law draw.  Values > 1 concentrate more levels
        in the upper ocean (default 1.7 matches training).

    Returns
    -------
    p_sparse, t_sparse, s_sparse : FloatArray
        Subsampled and sorted arrays of length ``min(n_levels, len(pressure))``.
    """
    indices = sample_pressure_indices(p=pressure, n_levels=n_levels, rng=rng, upper_ocean_bias=upper_ocean_bias)
    p = np.asarray(pressure, dtype=float)
    t = np.asarray(temperature, dtype=float)
    s = np.asarray(salinity, dtype=float)
    return p[indices], t[indices], s[indices]


def sample_pressure_indices(
    p: ArrayLike,
    n_levels: int = 20,
    rng: np.random.Generator | None = None,
    upper_ocean_bias: float = 1.7,
) -> NDArray[np.int64]:
    """Sample exact profile indices with an upper-ocean bias.

    The returned indices always include the shallowest and deepest valid levels
    and are sorted in increasing-pressure order.
    """
    rng = np.random.default_rng() if rng is None else rng
    pressure = np.asarray(p, dtype=float)
    n = pressure.size
    n_out = min(int(n_levels), n)
    if n_out < 2:
        raise ValueError("n_levels must be at least 2.")
    if n_out == n:
        return np.arange(n, dtype=np.int64)

    # Sample without replacement from the interior with a bias toward the upper
    # ocean. We use actual pressure rather than index position so irregularly
    # spaced profiles still work.
    p_norm = (pressure - pressure[0]) / max(float(pressure[-1] - pressure[0]), 1.0)
    weights = np.maximum(1.0 - p_norm, 0.05) ** upper_ocean_bias
    weights[0] = 0.0
    weights[-1] = 0.0
    interior = np.arange(1, n - 1, dtype=np.int64)
    if n_out == 2:
        return np.array([0, n - 1], dtype=np.int64)

    chosen = rng.choice(
        interior,
        size=n_out - 2,
        replace=False,
        p=weights[interior] / np.sum(weights[interior]),
    )
    indices = np.sort(np.concatenate([np.array([0, n - 1], dtype=np.int64), chosen]))
    return indices
