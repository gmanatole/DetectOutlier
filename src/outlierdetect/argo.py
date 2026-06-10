"""Argo float NetCDF reader for Tool 1 training.

Reads core-Argo profile files (one file = one float, many profiles) or
mono-profile files. Extracts PRES / TEMP_ADJUSTED / PSAL_ADJUSTED, applies
QC-flag filtering, and returns clean arrays that can be fed directly into
``degrade_highres_profile``.

NetCDF backend priority (first available wins)
----------------------------------------------
1. ``netCDF4``    – fastest, handles both CLASSIC and NetCDF-4/HDF5.
2. ``h5py``       – handles NetCDF-4/HDF5 directly.
3. ``scipy.io``   – handles NetCDF-3 CLASSIC only (many older Argo files).

Install at least one:
    pip install netCDF4          # recommended
    pip install h5py             # alternative for HDF5-backed files
    # scipy is usually present already
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

# ---------------------------------------------------------------------------
# NetCDF backend detection
# ---------------------------------------------------------------------------

def _open_nc(path: str | Path):
    """Return an object that supports dict-like variable access.

    Tries backends in order and returns the first that succeeds.
    The returned object is kept open; close it when done.
    """
    path = str(path)

    # 1. netCDF4 (handles NC3 + NC4/HDF5)
    try:
        import netCDF4  # noqa: PLC0415
        return netCDF4.Dataset(path, "r")
    except ImportError:
        pass
    except Exception as exc:
        raise RuntimeError(f"netCDF4 failed to open {path}: {exc}") from exc

    # 2. h5py (NC4/HDF5 only, but no HDF5 dimension-scale metadata)
    try:
        import h5py  # noqa: PLC0415
        return h5py.File(path, "r")
    except ImportError:
        pass
    except Exception:
        pass  # not HDF5 – fall through to scipy

    # 3. scipy (NC3 CLASSIC only)
    try:
        from scipy.io import netcdf_file  # noqa: PLC0415
        return netcdf_file(path, "r", mmap=False)
    except Exception as exc:
        raise RuntimeError(
            f"Could not open {path} with any available NetCDF backend.\n"
            "Install a backend: pip install netCDF4   (recommended)\n"
            "                   pip install h5py      (HDF5/NC4 files)\n"
            f"Original error: {exc}"
        ) from exc


def _get_var(ds, name: str) -> np.ndarray:
    """Extract a variable as a plain numpy float64 array, handling fill values."""
    var = ds[name]
    # netCDF4 Dataset, h5py Dataset, and scipy netcdf_variable all support [:]
    data = np.asarray(var[:], dtype=np.float64)

    # Mask fill / missing values → NaN
    fill_candidates = []
    for attr in ("_FillValue", "missing_value", "fill_value"):
        try:
            fill_candidates.append(float(getattr(var, attr, None) or np.nan))
        except Exception:
            pass
    for fv in fill_candidates:
        if np.isfinite(fv):
            data[data == fv] = np.nan

    # netCDF4 masked arrays
    if hasattr(data, "filled"):
        data = data.filled(np.nan)

    return data


def _get_qc(ds, name: str) -> np.ndarray | None:
    """Return QC flag array as uint8, or None if variable does not exist."""
    if not _has_var(ds, name):
        return None
    raw = np.asarray(ds[name][:])
    # QC flags may be stored as bytes (b'1') or characters or integers
    if raw.dtype.kind in ("S", "U", "O"):  # byte-string / str
        flat = raw.flatten()
        out = np.zeros(flat.shape, dtype=np.uint8)
        for i, v in enumerate(flat):
            try:
                out[i] = int(v) if v not in (b"", "", None) else 9
            except Exception:
                out[i] = 9
        return out.reshape(raw.shape)
    if raw.dtype.kind in ("i", "u", "f"):
        return raw.astype(np.uint8)
    return None


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------

#: QC flags considered "good" for Argo (ADMT standard)
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
        """Convert this Argo profile into a :class:`~outlierdetect.data.ProfileInput`.

        The inference path uses this to turn raw Argo or parquet-backed profiles
        into the sparse profile object consumed by the model.
        """
        from .data import ProfileInput

        pressure = np.asarray(self.pressure, dtype=float)
        temperature = np.asarray(self.temperature, dtype=float)
        salinity = np.asarray(self.salinity, dtype=float)

        valid = np.isfinite(pressure) & np.isfinite(temperature) & np.isfinite(salinity) & (pressure >= 0.0)
        if int(np.sum(valid)) < 2:
            raise ValueError("Profile must contain at least two finite non-negative levels.")

        if day_of_year is None and self.juld is not None and np.isfinite(self.juld):
            day_of_year = float(np.mod(self.juld, 365.2425))

        profile_attrs: dict[str, Any] = {
            "float_wmo": self.float_wmo,
            "cycle_number": self.cycle_number,
            "juld": self.juld,
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


# ---------------------------------------------------------------------------
# Core reader
# ---------------------------------------------------------------------------

def read_argo_file(
    path: str | Path,
    good_qc_only: bool = True,
    good_qc_flags: frozenset[int] = GOOD_QC_FLAGS,
    min_levels: int = 5,
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
        profiles = _parse_profiles(ds, path.stem, good_qc_only, good_qc_flags, min_levels)
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
) -> list[ArgoProfile]:
    """Internal: extract profiles from an open dataset handle."""

    # ------------------------------------------------------------------ dims
    # Argo files are either:
    #   (N_PROF, N_LEVELS) – multi-profile "Sprof" or "Rprof"
    #   (N_LEVELS,)        – single-profile mono files
    pres_raw = _get_var(ds, "PRES")
    temp_raw = _get_var(ds, "TEMP_ADJUSTED")
    psal_raw = _get_var(ds, "PSAL_ADJUSTED")

    two_d = pres_raw.ndim == 2
    if pres_raw.ndim == 1:
        pres_raw = pres_raw[np.newaxis, :]
        temp_raw = temp_raw[np.newaxis, :]
        psal_raw = psal_raw[np.newaxis, :]

    n_prof, _ = pres_raw.shape

    # ------------------------------------------------------------------ QC
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

    # ------------------------------------------------------------------ build
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


# ---------------------------------------------------------------------------
# Directory scanner
# ---------------------------------------------------------------------------

def iter_argo_files(
    root: str | Path,
    pattern: str = "**/*.nc",
    good_qc_only: bool = True,
    min_levels: int = 5,
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
            for prof in read_argo_file(nc_path, good_qc_only=good_qc_only,
                                       min_levels=min_levels):
                yield prof
        except Exception as exc:
            import warnings
            warnings.warn(f"Skipping {nc_path}: {exc}", stacklevel=2)


# ---------------------------------------------------------------------------
# Random subsampling for inference (any-size profile → sparse CTD-SRDL-like)
# ---------------------------------------------------------------------------

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
