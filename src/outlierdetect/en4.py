"""EN4 monthly NetCDF reader for clean-profile training sources.

The reader extracts one profile at a time from EN4 monthly files and returns
``ArgoProfile`` objects because the synthetic-training pipeline only needs the
common pressure/temperature/salinity and metadata fields.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from .argo import ArgoProfile, GOOD_QC_FLAGS
from .netcdf_backend import _get_qc, _get_var, _has_var, _open_nc, _profile_meta_value

PRESSURE_ADJUSTED_CANDIDATES: tuple[str, ...] = ("DEPH_CORRECTED", "PRES_ADJUSTED")
PRESSURE_RAW_CANDIDATES: tuple[str, ...] = ("PRES", "DEPTH")
TEMPERATURE_ADJUSTED_CANDIDATES: tuple[str, ...] = ("POTM_CORRECTED", "TEMP_ADJUSTED")
TEMPERATURE_RAW_CANDIDATES: tuple[str, ...] = ("TEMP",)
SALINITY_ADJUSTED_CANDIDATES: tuple[str, ...] = ("PSAL_CORRECTED", "PSAL_ADJUSTED")
SALINITY_RAW_CANDIDATES: tuple[str, ...] = ("PSAL",)

PROFILE_QC_CANDIDATES: tuple[str, ...] = (
    "PROFILE_DEPH_QC",
    "PROFILE_POTM_QC",
    "PROFILE_PSAL_QC",
    "JULD_QC",
    "POSITION_QC",
)


def read_en4_file(
    path: str | Path,
    good_qc_only: bool = True,
    good_qc_flags: frozenset[int] = GOOD_QC_FLAGS,
    min_levels: int = 5,
    profile_type: str = "adjusted",
    raw_fallback: bool = False,
    use_raw_values: bool = False,
) -> list[ArgoProfile]:
    """Read all EN4 profiles from a single monthly NetCDF file."""
    path = Path(path)
    ds = _open_nc(path)
    try:
        effective_good_qc_only = good_qc_only and not use_raw_values
        profiles = _parse_profiles(
            ds,
            path.stem,
            effective_good_qc_only,
            good_qc_flags,
            min_levels,
            profile_type=profile_type,
            raw_fallback=raw_fallback,
            use_raw_values=use_raw_values,
        )
        if effective_good_qc_only and not profiles:
            profiles = _parse_profiles(
                ds,
                path.stem,
                False,
                good_qc_flags,
                min_levels,
                profile_type=profile_type,
                raw_fallback=raw_fallback,
                use_raw_values=use_raw_values,
            )
    finally:
        try:
            ds.close()
        except Exception:
            pass
    return profiles


def iter_en4_files(
    root: str | Path,
    pattern: str = "**/*.nc",
    good_qc_only: bool = True,
    min_levels: int = 5,
    profile_type: str = "adjusted",
    raw_fallback: bool = False,
    use_raw_values: bool = False,
) -> Iterator[ArgoProfile]:
    """Recursively yield EN4 profiles from all NetCDF files under *root*."""
    root = Path(root)
    if root.is_file():
        files = [root]
    else:
        files = sorted(root.glob(pattern))

    for nc_path in files:
        try:
            for prof in read_en4_file(
                nc_path,
                good_qc_only=good_qc_only,
                min_levels=min_levels,
                profile_type=profile_type,
                raw_fallback=raw_fallback,
                use_raw_values=use_raw_values,
            ):
                yield prof
        except Exception as exc:
            import warnings

            warnings.warn(f"Skipping {nc_path}: {exc}", stacklevel=2)


def _parse_profiles(
    ds,
    stem: str,
    good_qc_only: bool,
    good_qc_flags: frozenset[int],
    min_levels: int,
    *,
    profile_type: str = "adjusted",
    raw_fallback: bool = False,
    use_raw_values: bool = False,
) -> list[ArgoProfile]:
    profile_type = _normalize_profile_type(profile_type)
    source_fields = _source_fields_for_profile_type(
        profile_type,
        raw_fallback=raw_fallback,
        use_raw_values=use_raw_values,
    )

    pressure_name, pres_raw = _select_required_array(ds, source_fields["pressure"], "pressure")
    temp_name, temp_raw = _select_required_array(ds, source_fields["temperature"], "temperature")
    sal_name, sal_raw = _select_required_array(ds, source_fields["salinity"], "salinity")

    pres_raw = _ensure_profile_matrix(pres_raw, pressure_name)
    temp_raw = _ensure_profile_matrix(temp_raw, temp_name)
    sal_raw = _ensure_profile_matrix(sal_raw, sal_name)
    if pres_raw.shape != temp_raw.shape or pres_raw.shape != sal_raw.shape:
        raise ValueError(
            "EN4 pressure, temperature, and salinity variables must have matching shapes."
        )

    n_prof, n_levels = pres_raw.shape
    cycle_numbers = _profile_values(ds, "CYCLE_NUMBER", n_prof)
    juldays = _profile_values(ds, "JULD", n_prof)
    latitudes = _profile_values(ds, "LATITUDE", n_prof)
    longitudes = _profile_values(ds, "LONGITUDE", n_prof)

    pressure_qc = None
    temp_qc = None
    sal_qc = None
    if good_qc_only:
        pressure_qc = _select_level_qc(ds, _level_qc_candidates(pressure_name), n_prof, n_levels)
        temp_qc = _select_level_qc(ds, _level_qc_candidates(temp_name), n_prof, n_levels)
        sal_qc = _select_level_qc(ds, _level_qc_candidates(sal_name), n_prof, n_levels)

    profiles: list[ArgoProfile] = []
    for k in range(n_prof):
        if good_qc_only and not _profile_is_good(ds, k, good_qc_flags):
            continue

        p = pres_raw[k].copy()
        t = temp_raw[k].copy()
        s = sal_raw[k].copy()

        if good_qc_only:
            p = _apply_level_qc(p, None if pressure_qc is None else pressure_qc[k], good_qc_flags)
            t = _apply_level_qc(t, None if temp_qc is None else temp_qc[k], good_qc_flags)
            s = _apply_level_qc(s, None if sal_qc is None else sal_qc[k], good_qc_flags)

        ok = np.isfinite(p) & np.isfinite(t) & np.isfinite(s)
        p, t, s = p[ok], t[ok], s[ok]
        if p.size < min_levels:
            continue

        order = np.argsort(p)
        p, t, s = p[order], t[order], s[order]

        cycle = None if cycle_numbers[k] is None else int(round(cycle_numbers[k]))
        cycle_str = f"{cycle:04d}" if cycle is not None else f"{k:04d}"
        profile_id = f"{stem}_{cycle_str}"

        prof = ArgoProfile(
            profile_id=profile_id,
            pressure=p,
            temperature=t,
            salinity=s,
            cycle_number=cycle,
            float_wmo=None,
            juld=juldays[k],
            latitude=latitudes[k],
            longitude=longitudes[k],
        )
        if prof.is_valid(min_levels):
            profiles.append(prof)

    return profiles


def _normalize_profile_type(value: str | None) -> str:
    if value is None:
        return "adjusted"
    mode = str(value).strip().lower()
    if mode not in {"adjusted", "raw"}:
        raise ValueError("profile_type must be either 'adjusted' or 'raw'.")
    return mode


def _source_fields_for_profile_type(
    profile_type: str,
    *,
    raw_fallback: bool,
    use_raw_values: bool,
) -> dict[str, tuple[str, ...]]:
    if use_raw_values:
        return {
            "pressure": PRESSURE_ADJUSTED_CANDIDATES,
            "temperature": TEMPERATURE_RAW_CANDIDATES,
            "salinity": SALINITY_RAW_CANDIDATES,
        }
    if profile_type == "raw":
        return {
            "pressure": PRESSURE_RAW_CANDIDATES,
            "temperature": TEMPERATURE_RAW_CANDIDATES,
            "salinity": SALINITY_RAW_CANDIDATES,
        }
    return {
        "pressure": PRESSURE_ADJUSTED_CANDIDATES + (PRESSURE_RAW_CANDIDATES if raw_fallback else ()),
        "temperature": TEMPERATURE_ADJUSTED_CANDIDATES + (TEMPERATURE_RAW_CANDIDATES if raw_fallback else ()),
        "salinity": SALINITY_ADJUSTED_CANDIDATES + (SALINITY_RAW_CANDIDATES if raw_fallback else ()),
    }


def _select_required_array(ds, candidates: tuple[str, ...], label: str) -> tuple[str, np.ndarray]:
    errors: list[str] = []
    for name in candidates:
        if _has_var(ds, name):
            try:
                return name, _get_var(ds, name)
            except Exception as exc:
                errors.append(f"{name}: {exc}")
    if errors:
        raise RuntimeError(
            f"Could not read an EN4 {label} variable from {candidates!r}: " + "; ".join(errors)
        )
    raise KeyError(f"Could not find an EN4 {label} variable in {candidates!r}.")


def _ensure_profile_matrix(values: np.ndarray, label: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        return arr[np.newaxis, :]
    if arr.ndim != 2:
        raise ValueError(f"{label} must be one- or two-dimensional, got shape {arr.shape}.")
    return arr


def _profile_values(ds, name: str, n_prof: int) -> list[float | None]:
    if not _has_var(ds, name):
        return [None] * n_prof
    try:
        raw = np.asarray(ds[name][:], dtype=float)
    except Exception:
        return [None] * n_prof
    values: list[float | None] = []
    for k in range(n_prof):
        values.append(_profile_meta_value(raw, k))
    return values


def _level_qc_candidates(var_name: str) -> tuple[str, ...]:
    if var_name == "DEPH_CORRECTED":
        return ("DEPH_CORRECTED_QC", "PRES_QC", "DEPTH_QC")
    if var_name == "PRES_ADJUSTED":
        return ("PRES_ADJUSTED_QC", "PRES_QC")
    if var_name == "PRES":
        return ("PRES_QC", "PRES_ADJUSTED_QC")
    if var_name == "DEPTH":
        return ("DEPTH_QC", "PRES_QC")
    if var_name == "POTM_CORRECTED":
        return ("POTM_CORRECTED_QC", "TEMP_ADJUSTED_QC", "TEMP_QC")
    if var_name == "TEMP_ADJUSTED":
        return ("TEMP_ADJUSTED_QC", "TEMP_QC")
    if var_name == "TEMP":
        return ("TEMP_QC", "TEMP_ADJUSTED_QC")
    if var_name == "PSAL_CORRECTED":
        return ("PSAL_CORRECTED_QC", "PSAL_ADJUSTED_QC", "PSAL_QC")
    if var_name == "PSAL_ADJUSTED":
        return ("PSAL_ADJUSTED_QC", "PSAL_QC")
    if var_name == "PSAL":
        return ("PSAL_QC", "PSAL_ADJUSTED_QC")
    return (f"{var_name}_QC",)


def _select_level_qc(
    ds,
    candidates: tuple[str, ...],
    n_prof: int,
    n_levels: int,
) -> np.ndarray | None:
    for name in candidates:
        if not _has_var(ds, name):
            continue
        try:
            qc = _get_qc(ds, name)
        except Exception:
            continue
        if qc is None:
            continue
        arr = np.asarray(qc)
        if arr.ndim == 1:
            if n_prof == 1 and arr.shape[0] == n_levels:
                return arr[np.newaxis, :]
            continue
        if arr.shape[0] != n_prof:
            continue
        if arr.shape[1] != n_levels:
            if n_prof == 1 and arr.shape[0] == 1 and arr.shape[1] == n_levels:
                return arr
            continue
        return arr
    return None


def _profile_is_good(ds, index: int, good_qc_flags: frozenset[int]) -> bool:
    for name in PROFILE_QC_CANDIDATES:
        if not _has_var(ds, name):
            continue
        try:
            qc = _get_qc(ds, name)
        except Exception:
            continue
        if qc is None:
            continue
        flag = _profile_qc_value(qc, index)
        if flag is None:
            continue
        if int(flag) not in good_qc_flags:
            return False
    return True


def _profile_qc_value(qc: np.ndarray, index: int) -> int | None:
    arr = np.asarray(qc)
    if arr.ndim == 0:
        try:
            value = int(arr.item())
        except Exception:
            return None
        return None if value in {0, 9} else value
    if index >= arr.shape[0]:
        return None
    sample = np.asarray(arr[index]).reshape(-1)
    if sample.size == 0:
        return None
    try:
        value = int(sample[0])
    except Exception:
        finite = sample[np.isfinite(sample)]
        if finite.size == 0:
            return None
        value = int(finite[0])
    return None if value in {0, 9} else value


def _apply_level_qc(values: np.ndarray, qc: np.ndarray | None, good_qc_flags: frozenset[int]) -> np.ndarray:
    arr = np.asarray(values, dtype=float).copy()
    if qc is None:
        return arr
    flags = np.asarray(qc).reshape(-1)
    if flags.size != arr.size:
        return arr
    bad = ~np.isin(flags.astype(int), list(good_qc_flags))
    arr[bad] = np.nan
    return arr
