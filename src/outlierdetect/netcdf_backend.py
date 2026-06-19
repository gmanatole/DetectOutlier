"""Shared NetCDF opener and array helpers.

This module keeps the file-format mechanics separate from the profile-specific
Argo and EN4 readers. It knows how to open a NetCDF file and normalize common
array/QC access patterns, but it does not know anything about training logic or
profile semantics.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

_TRUNCATED_HDF5_RE = re.compile(
    r"truncated file:\s+eof\s*=\s*(?P<eof>\d+).*?stored_eof\s*=\s*(?P<stored_eof>\d+)",
    re.IGNORECASE | re.DOTALL,
)


def _open_nc(path: str | Path):
    """Return a dataset handle using the first working NetCDF backend."""

    path = str(path)
    path_obj = Path(path)
    errors: list[str] = []
    h5py_error: Exception | None = None

    try:
        import netCDF4

        return netCDF4.Dataset(path, "r")
    except ImportError:
        pass
    except Exception as exc:
        errors.append(f"netCDF4: {exc}")

    try:
        import h5py

        return h5py.File(path, "r")
    except ImportError:
        pass
    except Exception as exc:
        errors.append(f"h5py: {exc}")
        h5py_error = exc

    try:
        from scipy.io import netcdf_file  # noqa: PLC0415

        return netcdf_file(path, "r", mmap=False)
    except Exception as exc:
        errors.append(f"scipy.io.netcdf_file: {exc}")

    padded = _open_padded_truncated_hdf5(path_obj, h5py_error)
    if padded is not None:
        return padded

    raise RuntimeError(
        f"Could not open {path} with any available NetCDF backend: " + "; ".join(errors)
    )


def _open_padded_truncated_hdf5(path: Path, error: Exception | None):
    """Retry a truncated HDF5-backed NetCDF file by padding it to the stored EOF."""

    if error is None:
        return None
    match = _TRUNCATED_HDF5_RE.search(str(error))
    if match is None:
        return None

    stored_eof = int(match.group("stored_eof"))
    actual_size = path.stat().st_size
    if stored_eof <= actual_size:
        return None

    raw = path.read_bytes()
    if len(raw) < stored_eof:
        raw = raw + b"\x00" * (stored_eof - len(raw))

    try:
        import netCDF4

        return netCDF4.Dataset("inmemory.nc", "r", memory=raw)
    except Exception:
        return None


def _has_var(ds, name: str) -> bool:
    """Return True when *name* is present as a data variable."""

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


def _get_var(ds, name: str) -> np.ndarray:
    """Extract a variable as a plain float64 array, handling fill values."""

    var = ds[name]
    data = np.asarray(var[:], dtype=np.float64)

    fill_candidates = []
    for attr in ("_FillValue", "missing_value", "fill_value"):
        try:
            fill_candidates.append(float(getattr(var, attr, None) or np.nan))
        except Exception:
            pass
    for fv in fill_candidates:
        if np.isfinite(fv):
            data[data == fv] = np.nan

    if hasattr(data, "filled"):
        data = data.filled(np.nan)

    return data


def _get_qc(ds, name: str) -> np.ndarray | None:
    """Return QC flag array as uint8, or None if variable does not exist."""

    if not _has_var(ds, name):
        return None
    raw = np.asarray(ds[name][:])
    if raw.dtype.kind in ("S", "U", "O"):
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


def _profile_meta_value(raw: np.ndarray, index: int) -> float | None:
    """Return a profile-level metadata value from a possibly nested array."""

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
