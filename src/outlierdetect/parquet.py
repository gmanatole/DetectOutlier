"""Utilities for exporting Argo profile data to parquet."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import warnings

import numpy as np

from .argo import ArgoProfile, read_argo_file


@dataclass(slots=True)
class ArgoParquetSummary:
    """Summary of a parquet export run."""

    output: str
    n_files: int
    n_profiles: int
    n_rows: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def collect_nc_paths(root: str | Path, pattern: str = "**/*.nc") -> list[Path]:
    """Collect Argo NetCDF files recursively under *root*.

    If *root* is itself a file, it is returned directly.
    """
    root_path = Path(root)
    if root_path.is_file():
        return [root_path]
    return sorted(path for path in root_path.glob(pattern) if path.is_file())


def argo_directory_to_dataframe(
    root: str | Path,
    *,
    pattern: str = "**/*.nc",
    good_qc_only: bool = True,
    min_levels: int = 5,
):
    """Flatten all valid Argo profiles under *root* into a pandas dataframe.

    The resulting table uses one row per observed level and includes file,
    profile, and level metadata so profiles with different sampling depths stay
    separate.
    """
    try:
        import pandas as pd
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Parquet export requires pandas. Install with: pip install -e '.[io]'"
        ) from exc

    rows: list[dict[str, Any]] = []
    n_files = 0
    n_profiles = 0

    for nc_path in collect_nc_paths(root, pattern=pattern):
        n_files += 1
        try:
            profiles = read_argo_file(nc_path, good_qc_only=good_qc_only, min_levels=min_levels)
        except Exception as exc:
            warnings.warn(f"Skipping {nc_path}: {exc}", stacklevel=2)
            continue

        for profile_index, profile in enumerate(profiles):
            n_profiles += 1
            rows.extend(_profile_to_rows(profile, nc_path, profile_index))

    df = pd.DataFrame.from_records(rows)
    if not df.empty:
        df = df.sort_values(["source_file", "profile_index_in_file", "level_index"]).reset_index(drop=True)
        for col in ("source_file", "profile_id", "float_wmo"):
            if col in df.columns:
                df[col] = df[col].astype("string")
        for col in ("profile_index_in_file", "level_index", "n_levels", "cycle_number"):
            if col in df.columns:
                df[col] = df[col].astype("Int64")
    df.attrs["n_files"] = n_files
    df.attrs["n_profiles"] = n_profiles
    df.attrs["n_rows"] = int(df.shape[0])
    df.attrs["source_root"] = str(Path(root))
    df.attrs["pattern"] = pattern
    return df


def write_argo_parquet(
    root: str | Path,
    output: str | Path,
    *,
    pattern: str = "**/*.nc",
    good_qc_only: bool = True,
    min_levels: int = 5,
    engine: str = "pyarrow",
) -> ArgoParquetSummary:
    """Write a recursive Argo profile export to parquet."""
    output_path = Path(output)
    if output_path.exists() and output_path.is_dir():
        raise IsADirectoryError(f"Output must be a parquet file, not a directory: {output_path}")

    try:
        import pandas  # noqa: F401
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Parquet export requires pandas. Install with: pip install -e '.[io]'"
        ) from exc

    df = argo_directory_to_dataframe(
        root,
        pattern=pattern,
        good_qc_only=good_qc_only,
        min_levels=min_levels,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        df.to_parquet(output_path, index=False, engine=engine)
    except Exception as exc:
        raise ImportError(
            "Writing parquet requires a parquet engine such as pyarrow. "
            "Install with: pip install -e '.[io]'"
        ) from exc

    return ArgoParquetSummary(
        output=str(output_path),
        n_files=int(df.attrs.get("n_files", 0)),
        n_profiles=int(df.attrs.get("n_profiles", 0)),
        n_rows=int(df.attrs.get("n_rows", 0)),
    )


def iter_argo_parquet_profiles(
    path: str | Path,
    *,
    min_levels: int = 5,
) -> list[ArgoProfile]:
    """Load Argo profiles back from a parquet export.

    The parquet file is expected to contain one row per observed level. The
    exporter in this package writes the following columns:

    - ``source_file``
    - ``profile_index_in_file``
    - ``profile_id``
    - ``float_wmo``
    - ``cycle_number``
    - ``juld``
    - ``n_levels``
    - ``level_index``
    - ``pressure``
    - ``temperature``
    - ``salinity``

    Rows are regrouped into profiles using the best available identifier. If a
    file was written by :func:`write_argo_parquet`, grouping by
    ``source_file`` + ``profile_index_in_file`` is enough to reconstruct each
    profile exactly as exported.
    """
    try:
        import pandas as pd
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Reading parquet profiles requires pandas. Install with: pip install -e '.[io]'"
        ) from exc

    df = pd.read_parquet(path)
    required = {"pressure", "temperature", "salinity"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Parquet file is missing required columns: {sorted(missing)}")

    group_cols = _parquet_group_columns(df.columns)
    if not group_cols:
        raise ValueError(
            "Parquet file does not contain enough grouping columns to reconstruct profiles."
        )

    sort_cols = [col for col in ("level_index", "pressure") if col in df.columns]
    profiles: list[ArgoProfile] = []
    for group_key, group in df.groupby(group_cols, sort=False, dropna=False):
        ordered = group.sort_values(sort_cols) if sort_cols else group
        pressure = pd.to_numeric(ordered["pressure"], errors="coerce").to_numpy(dtype=float)
        temperature = pd.to_numeric(ordered["temperature"], errors="coerce").to_numpy(dtype=float)
        salinity = pd.to_numeric(ordered["salinity"], errors="coerce").to_numpy(dtype=float)

        valid = np.isfinite(pressure) & np.isfinite(temperature) & np.isfinite(salinity)
        pressure = pressure[valid]
        temperature = temperature[valid]
        salinity = salinity[valid]
        if pressure.size < min_levels:
            continue

        order = np.argsort(pressure)
        pressure = pressure[order]
        temperature = temperature[order]
        salinity = salinity[order]

        profile_id = _parquet_first_value(ordered, "profile_id")
        if profile_id is None:
            profile_id = _parquet_group_label(group_key, group_cols)
        float_wmo = _parquet_first_value(ordered, "float_wmo")
        cycle_number = _parquet_optional_int(ordered, "cycle_number")
        juld = _parquet_optional_float(ordered, "juld")

        profiles.append(
            ArgoProfile(
                profile_id=str(profile_id),
                pressure=pressure,
                temperature=temperature,
                salinity=salinity,
                cycle_number=cycle_number,
                float_wmo=float_wmo,
                juld=juld,
            )
        )

    return profiles


def _profile_to_rows(profile: ArgoProfile, source_file: Path, profile_index: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n_levels = profile.n_levels
    source_file_str = str(source_file)
    for level_index in range(n_levels):
        rows.append(
            {
                "source_file": source_file_str,
                "profile_index_in_file": profile_index,
                "profile_id": profile.profile_id,
                "float_wmo": profile.float_wmo,
                "cycle_number": profile.cycle_number,
                "juld": profile.juld,
                "n_levels": n_levels,
                "level_index": level_index,
                "pressure": float(profile.pressure[level_index]),
                "temperature": float(profile.temperature[level_index]),
                "salinity": float(profile.salinity[level_index]),
            }
        )
    return rows


def _parquet_group_columns(columns: Any) -> list[str]:
    preferred = [
        ("source_file", "profile_index_in_file"),
        ("source_file", "profile_id"),
        ("profile_id",),
        ("profile_index_in_file",),
        ("source_file",),
    ]
    available = set(columns)
    for candidate in preferred:
        if all(col in available for col in candidate):
            return list(candidate)
    return []


def _parquet_group_label(group_key: Any, group_cols: list[str]) -> str:
    if isinstance(group_key, tuple):
        parts = [f"{col}={value}" for col, value in zip(group_cols, group_key, strict=False)]
    else:
        parts = [f"{group_cols[0]}={group_key}"] if group_cols else [str(group_key)]
    return "parquet_" + "_".join(str(part) for part in parts)


def _parquet_first_value(df: Any, column: str) -> str | None:
    if column not in df.columns:
        return None
    series = df[column].dropna()
    if series.empty:
        return None
    value = series.iloc[0]
    return None if value is None else str(value)


def _parquet_optional_int(df: Any, column: str) -> int | None:
    if column not in df.columns:
        return None
    series = df[column].dropna()
    if series.empty:
        return None
    try:
        return int(series.iloc[0])
    except Exception:
        return None


def _parquet_optional_float(df: Any, column: str) -> float | None:
    if column not in df.columns:
        return None
    series = df[column].dropna()
    if series.empty:
        return None
    try:
        return float(series.iloc[0])
    except Exception:
        return None
