"""Streaming dataset helpers for lazy profile training.

The preload training path still materializes examples in memory. The helpers in
this module build the alternative path where the loader opens source NetCDF
files on demand and yields collated profile examples lazily.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np

try:
    from torch.utils.data import IterableDataset, get_worker_info
except Exception:  # pragma: no cover - torch is optional
    IterableDataset = object  # type: ignore[assignment,misc]
    get_worker_info = None  # type: ignore[assignment]

from outlierdetect.argo import ArgoProfile, iter_argo_files, read_argo_file, sample_pressure_indices
from outlierdetect.en4 import iter_en4_files, read_en4_file
from outlierdetect.runtime_config import resolve_profile_input

from .dataset import ProfileExample, profile_example_to_sample
from .synthetic import degrade_highres_profile


def make_training_example_factory(
    root: str | Path,
    *,
    data_source: str,
    augment: bool,
    profile_type: str,
    raw_fallback: bool,
    good_qc_only: bool,
    min_levels: int,
    profile_limit: int | None,
    n_examples_per_profile: int,
    n_levels: int,
    grid_size: int,
    seed: int | None,
    upper_ocean_bias: float,
    use_raw_values: bool,
    reference_source: bool | str | Path | None,
    use_residual_t: bool,
    use_residual_s: bool,
    use_sigma_t: bool,
    use_sigma_s: bool,
    use_sigma_vert: bool,
    use_sigma_heave_t: bool,
    use_sigma_heave_s: bool,
    use_rho_ts: bool,
    use_day_of_year: bool,
    sigma_heave_source: bool | str | Path | None,
    source_name: str | None = None,
    source_files: Sequence[str | Path] | None = None,
) -> Callable[[], Iterator[ProfileExample]]:
    """Return a fresh iterator factory for streaming profile examples."""

    kwargs = {
        "root": root,
        "data_source": data_source,
        "augment": augment,
        "source_files": source_files,
        "profile_type": profile_type,
        "raw_fallback": raw_fallback,
        "good_qc_only": good_qc_only,
        "min_levels": min_levels,
        "profile_limit": profile_limit,
        "n_examples_per_profile": n_examples_per_profile,
        "n_levels": n_levels,
        "grid_size": grid_size,
        "seed": seed,
        "upper_ocean_bias": upper_ocean_bias,
        "use_raw_values": use_raw_values,
        "reference_source": reference_source,
        "use_residual_t": use_residual_t,
        "use_residual_s": use_residual_s,
        "use_sigma_t": use_sigma_t,
        "use_sigma_s": use_sigma_s,
        "use_sigma_vert": use_sigma_vert,
        "use_sigma_heave_t": use_sigma_heave_t,
        "use_sigma_heave_s": use_sigma_heave_s,
        "use_rho_ts": use_rho_ts,
        "use_day_of_year": use_day_of_year,
        "sigma_heave_source": sigma_heave_source,
        "source_name": source_name or data_source,
    }
    return StreamingExampleFactory(kwargs)


@dataclass(slots=True)
class StreamingExampleFactory:
    """Picklable callable that builds streaming examples on demand."""

    kwargs: dict[str, Any]

    def __call__(self) -> Iterator[ProfileExample]:
        return _iter_streaming_examples(**self.kwargs)


@dataclass(slots=True)
class StreamingProfileDataset(IterableDataset):
    """IterableDataset that builds samples lazily from NetCDF sources."""

    example_factory: Callable[[], Iterator[ProfileExample]]
    norm: Any = None

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for example in self.example_factory():
            yield profile_example_to_sample(example, norm=self.norm)

    def set_normalization(self, norm: Any) -> None:
        self.norm = norm


def preview_stream_examples(
    example_factory: Callable[[], Iterator[ProfileExample]],
    *,
    limit: int,
) -> list[ProfileExample]:
    """Materialize a small preview slice for model shape and plotting."""

    return list(islice(example_factory(), max(int(limit), 0)))


def list_source_files(root: str | Path, pattern: str = "**/*.nc") -> list[Path]:
    """Return the matching NetCDF files under *root* without opening them."""

    root_path = Path(root)
    if root_path.is_file():
        return [root_path]
    return sorted(root_path.glob(pattern))


def split_source_files(
    files: Sequence[str | Path],
    val_fraction: float,
    seed: int,
) -> tuple[list[Path], list[Path]]:
    """Split a file list into train and validation subsets."""

    paths = [Path(file).expanduser().resolve() for file in files]
    if not paths:
        return [], []
    rng = np.random.default_rng(int(seed))
    indices = np.arange(len(paths))
    rng.shuffle(indices)
    n_val = int(round(len(indices) * float(val_fraction)))
    n_val = min(max(n_val, 0), max(len(indices) - 1, 0))
    train_idx = indices[n_val:].tolist()
    val_idx = indices[:n_val].tolist()
    train = [paths[i] for i in train_idx]
    val = [paths[i] for i in val_idx]
    return train, val


def _iter_streaming_examples(
    *,
    root: str | Path,
    data_source: str,
    augment: bool,
    profile_type: str,
    raw_fallback: bool,
    good_qc_only: bool,
    min_levels: int,
    profile_limit: int | None,
    n_examples_per_profile: int,
    n_levels: int,
    grid_size: int,
    seed: int | None,
    upper_ocean_bias: float,
    use_raw_values: bool,
    reference_source: bool | str | Path | None,
    use_residual_t: bool,
    use_residual_s: bool,
    use_sigma_t: bool,
    use_sigma_s: bool,
    use_sigma_vert: bool,
    use_sigma_heave_t: bool,
    use_sigma_heave_s: bool,
    use_rho_ts: bool,
    use_day_of_year: bool,
    sigma_heave_source: bool | str | Path | None,
    source_name: str,
    source_files: Sequence[str | Path] | None = None,
) -> Iterator[ProfileExample]:
    if data_source not in {"argo", "en4"}:
        raise ValueError(f"Unsupported data_source: {data_source!r}")

    profile_iter = _iter_source_profiles(
        root,
        source_files=source_files,
        data_source=data_source,
        good_qc_only=good_qc_only,
        min_levels=min_levels,
        profile_type=profile_type,
        raw_fallback=raw_fallback,
        use_raw_values=use_raw_values,
    )

    if augment:
        yield from _iter_streaming_synthetic_examples(
            profile_iter,
            profile_limit=profile_limit,
            min_levels=min_levels,
            n_examples_per_profile=n_examples_per_profile,
            n_levels=n_levels,
            grid_size=grid_size,
            seed=seed,
            upper_ocean_bias=upper_ocean_bias,
            reference_source=reference_source,
            source_name=source_name,
        )
        return

    yield from _iter_streaming_profile_examples(
        profile_iter,
        profile_limit=profile_limit,
        use_residual_t=use_residual_t,
        use_residual_s=use_residual_s,
        use_sigma_t=use_sigma_t,
        use_sigma_s=use_sigma_s,
        use_sigma_vert=use_sigma_vert,
        use_sigma_heave_t=use_sigma_heave_t,
        use_sigma_heave_s=use_sigma_heave_s,
        use_rho_ts=use_rho_ts,
        use_day_of_year=use_day_of_year,
        sigma_heave_source=sigma_heave_source,
        reference_source=reference_source,
    )


def _iter_source_profiles(
    root: str | Path,
    *,
    source_files: Sequence[str | Path] | None,
    data_source: str,
    good_qc_only: bool,
    min_levels: int,
    profile_type: str,
    raw_fallback: bool,
    use_raw_values: bool,
) -> Iterator[ArgoProfile]:
    effective_good_qc_only = good_qc_only and not use_raw_values
    if source_files is not None:
        source_files = _shard_source_files(source_files)
        yield from _iter_profiles_from_files(
            source_files,
            data_source=data_source,
            good_qc_only=effective_good_qc_only,
            min_levels=min_levels,
            profile_type=profile_type,
            raw_fallback=raw_fallback,
            use_raw_values=use_raw_values,
        )
        return
    if data_source == "argo":
        yield from iter_argo_files(
            root,
            good_qc_only=effective_good_qc_only,
            min_levels=min_levels,
            profile_type=profile_type,
            raw_fallback=raw_fallback,
            use_raw_values=use_raw_values,
        )
        return
    yield from iter_en4_files(
        root,
        good_qc_only=effective_good_qc_only,
        min_levels=min_levels,
        profile_type=profile_type,
        raw_fallback=raw_fallback,
        use_raw_values=use_raw_values,
    )


def _shard_source_files(source_files: Sequence[str | Path]) -> list[Path]:
    paths = [Path(path).expanduser().resolve() for path in source_files]
    if not paths or get_worker_info is None:
        return paths
    worker = get_worker_info()
    if worker is None or worker.num_workers <= 1:
        return paths
    return [path for index, path in enumerate(paths) if index % worker.num_workers == worker.id]


def _iter_profiles_from_files(
    source_files: Sequence[str | Path],
    *,
    data_source: str,
    good_qc_only: bool,
    min_levels: int,
    profile_type: str,
    raw_fallback: bool,
    use_raw_values: bool,
) -> Iterator[ArgoProfile]:
    for nc_path in source_files:
        path = Path(nc_path)
        try:
            if data_source == "argo":
                profiles = read_argo_file(
                    path,
                    good_qc_only=good_qc_only,
                    min_levels=min_levels,
                    profile_type=profile_type,
                    raw_fallback=raw_fallback,
                    use_raw_values=use_raw_values,
                )
            else:
                profiles = read_en4_file(
                    path,
                    good_qc_only=good_qc_only,
                    min_levels=min_levels,
                    profile_type=profile_type,
                    raw_fallback=raw_fallback,
                    use_raw_values=use_raw_values,
                )
            for profile in profiles:
                yield profile
        except Exception as exc:
            import warnings

            warnings.warn(f"Skipping {path}: {exc}", stacklevel=2)


def _iter_streaming_profile_examples(
    profiles: Iterator[ArgoProfile],
    *,
    profile_limit: int | None,
    use_residual_t: bool,
    use_residual_s: bool,
    use_sigma_t: bool,
    use_sigma_s: bool,
    use_sigma_vert: bool,
    use_sigma_heave_t: bool,
    use_sigma_heave_s: bool,
    use_rho_ts: bool,
    use_day_of_year: bool,
    sigma_heave_source: bool | str | Path | None,
    reference_source: bool | str | Path | None,
) -> Iterator[ProfileExample]:
    max_profiles = None if profile_limit is None else max(int(profile_limit), 0)
    selected = 0
    for profile in profiles:
        if max_profiles is not None and selected >= max_profiles:
            break
        selected += 1
        profile_input = resolve_profile_input(
            profile.to_profile_input(),
            use_residual_t=use_residual_t,
            use_residual_s=use_residual_s,
            use_sigma_t=use_sigma_t,
            use_sigma_s=use_sigma_s,
            use_sigma_vert=use_sigma_vert,
            use_sigma_heave_t=use_sigma_heave_t,
            use_sigma_heave_s=use_sigma_heave_s,
            use_rho_ts=use_rho_ts,
            use_day_of_year=use_day_of_year,
            sigma_heave_source=sigma_heave_source,
            reference_source=reference_source,
        )
        yield ProfileExample(profile=profile_input)


def _iter_streaming_synthetic_examples(
    profiles: Iterator[ArgoProfile],
    *,
    profile_limit: int | None,
    min_levels: int,
    n_examples_per_profile: int,
    n_levels: int,
    grid_size: int,
    seed: int | None,
    upper_ocean_bias: float,
    reference_source: bool | str | Path | None,
    source_name: str,
) -> Iterator[ProfileExample]:
    rng = np.random.default_rng(seed)
    max_profiles = None if profile_limit is None else max(int(profile_limit), 0)
    selected = 0
    min_levels_required = max(int(min_levels), 5)

    for profile in profiles:
        if profile.n_levels < min_levels_required:
            continue
        if max_profiles is not None and selected >= max_profiles:
            break
        selected += 1

        day_of_year = None
        if profile.juld is not None and np.isfinite(profile.juld):
            day_of_year = float(np.mod(profile.juld, 365.2425))

        for _ in range(max(int(n_examples_per_profile), 1)):
            child_seed = int(rng.integers(0, 2**32 - 1))
            child_rng = np.random.default_rng(child_seed)
            indices = sample_pressure_indices(
                profile.pressure,
                n_levels=n_levels,
                rng=child_rng,
                upper_ocean_bias=upper_ocean_bias,
            )
            pressure_levels = profile.pressure[indices]
            try:
                synth = degrade_highres_profile(
                    profile.pressure,
                    profile.temperature,
                    profile.salinity,
                    rng=child_rng,
                    pressure_levels=pressure_levels,
                    grid_size=grid_size,
                    latitude=profile.latitude,
                    longitude=profile.longitude,
                    day_of_year=day_of_year,
                    profile_id=profile.profile_id,
                    reference_source=reference_source,
                )
            except ValueError:
                continue
            synth.example.profile.attrs.update(
                {
                    "source": source_name,
                    "source_profile_id": profile.profile_id,
                    "float_wmo": profile.float_wmo,
                    "cycle_number": profile.cycle_number,
                    "juld": profile.juld,
                    "latitude": profile.latitude,
                    "longitude": profile.longitude,
                }
            )
            yield synth.example
