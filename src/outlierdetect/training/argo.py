"""Argo-backed training example builders.

This module bridges clean Argo profiles to the synthetic corruption pipeline
used for profile-model training. Each source profile keeps its own native
pressure coordinates; subsampling is applied per profile before corruption and
no global resampling step forces all profiles onto the same observed depth grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

from outlierdetect.argo import ArgoProfile, iter_argo_files, sample_pressure_indices
from outlierdetect.parquet import iter_argo_parquet_profiles
from outlierdetect.training.dataset import ProfileDataset, ProfileExample
from outlierdetect.training.synthetic import SyntheticExample, degrade_highres_profile


@dataclass(slots=True)
class ArgoTrainingConfig:
    """Configuration for turning clean Argo profiles into training examples."""

    root: str | Path
    n_examples_per_profile: int = 1
    n_levels: int = 20
    grid_size: int = 80
    profile_limit: int | None = None
    min_levels: int = 5
    good_qc_only: bool = True
    seed: int | None = None
    upper_ocean_bias: float = 1.7


def build_argo_synthetic_examples(
    root: str | Path | Sequence[ArgoProfile],
    *,
    n_examples_per_profile: int = 1,
    n_levels: int = 20,
    grid_size: int = 80,
    profile_limit: int | None = None,
    min_levels: int = 5,
    good_qc_only: bool = True,
    seed: int | None = None,
    upper_ocean_bias: float = 1.7,
) -> list[SyntheticExample]:
    """Convert Argo profiles into synthetic training examples.

    Parameters
    ----------
    root:
        Either a directory/file path containing Argo NetCDF files or an in-memory
        sequence of :class:`~outlierdetect.argo.ArgoProfile` objects.
    n_examples_per_profile:
        Number of synthetic degradations to generate from each clean profile.
    n_levels:
        Target sparse level count before synthetic corruption.
    grid_size:
        Number of pressure grid points used for reconstruction targets.
    profile_limit:
        Optional cap on the number of source profiles to use.
    min_levels:
        Minimum number of finite source levels required from the clean profile.
    good_qc_only:
        Forwarded to the Argo reader when ``root`` is a path.
    seed:
        Seed for deterministic synthetic corruption.
    upper_ocean_bias:
        Bias exponent for subsampling pressure levels toward the shallow ocean.

    Notes
    -----
    The returned synthetic profiles retain profile-specific pressure sampling.
    If two Argo source profiles have different native depths, the subsampled
    observations remain different as well; there is no intermediate uniform
    pressure grid used to align the observed values across profiles.
    """

    rng = np.random.default_rng(seed)
    examples: list[SyntheticExample] = []

    profiles: Iterable[ArgoProfile]
    if isinstance(root, Sequence) and not isinstance(root, (str, bytes, Path)):
        if len(root) == 0:
            return []
        if isinstance(root[0], ArgoProfile):
            profiles = root  # type: ignore[assignment]
        else:
            raise TypeError("root sequences must contain ArgoProfile objects.")
    else:
        root_path = Path(root)
        if root_path.suffix.lower() in {".parquet", ".pq"}:
            profiles = iter_argo_parquet_profiles(root_path, min_levels=min_levels)
        else:
            profiles = iter_argo_files(
                root,
                good_qc_only=good_qc_only,
                min_levels=min_levels,
            )

    for profile_index, profile in enumerate(profiles):
        if profile_limit is not None and profile_index >= profile_limit:
            break

        if profile.n_levels < max(min_levels, 5):
            continue

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
                    day_of_year=day_of_year,
                    profile_id=profile.profile_id,
                )
            except ValueError:
                # Some source profiles still become unusable after cleaning. Skip them
                # rather than aborting the whole training run.
                continue
            synth.example.profile.attrs.update(
                {
                    "source": "argo",
                    "argo_profile_id": profile.profile_id,
                    "float_wmo": profile.float_wmo,
                    "cycle_number": profile.cycle_number,
                    "juld": profile.juld,
                }
            )
            examples.append(synth)

    return examples


def build_argo_examples(
    root: str | Path | Sequence[ArgoProfile],
    **kwargs: object,
) -> list[ProfileExample]:
    """Return only the labeled ``ProfileExample`` objects for training."""
    return [item.example for item in build_argo_synthetic_examples(root, **kwargs)]


def build_argo_dataset(
    root: str | Path | Sequence[ArgoProfile],
    norm: dict[str, float] | None = None,
    **kwargs: object,
) -> ProfileDataset:
    """Create a :class:`ProfileDataset` from Argo-backed synthetic examples."""
    return ProfileDataset(build_argo_examples(root, **kwargs), norm=norm)
