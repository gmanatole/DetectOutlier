"""Argo-backed training example builders.

This module bridges clean Argo profiles to the synthetic corruption pipeline
used for profile-model training. Each source profile keeps its own native
pressure coordinates; subsampling is applied per profile before corruption and
no global resampling step forces all profiles onto the same observed depth grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

from outlierdetect.argo import ArgoProfile, iter_argo_files
from outlierdetect.training.builders import iter_synthetic_examples_from_profiles
from outlierdetect.training.dataset import ProfileDataset, ProfileExample
from outlierdetect.training.synthetic import SyntheticExample


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


def iter_argo_synthetic_examples(
    root: str | Path | Sequence[ArgoProfile],
    *,
    n_examples_per_profile: int = 1,
    n_levels: int = 20,
    grid_size: int = 80,
    profile_limit: int | None = None,
    min_levels: int = 5,
    good_qc_only: bool = True,
    profile_type: str = "adjusted",
    raw_fallback: bool = False,
    seed: int | None = None,
    upper_ocean_bias: float = 1.7,
    use_raw_values: bool = False,
    reference_source: bool | str | Path | None = None,
) -> Iterator[SyntheticExample]:
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
        Optional cap on the number of source profiles to sample after
        ``min_levels`` filtering.
    min_levels:
        Minimum number of finite source levels required from the clean profile.
    good_qc_only:
        Forwarded to the Argo reader when ``root`` is a path.
    profile_type:
        Select adjusted values when available, or raw values only.
    raw_fallback:
        When ``profile_type`` is adjusted, allow raw values if the adjusted
        field is missing.
    use_raw_values:
        If True, read raw ``TEMP``/``PSAL`` values and skip QC masking.
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

    profiles: Iterable[ArgoProfile]
    if isinstance(root, Sequence) and not isinstance(root, (str, bytes, Path)):
        if len(root) == 0:
            return []
        if isinstance(root[0], ArgoProfile):
            profiles = root  # type: ignore[assignment]
        else:
            raise TypeError("root sequences must contain ArgoProfile objects.")
    else:
        effective_good_qc_only = good_qc_only and not use_raw_values
        root_path = Path(root)
        profiles = iter_argo_files(
            root,
            good_qc_only=effective_good_qc_only,
            min_levels=min_levels,
            profile_type=profile_type,
            raw_fallback=raw_fallback,
            use_raw_values=use_raw_values,
        )
    return iter_synthetic_examples_from_profiles(
        profiles,
        source_name="argo",
        source_profile_id_attr="argo_profile_id",
        n_examples_per_profile=n_examples_per_profile,
        n_levels=n_levels,
        grid_size=grid_size,
        profile_limit=profile_limit,
        min_levels=min_levels,
        seed=seed,
        upper_ocean_bias=upper_ocean_bias,
        reference_source=reference_source,
    )


def build_argo_synthetic_examples(
    root: str | Path | Sequence[ArgoProfile],
    **kwargs: object,
) -> list[SyntheticExample]:
    """Convert Argo profiles into synthetic training examples."""

    return list(iter_argo_synthetic_examples(root, **kwargs))


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
