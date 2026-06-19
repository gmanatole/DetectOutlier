"""EN4-backed training example builders.

This module mirrors :mod:`outlierdetect.training.argo`, but uses the EN4
monthly NetCDF reader as its clean-profile source.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

from outlierdetect.argo import ArgoProfile
from outlierdetect.en4 import iter_en4_files
from outlierdetect.training.builders import build_synthetic_examples_from_profiles
from outlierdetect.training.dataset import ProfileDataset, ProfileExample
from outlierdetect.training.synthetic import SyntheticExample


def build_en4_synthetic_examples(
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
    use_raw_values: bool = False,
) -> list[SyntheticExample]:
    """Convert EN4 profiles into synthetic training examples."""

    profiles: Sequence[ArgoProfile] | Iterable[ArgoProfile]
    if isinstance(root, Sequence) and not isinstance(root, (str, bytes, Path)):
        if len(root) == 0:
            return []
        if isinstance(root[0], ArgoProfile):
            profiles = root
        else:
            raise TypeError("root sequences must contain ArgoProfile objects.")
    else:
        effective_good_qc_only = good_qc_only and not use_raw_values
        profiles = iter_en4_files(
            root,
            good_qc_only=effective_good_qc_only,
            min_levels=min_levels,
            use_raw_values=use_raw_values,
        )

    return build_synthetic_examples_from_profiles(
        profiles,
        source_name="en4",
        source_profile_id_attr="en4_profile_id",
        n_examples_per_profile=n_examples_per_profile,
        n_levels=n_levels,
        grid_size=grid_size,
        profile_limit=profile_limit,
        min_levels=min_levels,
        seed=seed,
        upper_ocean_bias=upper_ocean_bias,
    )


def build_en4_examples(
    root: str | Path | Sequence[ArgoProfile],
    **kwargs: object,
) -> list[ProfileExample]:
    """Return only the labeled ``ProfileExample`` objects for training."""
    return [item.example for item in build_en4_synthetic_examples(root, **kwargs)]


def build_en4_dataset(
    root: str | Path | Sequence[ArgoProfile],
    norm: dict[str, float] | None = None,
    **kwargs: object,
) -> ProfileDataset:
    """Create a :class:`ProfileDataset` from EN4-backed synthetic examples."""
    return ProfileDataset(build_en4_examples(root, **kwargs), norm=norm)
