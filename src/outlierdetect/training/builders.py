"""Shared synthetic-example builders for profile training."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np

from outlierdetect.argo import ArgoProfile, sample_pressure_indices
from outlierdetect.training.synthetic import SyntheticExample, degrade_highres_profile


def iter_synthetic_examples_from_profiles(
    profiles: Iterable[ArgoProfile],
    *,
    source_name: str,
    source_profile_id_attr: str,
    n_examples_per_profile: int = 1,
    n_levels: int = 20,
    grid_size: int = 80,
    profile_limit: int | None = None,
    min_levels: int = 5,
    seed: int | None = None,
    upper_ocean_bias: float = 1.7,
    reference_source: bool | str | Path | None = None,
) -> Iterator[SyntheticExample]:
    """Convert clean profiles into synthetic training examples.

    When ``profile_limit`` is set, the builder first filters out profiles that
    do not meet ``min_levels`` and then draws a random subset of the remaining
    eligible profiles. The cap is therefore applied after QC/min-level
    filtering rather than to the raw input order.
    """

    rng = np.random.default_rng(seed)
    min_levels_required = max(min_levels, 5)

    if profile_limit is None:
        selected_profiles: Iterable[ArgoProfile] = profiles
    else:
        eligible_profiles = [profile for profile in profiles if profile.n_levels >= min_levels_required]
        if not eligible_profiles:
            return []
        limit = max(int(profile_limit), 0)
        if limit == 0:
            return []
        if len(eligible_profiles) > limit:
            selected_indices = np.sort(rng.choice(len(eligible_profiles), size=limit, replace=False))
            selected_profiles = [eligible_profiles[int(index)] for index in selected_indices]
        else:
            selected_profiles = eligible_profiles

    for profile in selected_profiles:
        if profile.n_levels < min_levels_required:
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
                    source_profile_id_attr: profile.profile_id,
                    "float_wmo": profile.float_wmo,
                    "cycle_number": profile.cycle_number,
                    "juld": profile.juld,
                    "latitude": profile.latitude,
                    "longitude": profile.longitude,
                }
            )
            yield synth


def build_synthetic_examples_from_profiles(
    profiles: Iterable[ArgoProfile],
    **kwargs: object,
) -> list[SyntheticExample]:
    """Materialize synthetic examples from clean profiles."""

    return list(iter_synthetic_examples_from_profiles(profiles, **kwargs))
