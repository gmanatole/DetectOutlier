"""Shared synthetic-example builders for profile training."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from outlierdetect.argo import ArgoProfile, sample_pressure_indices
from outlierdetect.training.synthetic import SyntheticExample, degrade_highres_profile


def build_synthetic_examples_from_profiles(
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
) -> list[SyntheticExample]:
    """Convert clean profiles into synthetic training examples."""

    rng = np.random.default_rng(seed)
    examples: list[SyntheticExample] = []

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
                    latitude=profile.latitude,
                    longitude=profile.longitude,
                    day_of_year=day_of_year,
                    profile_id=profile.profile_id,
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
            examples.append(synth)

    return examples
