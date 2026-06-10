"""PyTorch dataset and padding utilities for profile-model training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from outlierdetect.data import NormalizationStats, ProfileInput
from outlierdetect.features import build_level_features

try:
    import torch
    from torch.utils.data import Dataset as TorchDataset
except Exception:  # pragma: no cover - torch is optional
    torch = None  # type: ignore[assignment]
    TorchDataset = object  # type: ignore[assignment,misc]


@dataclass(slots=True)
class ProfileLabels:
    """Training labels for one profile."""

    profile_bad: float | None = None
    point_bad_t: np.ndarray | None = None
    point_bad_s: np.ndarray | None = None
    point_density_inconsistent: np.ndarray | None = None
    nuisance_mean: np.ndarray | None = None  # [a_t, b_t, a_s, b_s]
    pressure_grid: np.ndarray | None = None
    truth_t: np.ndarray | None = None
    truth_s: np.ndarray | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ProfileExample:
    profile: ProfileInput
    labels: ProfileLabels = field(default_factory=ProfileLabels)


class ProfileDataset(TorchDataset):
    """Dataset of ProfileExample objects.

    The dataset keeps profiles as dataclasses and builds features lazily. This
    makes it easy to swap feature versions during prototyping.
    """

    def __init__(
        self,
        examples: Sequence[ProfileExample],
        norm: NormalizationStats | dict[str, float] | None = None,
    ):
        if torch is None:  # pragma: no cover
            raise ImportError("ProfileDataset requires PyTorch. Install the train extra.")
        self.examples = list(examples)
        self.norm = NormalizationStats.from_mapping(norm)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        ex = self.examples[index]
        feats = build_level_features(ex.profile, normalization=self.norm)
        labels = ex.labels
        point_labels = _stack_point_labels(labels, ex.profile.n_levels)
        recon_truth = _stack_reconstruction(labels)
        recon_truth_physical = None if recon_truth is None else np.asarray(recon_truth, dtype=float)
        if recon_truth is not None and self.norm is not None:
            recon_truth = np.column_stack(
                [
                    self.norm.normalize_temperature(recon_truth[:, 0]),
                    self.norm.normalize_salinity(recon_truth[:, 1]),
                ]
            ).astype("float32")
        sample = {
            "features": feats.level_features.astype("float32"),
            "mask": feats.mask.astype(bool),
            "feature_names": feats.feature_names,
            "profile_bad": _optional_scalar(labels.profile_bad),
            "point_labels": point_labels.astype("float32"),
            "point_label_mask": np.isfinite(point_labels).all(axis=-1),
            "nuisance_mean": _optional_array(labels.nuisance_mean, shape=(4,)),
            "recon_truth": recon_truth,
            "recon_truth_physical": recon_truth_physical,
            "pressure_grid": labels.pressure_grid,
            "profile_id": ex.profile.profile_id,
            "norm_stats": None if self.norm is None else self.norm.as_dict(),
        }
        return sample


def collate_profiles(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad a list of variable-length profile samples into tensors."""
    if torch is None:  # pragma: no cover
        raise ImportError("collate requires PyTorch. Install the train extra.")

    max_n = max(item["features"].shape[0] for item in batch)
    input_dim = batch[0]["features"].shape[1]
    bsz = len(batch)
    features = np.zeros((bsz, max_n, input_dim), dtype="float32")
    mask = np.zeros((bsz, max_n), dtype=bool)
    point_labels = np.full((bsz, max_n, 3), np.nan, dtype="float32")
    point_label_mask = np.zeros((bsz, max_n), dtype=bool)

    for b, item in enumerate(batch):
        n = item["features"].shape[0]
        features[b, :n] = item["features"]
        mask[b, :n] = item["mask"]
        point_labels[b, :n] = item["point_labels"]
        point_label_mask[b, :n] = item["point_label_mask"]

    profile_bad = np.array([item["profile_bad"] for item in batch], dtype="float32")
    nuisance = np.stack([item["nuisance_mean"] for item in batch]).astype("float32")
    recon_truth = _collate_optional_recon([item["recon_truth"] for item in batch])
    recon_truth_physical = _collate_optional_recon([item["recon_truth_physical"] for item in batch])
    pressure_grid = _collate_optional_pressure_grid([item["pressure_grid"] for item in batch])

    return {
        "features": torch.from_numpy(features),
        "mask": torch.from_numpy(mask),
        "point_labels": torch.from_numpy(point_labels),
        "point_label_mask": torch.from_numpy(point_label_mask),
        "profile_bad": torch.from_numpy(profile_bad),
        "profile_bad_mask": torch.isfinite(torch.from_numpy(profile_bad)),
        "nuisance_mean": torch.from_numpy(nuisance),
        "nuisance_mask": torch.isfinite(torch.from_numpy(nuisance)).all(dim=-1),
        "recon_truth": None if recon_truth is None else torch.from_numpy(recon_truth),
        "recon_truth_physical": None if recon_truth_physical is None else torch.from_numpy(recon_truth_physical),
        "pressure_grid": None if pressure_grid is None else torch.from_numpy(pressure_grid),
        "profile_id": [item["profile_id"] for item in batch],
        "feature_names": batch[0]["feature_names"],
    }


# Compatibility aliases.
Dataset = ProfileDataset
Example = ProfileExample
Labels = ProfileLabels
collate = collate_profiles


def _optional_scalar(value: float | None) -> float:
    return float("nan") if value is None else float(value)


def _optional_array(value: np.ndarray | None, shape: tuple[int, ...]) -> np.ndarray:
    if value is None:
        return np.full(shape, np.nan, dtype=float)
    arr = np.asarray(value, dtype=float)
    if arr.shape != shape:
        raise ValueError(f"Expected shape {shape}, got {arr.shape}.")
    return arr


def _stack_point_labels(labels: ProfileLabels, n: int) -> np.ndarray:
    arrays = []
    for arr in (labels.point_bad_t, labels.point_bad_s, labels.point_density_inconsistent):
        if arr is None:
            arrays.append(np.full(n, np.nan, dtype=float))
        else:
            value = np.asarray(arr, dtype=float)
            if value.shape != (n,):
                raise ValueError(f"Point label must have shape ({n},), got {value.shape}.")
            arrays.append(value)
    return np.column_stack(arrays)


def _stack_reconstruction(labels: ProfileLabels) -> np.ndarray | None:
    if labels.truth_t is None or labels.truth_s is None:
        return None
    t = np.asarray(labels.truth_t, dtype=float)
    s = np.asarray(labels.truth_s, dtype=float)
    if t.shape != s.shape:
        raise ValueError("truth_t and truth_s must have the same shape.")
    return np.column_stack([t, s]).astype("float32")


def _collate_optional_recon(recons: list[np.ndarray | None]) -> np.ndarray | None:
    if any(r is None for r in recons):
        return None
    shapes = {r.shape for r in recons if r is not None}
    if len(shapes) != 1:
        raise ValueError("All reconstruction targets must have the same shape.")
    return np.stack([r for r in recons if r is not None]).astype("float32")


def _collate_optional_pressure_grid(grids: list[np.ndarray | None]) -> np.ndarray | None:
    if any(g is None for g in grids):
        return None
    shapes = {g.shape for g in grids if g is not None}
    if len(shapes) != 1:
        raise ValueError("All pressure grids must have the same shape.")
    return np.stack([np.asarray(g, dtype=float) for g in grids if g is not None]).astype("float32")


def compute_normalization_stats(examples: Sequence[ProfileExample]) -> NormalizationStats:
    """Compute run-level T/S standardization stats from training examples."""
    temp_values: list[np.ndarray] = []
    sal_values: list[np.ndarray] = []
    for example in examples:
        if example.labels.truth_t is not None and example.labels.truth_s is not None:
            t = np.asarray(example.labels.truth_t, dtype=float).ravel()
            s = np.asarray(example.labels.truth_s, dtype=float).ravel()
        else:
            t = np.asarray(example.profile.temperature, dtype=float).ravel()
            s = np.asarray(example.profile.salinity, dtype=float).ravel()
        t = t[np.isfinite(t)]
        s = s[np.isfinite(s)]
        if t.size:
            temp_values.append(t)
        if s.size:
            sal_values.append(s)

    if not temp_values or not sal_values:
        raise ValueError("Cannot compute normalization stats from empty training examples.")

    temp_all = np.concatenate(temp_values)
    sal_all = np.concatenate(sal_values)
    return NormalizationStats(
        temperature_mean=float(np.mean(temp_all)),
        temperature_std=float(np.std(temp_all)) if float(np.std(temp_all)) > 0 else 1.0,
        salinity_mean=float(np.mean(sal_all)),
        salinity_std=float(np.std(sal_all)) if float(np.std(sal_all)) > 0 else 1.0,
    )
