"""Training utilities for the profile QC model."""

from .argo import ArgoTrainingConfig, build_argo_dataset, build_argo_examples, build_argo_synthetic_examples
from .dataset import (
    Dataset,
    Example,
    Labels,
    ProfileDataset,
    ProfileExample,
    ProfileLabels,
    collate,
    collate_profiles,
    compute_normalization_stats,
)
from .synthetic import SyntheticExample, degrade_highres_profile
from .train import (
    LossWeights,
    compute_loss,
    eval_epoch,
    fit_model,
    load_checkpoint,
    load_model_from_checkpoint,
    loss,
    save_checkpoint,
    train_epoch,
)

__all__ = [
    "ArgoTrainingConfig",
    "SyntheticExample",
    "Dataset",
    "Example",
    "Labels",
    "ProfileDataset",
    "ProfileExample",
    "ProfileLabels",
    "compute_normalization_stats",
    "collate",
    "collate_profiles",
    "degrade_highres_profile",
    "build_argo_dataset",
    "build_argo_examples",
    "build_argo_synthetic_examples",
    "LossWeights",
    "compute_loss",
    "loss",
    "train_epoch",
    "eval_epoch",
    "fit_model",
    "save_checkpoint",
    "load_checkpoint",
    "load_model_from_checkpoint",
]
