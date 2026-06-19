"""Training utilities for the profile QC model.

This package groups the synthetic-data generator, dataset/collation helpers,
loss definitions, and training loop used to fit the neural QC model from clean
Argo or EN4 profiles. The
separation keeps the physical assumptions visible: the synthetic generator
creates labels from known oceanographic corruption modes, the dataset preserves
profile structure, and the loss keeps the nuisance posterior anchored to the
physical prior.
"""

from .argo import ArgoTrainingConfig, build_argo_dataset, build_argo_examples, build_argo_synthetic_examples
from .en4 import build_en4_dataset, build_en4_examples, build_en4_synthetic_examples
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
    "build_en4_dataset",
    "build_en4_examples",
    "build_en4_synthetic_examples",
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
