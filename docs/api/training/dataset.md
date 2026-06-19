# `outlierdetect.training.dataset`

This module defines the PyTorch dataset and padding logic. The dataset is profile-centric rather than row-centric because the model consumes a whole sparse vertical profile at once.

The batch collation preserves masks and target shapes so the training loop can compute losses only where labels exist. That is important for CTD-SRDL-style data, where some examples have reconstruction targets and others only have point-level labels.

```{automodule} outlierdetect.training.dataset
:members:
:undoc-members:
:private-members:
:show-inheritance:
```
