# `outlierdetect.data`

This module defines the core data containers passed between ingestion, feature building, prediction, training, and artifact writing. The dataclasses are intentionally explicit because the package needs to preserve both the observed profile and the derived physical diagnostics without burying them in an opaque dictionary.

The `Result` object is the main output contract. It carries the point-level probabilities, the profile-level probability, the nuisance estimate, and the reconstruction products in a single structured object so downstream code can serialize or inspect the same result without re-running inference.

```{automodule} outlierdetect.data
:members:
:undoc-members:
:private-members:
:show-inheritance:
```
