# `outlierdetect.model`

This module contains the compact transformer backbone used by the package. The design is intentionally small: the model needs enough capacity to read local level context, but it should remain easy to train and easy to inspect.

The network predicts three point-level probabilities, a profile-level flag, a nuisance posterior summary, and a reconstruction over a fixed pressure grid. That split reflects the physical structure of the problem. Point flags catch local corruption, the profile flag captures whether the whole profile is suspect, and the reconstruction head tries to recover a stable T/S profile that is consistent with the observed features.

```{automodule} outlierdetect.model
:members:
:undoc-members:
:private-members:
:show-inheritance:
```
