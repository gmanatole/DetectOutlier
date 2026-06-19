# `outlierdetect.training.artifacts`

This module writes the training and prediction artifacts: progress snapshots, probability JSON, and reconstruction plots. It is the bridge between model execution and human inspection.

The plot writer uses the same point-level probabilities and the same density contour logic as the rest of the package, so the PNGs are not just pretty pictures. They are a visual serialization of the same physical quantities that the predictor is using internally.

```{automodule} outlierdetect.training.artifacts
:members:
:undoc-members:
:private-members:
:show-inheritance:
```
