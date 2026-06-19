# `outlierdetect.features`

The feature builder turns a sparse profile into a per-level tensor for both the neural network and the heuristic predictor. The inputs are chosen to expose the geometry and local physics that matter most: normalized pressure, local gaps, gradients, residuals, uncertainty scales, and density consistency.

The derived nuisance posterior is included as a feature source because the model should see the same correction structure that the physics layer computes. That keeps the learning path and the heuristic path aligned instead of letting them drift apart.

```{automodule} outlierdetect.features
:members:
:undoc-members:
:private-members:
:show-inheritance:
```
