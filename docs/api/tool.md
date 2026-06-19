# `outlierdetect.tool`

This is the inference layer. It combines the feature builder, the point-level outlier heads, the profile-level flag, the nuisance posterior, and the reconstruction step into a single prediction API.

Two predictor styles live here:

- `Heuristic`, which is a transparent rule-based baseline and physical sanity check,
- `Neural`, which wraps the trained PyTorch model while keeping the same output contract.

The two predictors share the same result object because the downstream code should not have to care whether the probabilities came from a heuristic or a learned model. The physics layer remains the same in both paths.

```{automodule} outlierdetect.tool
:members:
:undoc-members:
:private-members:
:show-inheritance:
```
