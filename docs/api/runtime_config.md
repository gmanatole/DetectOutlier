# `outlierdetect.runtime_config`

This module centralizes runtime configuration, CLI defaults, and profile-input toggles. It exists so the operational surface does not leak into the model code.

The important physical behavior here is not prediction itself but input resolution: whether residuals are present, whether heave uncertainty should be derived, whether latitude and longitude should be carried through, and whether the profile should be prepared in a form suitable for TEOS-10 density calculations. That separation keeps the inference path clean and makes the runtime behavior reproducible.

```{automodule} outlierdetect.runtime_config
:members:
:undoc-members:
:private-members:
:show-inheritance:
```
