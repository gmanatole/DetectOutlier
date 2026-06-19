# `outlierdetect.training.argo`

This module bridges clean Argo profiles to synthetic training examples. It preserves each profile's own pressure grid before degradation, which matters because the model needs to learn from the sampling geometry it will later see in practice.

Latitude and longitude are propagated into the synthetic examples so the same TEOS-10 density logic can be used during training and inference. That keeps the synthetic path physically aligned with the operational one.

```{automodule} outlierdetect.training.argo
:members:
:undoc-members:
:private-members:
:show-inheritance:
```
