# `outlierdetect.argo`

This module reads Argo NetCDF profiles and turns them into `ProfileInput` objects without destroying the profile's own vertical structure. That matters because the QC and reconstruction logic depend on the observed pressure grid, not on an externally imposed resampling mesh.

The code also preserves latitude, longitude, and time metadata when present. Those fields are used to evaluate TEOS-10 density and to give physical context to a sparse profile.

```{automodule} outlierdetect.argo
:members:
:undoc-members:
:private-members:
:show-inheritance:
```
