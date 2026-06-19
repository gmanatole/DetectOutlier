# `outlierdetect.parquet`

This module handles round-tripping Argo profiles through parquet without flattening away the metadata needed for inference. The output stays one row per observed level so different profiles remain distinct and the original level ordering can be reconstructed later.

The parquet format is used as a transport layer, not a scientific abstraction. It exists so large profile collections can be loaded, filtered, and trained on efficiently while preserving the pressure, temperature, salinity, and location fields that the physics code depends on.

```{automodule} outlierdetect.parquet
:members:
:undoc-members:
:private-members:
:show-inheritance:
```
