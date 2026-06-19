# `outlierdetect.density`

This module now carries the actual density physics for the package. The earlier linear proxy is no longer the inference basis; the code uses TEOS-10 through `gsw` so density consistency is computed from the same equation of state that oceanographers expect.

The functions in this module do three jobs:

1. compute sigma0 from T/S and location metadata,
2. detect density inversions along a sparse profile,
3. project a profile back onto a statically stable density sequence using the minimum salinity adjustment needed.

That last step is intentionally conservative. It is not a full inverse model. It is a monotonicity repair that respects the observed temperature field and changes salinity only enough to remove the local instability.

```{automodule} outlierdetect.density
:members:
:undoc-members:
:private-members:
:show-inheritance:
```
