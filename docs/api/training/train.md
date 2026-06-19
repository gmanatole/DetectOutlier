# `outlierdetect.training.train`

This module is the optimization loop. It defines the multitask loss, the nuisance prior KL term, and the epoch-level training and evaluation helpers.

The extra prior KL term is there because the nuisance coefficients are not just another regression target. They are a physical latent variable with a preferred scale and covariance structure. Penalizing deviation from that prior keeps the learned nuisance head anchored to the oceanographic interpretation we want.

```{automodule} outlierdetect.training.train
:members:
:undoc-members:
:private-members:
:show-inheritance:
```
