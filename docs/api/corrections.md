# `outlierdetect.corrections`

This is the Bayesian nuisance layer. It models local T/S bias as a Gaussian latent variable with a physically motivated prior, posterior uncertainty, and explicit correlation between temperature and salinity corrections.

The method is deliberately linear in pressure because the correction being estimated is a local calibration-like nuisance field, not a full ocean state estimate. A linear pressure coordinate keeps the posterior interpretable, keeps the prior compact, and makes it easy to convert the result into a correction profile and uncertainty summary.

```{automodule} outlierdetect.corrections
:members:
:undoc-members:
:private-members:
:show-inheritance:
```
