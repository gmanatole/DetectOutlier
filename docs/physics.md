# Physical Model

This package is not just a classifier. It tries to preserve the oceanographic meaning of the inputs while detecting where a profile is locally inconsistent.

## TEOS-10 Density

The current inference path uses the `gsw` package and TEOS-10 rather than a linear density proxy. That choice matters because the question being asked is not "is temperature large?" or "is salinity large?" but "does this T/S pair correspond to a physically stable water parcel at this pressure and location?".

For that reason the code computes `sigma0` from Absolute Salinity and Conservative Temperature, using station latitude and longitude when they are available. `sigma0` is a practical static-stability variable because it expresses density referenced to the surface pressure and is directly suited to comparing adjacent levels in a sparse profile.

## Nuisance Corrections

The nuisance correction is treated as a 4D latent variable:

- `a_t`: temperature offset,
- `b_t`: temperature slope with pressure,
- `a_s`: salinity offset,
- `b_s`: salinity slope with pressure.

The model uses a Gaussian prior and posterior because the physical interpretation is simple and useful: coherent calibration bias is expected to vary smoothly with pressure, not as isolated spikes. A prior is preferable to a hard constraint because the data still need to pull the estimate away from the default when the profile supports it.

The prior is intentionally asymmetric. Salinity offsets are allowed to be larger than temperature offsets because salinity calibration and effective salinity bias often move more strongly than temperature bias in practical data. The slopes are kept tight, around a few hundredths per km, because a large linear drift over depth would imply a correction field that is too aggressive for the nuisance term the model is meant to represent.

Temperature and salinity corrections are correlated because the bias terms do not live in isolation. In a real instrument or reference-mismatch scenario, a shift in temperature and a shift in salinity can come from the same underlying source. The posterior keeps that relationship visible by returning the covariance and the derived correlation matrix.

## Stability Projection

Reconstruction uses a salinity-only stability projection as a pragmatic correction step. The goal is not to invent a full inverse ocean model; it is to ensure the reconstructed profile does not violate monotonic density ordering. Temperature is kept fixed and salinity is nudged only as much as needed to make sigma0 non-decreasing.

That is a conservative choice. It preserves the observed temperature structure, which is often the harder part of the profile to reconstruct robustly, while allowing salinity to absorb the minimum amount of correction needed to restore static stability.

## Synthetic Labels

The synthetic corruption pipeline produces the point labels used for training. Temperature and salinity spikes are modeled explicitly, and density inconsistency is derived from the TEOS-10 inversion metric rather than from a hand-written heuristic. This keeps the model aligned with the physical definition of an unstable T/S pair.

## Why the Model is Local

The inference path is intentionally local. The model should decide from the profile itself, its local uncertainty scales, and the derived density behavior, not from a full synoptic ocean state. That design keeps the system deployable on sparse profile data while still respecting the physics that matter for QC.
