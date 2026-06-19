# `outlierdetect.training.synthetic`

This is the synthetic corruption model used to create labeled training examples from trusted high-resolution profiles. The code injects the kinds of errors the model is meant to detect: coherent T/S bias, spikes, pressure mistakes, and reference mismatch from vertical heave.

The labels are built from the same physical ideas used in inference. Density inconsistency is derived from the inversion metric, not from a separate hand-wavy rule, so the training targets and the deployed physics stay aligned.

```{automodule} outlierdetect.training.synthetic
:members:
:undoc-members:
:private-members:
:show-inheritance:
```
