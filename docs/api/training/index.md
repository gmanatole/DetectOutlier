# `outlierdetect.training`

The training package turns clean Argo or EN4 profiles into synthetic examples, batches them for PyTorch, trains the model, and records artifacts. It is split into submodules because the synthetic data logic, the batch collation, the optimization loop, and the run writer each have different responsibilities.

```{toctree}
:maxdepth: 2

argo
en4
dataset
synthetic
train
artifacts
```

```{automodule} outlierdetect.training
:members:
:undoc-members:
:private-members:
:show-inheritance:
```
