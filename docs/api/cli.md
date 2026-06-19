# `outlierdetect.cli`

The CLI module is intentionally thin. It parses the operational flags, resolves configuration, and delegates to the model and training layers. The code stays small because the scientific logic should not be scattered across argument parsing and file handling.

That thinness is a design choice, not an omission. It keeps the command-line interface easy to inspect while leaving the physical model, the data transformation, and the training objective in their own modules.

```{automodule} outlierdetect.cli
:members:
:undoc-members:
:private-members:
:show-inheritance:
```
