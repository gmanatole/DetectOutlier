# Configuration Workflow

OutlierDetect supports an optional user-editable TOML configuration file. The package does not require a config file to be importable, but the CLI can load one to set default paths and input toggles for training and prediction.

## Why the config is external

The configuration is treated as user state, not package state. That keeps the library usable from Python without local files, and it keeps the installed package immutable. Users can copy or generate a config file in their own project directory and update it without touching the package install.

## Commands

`outlierdetect config init`

: Write a starter `outlierdetect.toml` to the current directory. This is the recommended starting point for a new project.

`outlierdetect config show`

: Print the starter TOML template to stdout. This is useful when you want to inspect or redirect the default configuration without creating a file first.

`outlierdetect config validate --config path/to/outlierdetect.toml`

: Load a specific config file, resolve its relative paths, and print the resolved configuration snapshot as JSON. This is useful when debugging path resolution or checking that the file still parses cleanly.

## What the template contains

The starter file includes:

- a shared data root,
- checkpoint and artifact paths,
- toggles for residuals and uncertainty inputs,
- the training defaults used by the current pipeline, including the clean-profile source,
- the prediction defaults used by the current pipeline,
- the heave-source switch.

The template is intentionally conservative. It is designed to be a readable starting point rather than a hidden source of truth. The actual defaults still live in the code, and any TOML file can override them.
