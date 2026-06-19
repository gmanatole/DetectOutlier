"""Runtime configuration helpers for the CLI entry points.

The package keeps the model itself small, so the config layer handles the
larger concerns: path resolution, default TOML loading, input toggles, and the
optional heave-sigma source used during prediction.
"""

from __future__ import annotations

from argparse import Namespace
import json
from dataclasses import dataclass, field, fields, is_dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib  # type: ignore[no-redef]

from .data import ProfileInput

DEFAULT_TRAIN_RUN_ROOT = Path(__file__).resolve().parent / "train" / "data" / "run"
DEFAULT_PREDICT_RUN_ROOT = Path("artifacts") / "predict_runs"


@dataclass(slots=True)
class PathConfig:
    """Shared filesystem locations for train and predict."""

    data_root: str | Path | None = None
    model_checkpoint: str | Path | None = None
    train_run_root: str | Path | None = None
    predict_run_root: str | Path | None = None


@dataclass(slots=True)
class HeaveConfig:
    """Optional vertical-heave source.

    ``source`` can be:
    - ``False`` or ``None``: do not attempt to derive heave from an auxiliary source;
    - ``True``: use the profile's own sigma/heave metadata when available;
    - a path to a NetCDF file containing sigma and latitude/longitude/time metadata.
    """

    source: bool | str | Path | None = False


@dataclass(slots=True)
class InputConfig:
    """Booleans that control which optional profile inputs are populated."""

    residual_t: bool = True
    residual_s: bool = True
    sigma_t: bool = True
    sigma_s: bool = True
    sigma_vert: bool = True
    sigma_heave_t: bool = True
    sigma_heave_s: bool = True
    rho_ts: bool = True
    day_of_year: bool = True


@dataclass(slots=True)
class TrainConfig:
    """Train-specific defaults loaded from TOML."""

    train_root: str | Path | None = None
    test_root: str | Path | None = None
    data_source: str = "argo"
    output: str | Path | None = None
    seed: int = 4
    profile_limit: int | None = None
    n_examples_per_profile: int = 1
    n_levels: int = 20
    min_levels: int = 5
    grid_size: int = 80
    upper_ocean_bias: float = 1.7
    epochs: int = 5
    batch_size: int = 8
    learning_rate: float = 1e-3
    val_fraction: float = 0.1
    device: str = "cpu"
    run_root: str | Path | None = DEFAULT_TRAIN_RUN_ROOT
    epoch_plot_count: int = 10
    good_qc_only: bool = True
    test_augment: bool = False


@dataclass(slots=True)
class PredictConfig:
    """Predict-specific defaults loaded from TOML."""

    predict_root: str | Path | None = None
    checkpoint: str | Path | None = None
    pattern: str = "**/*.nc"
    min_levels: int = 5
    good_qc_only: bool = False
    profile_limit: int | None = None
    device: str = "cpu"
    run_root: str | Path | None = DEFAULT_PREDICT_RUN_ROOT
    seed: int = 4


@dataclass(slots=True)
class AppConfig:
    """Typed configuration loaded from TOML."""

    paths: PathConfig = field(default_factory=PathConfig)
    heave: HeaveConfig = field(default_factory=HeaveConfig)
    inputs: InputConfig = field(default_factory=InputConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    predict: PredictConfig = field(default_factory=PredictConfig)
    source_path: Path | None = None


def extract_config_path(argv: Sequence[str] | None) -> Path | None:
    """Return ``--config`` if present in *argv*."""
    import argparse

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path)
    args, _ = parser.parse_known_args([] if argv is None else list(argv))
    return args.config


def load_app_config(path: str | Path | None = None) -> AppConfig:
    """Load TOML config into typed dataclasses and resolve relative paths."""
    config = AppConfig()
    if path is None:
        return config

    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    if config_path.is_dir():
        raise IsADirectoryError(f"Config path must be a TOML file, not a directory: {config_path}")

    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    if not isinstance(raw, Mapping):
        raise TypeError("Config file did not parse into a mapping.")

    _merge_section(config.paths, raw.get("paths"))
    _merge_section(config.heave, raw.get("heave"))
    _merge_section(config.inputs, raw.get("inputs"))
    train_values = raw.get("train")
    if isinstance(train_values, Mapping):
        train_values = dict(train_values)
        if "train_root" not in train_values and "argo_root" in train_values:
            train_values["train_root"] = train_values["argo_root"]
    _merge_section(config.train, train_values)
    predict_values = raw.get("predict")
    if isinstance(predict_values, Mapping):
        predict_values = dict(predict_values)
        if "predict_root" not in predict_values and "argo_root" in predict_values:
            predict_values["predict_root"] = predict_values["argo_root"]
    _merge_section(config.predict, predict_values)
    config.source_path = config_path
    _resolve_config_paths(config, base_dir=config_path.parent)
    return config


def app_config_to_dict(config: AppConfig) -> dict[str, Any]:
    """Convert a config object into a JSON-friendly dictionary."""
    return {
        "source_path": None if config.source_path is None else str(config.source_path),
        "paths": _section_to_dict(config.paths),
        "heave": _section_to_dict(config.heave),
        "inputs": _section_to_dict(config.inputs),
        "train": _section_to_dict(config.train),
        "predict": _section_to_dict(config.predict),
    }


def example_app_config() -> AppConfig:
    """Return the starter configuration used by ``outlierdetect config init``.

    The values mirror the repository's example TOML file: relative data roots,
    artifacts under ``artifacts/``, and the default checkpoint path used by the
    current training workflow. This is a user-editable template, not a required
    runtime setting.
    """

    config = AppConfig()
    config.paths.data_root = "data"
    config.paths.model_checkpoint = "checkpoints/train_dataset_20ep.pt"
    config.paths.train_run_root = "artifacts/train_runs"
    config.paths.predict_run_root = "artifacts/predict_runs"

    config.heave.source = False

    config.inputs.residual_t = True
    config.inputs.residual_s = True
    config.inputs.sigma_t = True
    config.inputs.sigma_s = True
    config.inputs.sigma_vert = True
    config.inputs.sigma_heave_t = True
    config.inputs.sigma_heave_s = True
    config.inputs.rho_ts = True
    config.inputs.day_of_year = True

    config.train.train_root = "data"
    config.train.test_root = None
    config.train.data_source = "argo"
    config.train.output = "checkpoints/train_dataset_20ep.pt"
    config.train.seed = 4
    config.train.profile_limit = None
    config.train.n_examples_per_profile = 1
    config.train.n_levels = 20
    config.train.min_levels = 5
    config.train.grid_size = 80
    config.train.upper_ocean_bias = 1.7
    config.train.epochs = 20
    config.train.batch_size = 8
    config.train.learning_rate = 1e-3
    config.train.val_fraction = 0.1
    config.train.device = "cpu"
    config.train.run_root = "artifacts/train_runs"
    config.train.epoch_plot_count = 10
    config.train.good_qc_only = True
    config.train.test_augment = False

    config.predict.predict_root = "data"
    config.predict.checkpoint = "checkpoints/train_dataset_20ep.pt"
    config.predict.pattern = "**/*.nc"
    config.predict.min_levels = 5
    config.predict.good_qc_only = False
    config.predict.profile_limit = None
    config.predict.device = "cpu"
    config.predict.run_root = "artifacts/predict_runs"
    config.predict.seed = 4
    return config


def app_config_to_toml(config: AppConfig) -> str:
    """Render an ``AppConfig`` instance to TOML text.

    Only populated fields are written. This is useful for ``config show`` and
    for serializing a validated configuration snapshot.
    """

    sections: list[str] = []
    for section_name in ("paths", "heave", "inputs", "train", "predict"):
        section = getattr(config, section_name)
        rendered = _render_toml_section(section_name, section)
        if rendered:
            sections.append(rendered)
    return "\n\n".join(sections) + ("\n" if sections else "")


def default_config_toml() -> str:
    """Return the starter configuration template as TOML text."""

    header = [
        "# Default runtime config for OutlierDetect.",
        "#",
        "# Use `outlierdetect config init` to write this template to the current directory.",
        "# Use `outlierdetect config show` to print it to stdout.",
        "#",
        "# All paths are resolved relative to the config file location.",
        "",
    ]
    return "\n".join(header) + app_config_to_toml(example_app_config())


def write_default_config(path: str | Path, *, overwrite: bool = False) -> Path:
    """Write the starter TOML template to *path*.

    Parameters
    ----------
    path:
        Output file location.
    overwrite:
        If ``False`` and the file exists, raise ``FileExistsError``.
    """

    output = Path(path).expanduser()
    if output.exists() and not overwrite:
        raise FileExistsError(f"Config file already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(default_config_toml(), encoding="utf-8")
    return output


def train_parser_defaults(config: AppConfig) -> dict[str, Any]:
    """Return parser defaults for ``train_main``."""
    return {
        "train_root": _first_path(config.train.train_root, config.paths.data_root),
        "test_root": config.train.test_root,
        "data_source": config.train.data_source,
        "output": _first_path(config.train.output, config.paths.model_checkpoint),
        "seed": config.train.seed,
        "profile_limit": config.train.profile_limit,
        "n_examples_per_profile": config.train.n_examples_per_profile,
        "n_levels": config.train.n_levels,
        "min_levels": config.train.min_levels,
        "grid_size": config.train.grid_size,
        "upper_ocean_bias": config.train.upper_ocean_bias,
        "epochs": config.train.epochs,
        "batch_size": config.train.batch_size,
        "learning_rate": config.train.learning_rate,
        "val_fraction": config.train.val_fraction,
        "device": config.train.device,
        "run_root": _first_path(config.train.run_root, config.paths.train_run_root, DEFAULT_TRAIN_RUN_ROOT),
        "epoch_plot_count": config.train.epoch_plot_count,
        "good_qc_only": config.train.good_qc_only,
        "test_augment": config.train.test_augment,
        "use_residual_t": config.inputs.residual_t,
        "use_residual_s": config.inputs.residual_s,
        "use_sigma_t": config.inputs.sigma_t,
        "use_sigma_s": config.inputs.sigma_s,
        "use_sigma_vert": config.inputs.sigma_vert,
        "use_sigma_heave_t": config.inputs.sigma_heave_t,
        "use_sigma_heave_s": config.inputs.sigma_heave_s,
        "use_rho_ts": config.inputs.rho_ts,
        "use_day_of_year": config.inputs.day_of_year,
        "sigma_heave_source": config.heave.source,
    }


def predict_parser_defaults(config: AppConfig) -> dict[str, Any]:
    """Return parser defaults for ``predict_main``."""
    return {
        "predict_root": _first_path(config.predict.predict_root, config.paths.data_root),
        "checkpoint": _first_path(config.predict.checkpoint, config.paths.model_checkpoint),
        "pattern": config.predict.pattern,
        "min_levels": config.predict.min_levels,
        "good_qc_only": config.predict.good_qc_only,
        "profile_limit": config.predict.profile_limit,
        "device": config.predict.device,
        "run_root": _first_path(config.predict.run_root, config.paths.predict_run_root, DEFAULT_PREDICT_RUN_ROOT),
        "seed": config.predict.seed,
        "use_residual_t": config.inputs.residual_t,
        "use_residual_s": config.inputs.residual_s,
        "use_sigma_t": config.inputs.sigma_t,
        "use_sigma_s": config.inputs.sigma_s,
        "use_sigma_vert": config.inputs.sigma_vert,
        "use_sigma_heave_t": config.inputs.sigma_heave_t,
        "use_sigma_heave_s": config.inputs.sigma_heave_s,
        "use_rho_ts": config.inputs.rho_ts,
        "use_day_of_year": config.inputs.day_of_year,
        "sigma_heave_source": config.heave.source,
    }


def resolve_profile_input(
    profile: ProfileInput,
    *,
    use_residual_t: bool = True,
    use_residual_s: bool = True,
    use_sigma_t: bool = True,
    use_sigma_s: bool = True,
    use_sigma_vert: bool = True,
    use_sigma_heave_t: bool = True,
    use_sigma_heave_s: bool = True,
    use_rho_ts: bool = True,
    use_day_of_year: bool = True,
    sigma_heave_source: bool | str | Path | None = False,
) -> ProfileInput:
    """Return a profile with optional fields enabled/disabled per config."""
    attrs = dict(profile.attrs)
    sigma_vert = _extract_sigma_vert(profile)
    if sigma_vert is None and isinstance(sigma_heave_source, Path):
        sigma_vert = _load_sigma_heave_source(sigma_heave_source, profile, sigma_vert=sigma_vert)
    elif sigma_heave_source is True and sigma_vert is None:
        sigma_vert = _extract_sigma_from_attrs(profile)

    sigma_heave_t = profile.sigma_heave_t
    sigma_heave_s = profile.sigma_heave_s
    if sigma_vert is not None and (sigma_heave_t is None or sigma_heave_s is None):
        computed_t, computed_s = _compute_sigma_heave(profile, sigma_vert)
        if sigma_heave_t is None:
            sigma_heave_t = computed_t
        if sigma_heave_s is None:
            sigma_heave_s = computed_s

    return ProfileInput(
        pressure=profile.pressure,
        temperature=profile.temperature,
        salinity=profile.salinity,
        residual_t=profile.residual_t if use_residual_t else None,
        residual_s=profile.residual_s if use_residual_s else None,
        sigma_t=profile.sigma_t if use_sigma_t else None,
        sigma_s=profile.sigma_s if use_sigma_s else None,
        sigma_vert=sigma_vert if use_sigma_vert else None,
        sigma_heave_t=sigma_heave_t if use_sigma_heave_t else None,
        sigma_heave_s=sigma_heave_s if use_sigma_heave_s else None,
        rho_ts=profile.rho_ts if use_rho_ts else None,
        day_of_year=profile.day_of_year if use_day_of_year else None,
        profile_id=profile.profile_id,
        attrs=attrs,
    )


def resolved_run_config_dict(
    *,
    config: AppConfig,
    command: str,
    args: Namespace,
) -> dict[str, Any]:
    """Return the fully resolved config snapshot that should be saved with a run."""
    resolved = app_config_to_dict(config)
    runtime = _namespace_to_dict(args)
    _apply_runtime_overrides(resolved, command=command, runtime=runtime)
    resolved["command"] = command
    resolved["runtime"] = runtime
    resolved["runtime"]["command"] = command
    return resolved


def _merge_section(instance: Any, values: Any) -> None:
    if values is None:
        return
    if not isinstance(values, Mapping):
        raise TypeError(f"Config section for {type(instance).__name__} must be a mapping.")
    valid_fields = {field.name for field in fields(instance)}
    for key, value in values.items():
        if key in valid_fields:
            setattr(instance, key, value)


def _resolve_config_paths(config: AppConfig, *, base_dir: Path) -> None:
    config.paths.data_root = _resolve_path_value(config.paths.data_root, base_dir)
    config.paths.model_checkpoint = _resolve_path_value(config.paths.model_checkpoint, base_dir)
    config.paths.train_run_root = _resolve_path_value(config.paths.train_run_root, base_dir)
    config.paths.predict_run_root = _resolve_path_value(config.paths.predict_run_root, base_dir)

    config.train.train_root = _resolve_path_value(config.train.train_root, base_dir)
    config.train.test_root = _resolve_path_value(config.train.test_root, base_dir)
    config.train.output = _resolve_path_value(config.train.output, base_dir)
    config.train.run_root = _resolve_path_value(config.train.run_root, base_dir)
    config.predict.predict_root = _resolve_path_value(config.predict.predict_root, base_dir)
    config.predict.checkpoint = _resolve_path_value(config.predict.checkpoint, base_dir)
    config.predict.run_root = _resolve_path_value(config.predict.run_root, base_dir)

    config.heave.source = _resolve_heave_source(config.heave.source, base_dir)


def _resolve_path_value(value: str | Path | None, base_dir: Path) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _resolve_heave_source(value: bool | str | Path | None, base_dir: Path) -> bool | Path | None:
    if isinstance(value, bool) or value is None:
        return value
    return _resolve_path_value(value, base_dir)


def _first_path(*values: str | Path | None) -> Path | None:
    for value in values:
        if value is not None:
            return Path(value)
    return None


def _namespace_to_dict(namespace: Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in vars(namespace).items():
        if isinstance(value, Path):
            out[key] = str(value)
        elif isinstance(value, np.generic):
            out[key] = value.item()
        else:
            out[key] = value
    return out


def _apply_runtime_overrides(
    resolved: dict[str, Any],
    *,
    command: str,
    runtime: Mapping[str, Any],
) -> None:
    inputs = resolved.get("inputs")
    heave = resolved.get("heave")
    paths = resolved.get("paths")
    train = resolved.get("train")
    predict = resolved.get("predict")

    if command == "train":
        train_root = runtime.get("train_root", runtime.get("argo_root"))
        _set_if_not_none(paths, "data_root", train_root)
        _set_if_not_none(paths, "model_checkpoint", runtime.get("output"))
        _set_if_not_none(paths, "train_run_root", runtime.get("run_root"))
        _set_if_not_none(train, "train_root", train_root)
        _set_if_not_none(train, "test_root", runtime.get("test_root"))
        _set_if_not_none(train, "data_source", runtime.get("data_source"))
        _set_if_not_none(train, "output", runtime.get("output"))
        _set_if_not_none(train, "seed", runtime.get("seed"))
        _set_if_not_none(train, "profile_limit", runtime.get("profile_limit"))
        _set_if_not_none(train, "n_examples_per_profile", runtime.get("n_examples_per_profile"))
        _set_if_not_none(train, "n_levels", runtime.get("n_levels"))
        _set_if_not_none(train, "min_levels", runtime.get("min_levels"))
        _set_if_not_none(train, "grid_size", runtime.get("grid_size"))
        _set_if_not_none(train, "upper_ocean_bias", runtime.get("upper_ocean_bias"))
        _set_if_not_none(train, "epochs", runtime.get("epochs"))
        _set_if_not_none(train, "batch_size", runtime.get("batch_size"))
        _set_if_not_none(train, "learning_rate", runtime.get("learning_rate"))
        _set_if_not_none(train, "val_fraction", runtime.get("val_fraction"))
        _set_if_not_none(train, "device", runtime.get("device"))
        _set_if_not_none(train, "run_root", runtime.get("run_root"))
        _set_if_not_none(train, "epoch_plot_count", runtime.get("epoch_plot_count"))
        _set_if_not_none(train, "good_qc_only", runtime.get("good_qc_only"))
        _set_if_not_none(train, "test_augment", runtime.get("test_augment"))
    elif command == "predict":
        predict_root = runtime.get("predict_root", runtime.get("argo_root"))
        _set_if_not_none(paths, "data_root", predict_root)
        _set_if_not_none(paths, "model_checkpoint", runtime.get("checkpoint"))
        _set_if_not_none(paths, "predict_run_root", runtime.get("run_root"))
        _set_if_not_none(predict, "predict_root", predict_root)
        _set_if_not_none(predict, "checkpoint", runtime.get("checkpoint"))
        _set_if_not_none(predict, "pattern", runtime.get("pattern"))
        _set_if_not_none(predict, "min_levels", runtime.get("min_levels"))
        _set_if_not_none(predict, "good_qc_only", runtime.get("good_qc_only"))
        _set_if_not_none(predict, "profile_limit", runtime.get("profile_limit"))
        _set_if_not_none(predict, "device", runtime.get("device"))
        _set_if_not_none(predict, "run_root", runtime.get("run_root"))
        _set_if_not_none(predict, "seed", runtime.get("seed"))

    if isinstance(inputs, dict):
        _set_if_not_none(inputs, "residual_t", runtime.get("use_residual_t"))
        _set_if_not_none(inputs, "residual_s", runtime.get("use_residual_s"))
        _set_if_not_none(inputs, "sigma_t", runtime.get("use_sigma_t"))
        _set_if_not_none(inputs, "sigma_s", runtime.get("use_sigma_s"))
        _set_if_not_none(inputs, "sigma_vert", runtime.get("use_sigma_vert"))
        _set_if_not_none(inputs, "sigma_heave_t", runtime.get("use_sigma_heave_t"))
        _set_if_not_none(inputs, "sigma_heave_s", runtime.get("use_sigma_heave_s"))
        _set_if_not_none(inputs, "rho_ts", runtime.get("use_rho_ts"))
        _set_if_not_none(inputs, "day_of_year", runtime.get("use_day_of_year"))

    if isinstance(heave, dict):
        _set_if_not_none(heave, "source", runtime.get("sigma_heave_source"))


def _set_if_not_none(mapping: Mapping[str, Any] | dict[str, Any] | Any, key: str, value: Any) -> None:
    if value is None or not isinstance(mapping, dict):
        return
    mapping[key] = _jsonify(value)


def _section_to_dict(section: Any) -> dict[str, Any]:
    if not is_dataclass(section):
        raise TypeError(f"Expected dataclass section, got {type(section)!r}.")
    out: dict[str, Any] = {}
    for item in fields(section):
        out[item.name] = _jsonify(getattr(section, item.name))
    return out


def _render_toml_section(section_name: str, section: Any) -> str:
    if not is_dataclass(section):
        raise TypeError(f"Expected dataclass section, got {type(section)!r}.")

    lines = [f"[{section_name}]"]
    for item in fields(section):
        value = getattr(section, item.name)
        if value is None:
            continue
        lines.append(f"{item.name} = {_toml_literal(value)}")
    if len(lines) == 1:
        return ""
    return "\n".join(lines)


def _toml_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Path):
        return json.dumps(str(value))
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        return repr(float(value))
    if isinstance(value, Mapping):
        items = ", ".join(f"{key} = {_toml_literal(item)}" for key, item in value.items())
        return "{ " + items + " }"
    if isinstance(value, tuple):
        return "[ " + ", ".join(_toml_literal(item) for item in value) + " ]"
    if isinstance(value, list):
        return "[ " + ", ".join(_toml_literal(item) for item in value) + " ]"
    return json.dumps(str(value))


def _jsonify(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return {field.name: _jsonify(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonify(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonify(item) for item in value]
    return value


def _extract_sigma_vert(profile: ProfileInput) -> np.ndarray | None:
    sigma = profile.sigma_vert
    if sigma is None:
        sigma = _extract_sigma_from_attrs(profile)
    if sigma is None:
        return None
    return _coerce_sigma_array(sigma, profile.n_levels)


def _extract_sigma_from_attrs(profile: ProfileInput) -> np.ndarray | None:
    for key in ("sigma_vert", "sigma", "heave_sigma", "sigma_heave"):
        value = profile.attrs.get(key)
        if value is not None:
            return _coerce_sigma_array(value, profile.n_levels)
    return None


def _compute_sigma_heave(profile: ProfileInput, sigma_vert: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
    sigma_arr = _coerce_sigma_array(sigma_vert, profile.n_levels)
    if sigma_arr is None:
        return np.zeros(profile.n_levels, dtype=float), np.zeros(profile.n_levels, dtype=float)

    temp_grad = _safe_gradient(profile.temperature, profile.pressure)
    sal_grad = _safe_gradient(profile.salinity, profile.pressure)
    sigma_heave_t = np.abs(temp_grad) * sigma_arr
    sigma_heave_s = np.abs(sal_grad) * sigma_arr
    return sigma_heave_t.astype(float), sigma_heave_s.astype(float)


def _safe_gradient(values: np.ndarray, pressure: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    p = np.asarray(pressure, dtype=float)
    if x.size != p.size:
        raise ValueError("pressure and values must have the same length.")
    finite = np.isfinite(x) & np.isfinite(p)
    if int(np.sum(finite)) < 2:
        return np.zeros_like(x, dtype=float)
    filled = _fill_missing(x, p)
    try:
        grad = np.gradient(filled, p, edge_order=1)
    except Exception:
        grad = np.zeros_like(filled, dtype=float)
    return np.nan_to_num(np.asarray(grad, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)


def _fill_missing(values: np.ndarray, pressure: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float).copy()
    p = np.asarray(pressure, dtype=float)
    finite = np.isfinite(x) & np.isfinite(p)
    if int(np.sum(finite)) == 0:
        return np.zeros_like(x, dtype=float)
    if int(np.sum(finite)) == 1:
        x[:] = x[finite][0]
        return x
    x[~finite] = np.interp(p[~finite], p[finite], x[finite])
    return x


def _coerce_sigma_array(value: Any, n_levels: int) -> np.ndarray | None:
    arr = np.asarray(value, dtype=float)
    arr = np.squeeze(arr)
    if arr.size == 0:
        return None
    if arr.ndim == 0:
        return np.full(n_levels, float(arr), dtype=float)
    if arr.size == n_levels:
        return np.asarray(arr, dtype=float).reshape(n_levels)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None
    return np.full(n_levels, float(np.nanmean(finite)), dtype=float)


def _load_sigma_heave_source(path: Path, profile: ProfileInput, *, sigma_vert: np.ndarray | None = None) -> np.ndarray | None:
    """Load an auxiliary sigma-heave NetCDF file and pick the best sigma profile."""
    source = _read_sigma_source(str(path))
    sigma = source.get("sigma")
    if sigma is None:
        return sigma_vert

    sigma_arr = np.asarray(sigma, dtype=float).squeeze()
    if sigma_arr.ndim == 0:
        return np.full(profile.n_levels, float(sigma_arr), dtype=float)
    if sigma_arr.ndim == 1 and sigma_arr.size == profile.n_levels:
        return sigma_arr.astype(float)

    coord_lat = _coerce_optional_1d(source.get("latitude"))
    coord_lon = _coerce_optional_1d(source.get("longitude"))
    coord_time = _coerce_optional_1d(source.get("time"))
    lat = _profile_attr_float(profile, "latitude")
    lon = _profile_attr_float(profile, "longitude")
    juld = _profile_attr_float(profile, "juld")
    if lat is None and lon is None and juld is None:
        return _coerce_sigma_array(sigma_arr, profile.n_levels)

    if coord_lat is None and coord_lon is None and coord_time is None:
        return _coerce_sigma_array(sigma_arr, profile.n_levels)

    index = _select_source_index(coord_lat, coord_lon, coord_time, lat, lon, juld)
    if index is None:
        return _coerce_sigma_array(sigma_arr, profile.n_levels)
    index = min(max(int(index), 0), sigma_arr.shape[0] - 1)

    if sigma_arr.ndim == 1:
        return np.full(profile.n_levels, float(sigma_arr.reshape(-1)[index]), dtype=float)

    picked = np.take(sigma_arr, index, axis=0)
    coerced = _coerce_sigma_array(picked, profile.n_levels)
    if coerced is not None:
        return coerced
    finite = sigma_arr[np.isfinite(sigma_arr)]
    if finite.size:
        return np.full(profile.n_levels, float(np.nanmean(finite)), dtype=float)
    return sigma_vert


@lru_cache(maxsize=8)
def _read_sigma_source(path: str) -> dict[str, Any]:
    ds = _open_sigma_source(Path(path))
    try:
        sigma = _read_first_available(ds, ("sigma", "sigma_vert", "heave_sigma", "sigma_heave"))
        latitude = _read_first_available(ds, ("latitude", "lat", "LATITUDE", "LAT"))
        longitude = _read_first_available(ds, ("longitude", "lon", "LONGITUDE", "LON"))
        time = _read_first_available(ds, ("time", "juld", "JULD", "TIME"))
        return {
            "sigma": sigma,
            "latitude": latitude,
            "longitude": longitude,
            "time": time,
        }
    finally:
        try:
            ds.close()
        except Exception:
            pass


def _open_sigma_source(path: Path) -> Any:
    path_str = str(path)
    try:
        import netCDF4

        return netCDF4.Dataset(path_str, "r")
    except Exception:
        pass

    try:
        from scipy.io import netcdf_file

        return netcdf_file(path_str, "r", mmap=False)
    except Exception as exc:
        raise RuntimeError(
            f"Could not open auxiliary sigma source {path_str!s}. Install netCDF4 or scipy."
        ) from exc


def _read_first_available(ds: Any, names: Sequence[str]) -> Any:
    for name in names:
        if _has_var(ds, name):
            raw = np.asarray(ds[name][:], dtype=float)
            raw = np.squeeze(raw)
            if raw.size == 0:
                continue
            return raw
    return None


def _has_var(ds: Any, name: str) -> bool:
    variables = getattr(ds, "variables", None)
    if variables is not None:
        try:
            return name in variables
        except Exception:
            pass
    try:
        return name in ds
    except Exception:
        return False


def _coerce_optional_1d(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=float).squeeze()
    if arr.ndim == 0:
        return np.asarray([float(arr)], dtype=float)
    if arr.ndim == 1:
        return arr.astype(float)
    return arr.reshape(-1).astype(float)


def _profile_attr_float(profile: ProfileInput, key: str) -> float | None:
    value = profile.attrs.get(key)
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return out


def _select_source_index(
    latitude: np.ndarray | None,
    longitude: np.ndarray | None,
    time: np.ndarray | None,
    lat: float | None,
    lon: float | None,
    juld: float | None,
) -> int | None:
    arrays = [arr for arr in (latitude, longitude, time) if arr is not None]
    n = max((arr.size for arr in arrays), default=0)
    if n == 0:
        return None

    best_index: int | None = None
    best_distance = float("inf")
    for index in range(n):
        dist = 0.0
        if latitude is not None and lat is not None and index < latitude.size and np.isfinite(latitude[index]):
            dist += float((latitude[index] - lat) ** 2)
        if longitude is not None and lon is not None and index < longitude.size and np.isfinite(longitude[index]):
            dist += float((longitude[index] - lon) ** 2)
        if time is not None and juld is not None and index < time.size and np.isfinite(time[index]):
            dist += float(((time[index] - juld) / 30.0) ** 2)
        if dist < best_distance:
            best_distance = dist
            best_index = index
    return best_index
