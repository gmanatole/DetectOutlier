"""Thin command-line entry points for the package workflows.

The CLI is deliberately narrow. It parses file paths and flags, resolves the
runtime configuration, and delegates to the density, inference, and training
modules where the physical logic lives.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import warnings

import numpy as np

from .argo import ArgoProfile, iter_argo_files
from .data import NormalizationAccumulator
from .en4 import iter_en4_files
from .runtime_config import (
    AppConfig,
    app_config_to_dict,
    extract_config_path,
    default_config_toml,
    load_app_config,
    predict_parser_defaults,
    resolve_profile_input,
    resolve_reference_source_mode,
    resolved_run_config_dict,
    train_parser_defaults,
    write_default_config,
)
from .tool import Config, Neural
from .training.argo import build_argo_synthetic_examples
from .training.en4 import build_en4_synthetic_examples
from .training.artifacts import TrainingRunWriter
from .training.dataset import ProfileDataset, ProfileExample, collate_profiles, compute_normalization_stats
from .training.streaming import (
    StreamingProfileDataset,
    list_source_files,
    make_training_example_factory,
    preview_stream_examples,
    split_source_files,
)
from .training.train import fit_model, load_model_from_checkpoint, save_checkpoint


def _add_input_toggle_arguments(parser: argparse.ArgumentParser, defaults: dict[str, object]) -> None:
    group = parser.add_argument_group("Input selection")
    group.add_argument(
        "--use-residual-t",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults["use_residual_t"]),
        help="Feed residual_t into the model",
    )
    group.add_argument(
        "--use-residual-s",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults["use_residual_s"]),
        help="Feed residual_s into the model",
    )
    group.add_argument(
        "--use-sigma-t",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults["use_sigma_t"]),
        help="Feed sigma_t into the model",
    )
    group.add_argument(
        "--use-sigma-s",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults["use_sigma_s"]),
        help="Feed sigma_s into the model",
    )
    group.add_argument(
        "--use-sigma-vert",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults["use_sigma_vert"]),
        help="Feed sigma_vert into the model",
    )
    group.add_argument(
        "--use-sigma-heave-t",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults["use_sigma_heave_t"]),
        help="Feed sigma_heave_t into the model",
    )
    group.add_argument(
        "--use-sigma-heave-s",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults["use_sigma_heave_s"]),
        help="Feed sigma_heave_s into the model",
    )
    group.add_argument(
        "--use-rho-ts",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults["use_rho_ts"]),
        help="Feed rho_ts into the model",
    )
    group.add_argument(
        "--use-day-of-year",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults["use_day_of_year"]),
        help="Feed day_of_year into the model",
    )


def _parse_sigma_heave_source_arg(value: str) -> bool | Path:
    lowered = value.strip().lower()
    if lowered in {"true", "t", "1", "yes", "on"}:
        return True
    if lowered in {"false", "f", "0", "no", "off", "none", "null"}:
        return False
    return Path(value)


def _add_sigma_heave_argument(parser: argparse.ArgumentParser, default: object) -> None:
    parser.add_argument(
        "--sigma-heave-source",
        nargs="?",
        const=True,
        default=default,
        type=_parse_sigma_heave_source_arg,
        help=(
            "Load sigma_vert from profile metadata with no value, disable with false, "
            "or pass a NetCDF path containing sigma + latitude/longitude/time. "
            "Heave uncertainty is then derived from the active reference profile."
        ),
    )


def _resolve_path_arg(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    return Path(value).expanduser().resolve()


def _resolve_sigma_heave_source(value: object) -> bool | Path | None:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, Path):
        return value.expanduser().resolve()
    if isinstance(value, str):
        parsed = _parse_sigma_heave_source_arg(value)
        return parsed.expanduser().resolve() if isinstance(parsed, Path) else parsed
    return value  # pragma: no cover - defensive fallback


def _save_run_config(run_dir: Path, payload: dict[str, object]) -> Path:
    path = run_dir / "resolved_config.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _build_main_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OutlierDetect command line interface",
        epilog="Use `outlierdetect config --help` to manage user-editable TOML config files.",
    )
    parser.add_argument(
        "--predict",
        action="store_true",
        help="Run a trained model on raw Argo profiles",
    )
    return parser


def _build_config_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="outlierdetect config", description="Manage runtime config files")
    subparsers = parser.add_subparsers(dest="config_command", required=True)

    init_parser = subparsers.add_parser("init", help="Write a starter outlierdetect.toml")
    init_parser.add_argument(
        "--output",
        type=Path,
        default=Path("outlierdetect.toml"),
        help="Destination TOML file",
    )
    init_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output file if it already exists",
    )

    subparsers.add_parser("show", help="Print the starter TOML template")

    validate_parser = subparsers.add_parser("validate", help="Validate a TOML config file")
    validate_parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to a TOML config file",
    )
    return parser


def _split_examples(examples: list[object], val_fraction: float, seed: int) -> tuple[list[object], list[object]]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(examples))
    rng.shuffle(indices)
    n_val = int(round(len(indices) * float(val_fraction)))
    n_val = min(max(n_val, 0), max(len(indices) - 1, 0))
    val_idx = set(indices[:n_val].tolist())
    train = [ex for i, ex in enumerate(examples) if i not in val_idx]
    val = [ex for i, ex in enumerate(examples) if i in val_idx]
    return train, val


def config_main(argv: list[str] | None = None) -> None:
    """Manage user-editable runtime config files."""

    argv_list = sys.argv[1:] if argv is None else list(argv)
    parser = _build_config_parser()
    if not argv_list or any(arg in {"-h", "--help"} for arg in argv_list):
        parser.print_help()
        return

    args = parser.parse_args(argv_list)
    if args.config_command == "init":
        try:
            output = write_default_config(args.output, overwrite=args.force)
        except FileExistsError as exc:
            parser.error(str(exc))
        print(json.dumps({"status": "written", "output": str(output)}, indent=2, sort_keys=True))
        return

    if args.config_command == "show":
        sys.stdout.write(default_config_toml())
        return

    if args.config_command == "validate":
        try:
            config = load_app_config(args.config)
        except Exception as exc:
            parser.error(str(exc))
        payload = {
            "status": "valid",
            "config_path": str(Path(args.config).expanduser().resolve()),
            "config": app_config_to_dict(config),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    parser.error(f"Unknown config subcommand: {args.config_command}")


def _build_predict_parser(config: AppConfig | None = None) -> argparse.ArgumentParser:
    config = config or load_app_config(None)
    defaults = predict_parser_defaults(config)
    parser = argparse.ArgumentParser(description="Run a trained profile QC model on Argo data")
    parser.add_argument("--config", type=Path, default=None, help="TOML config file")
    parser.add_argument(
        "--checkpoint",
        required=defaults["checkpoint"] is None,
        default=defaults["checkpoint"],
        type=Path,
        help="Model checkpoint to load",
    )
    parser.add_argument(
        "--predict-root",
        default=defaults["predict_root"],
        type=Path,
        help="Directory or file containing Argo NetCDF profiles",
    )
    parser.add_argument(
        "--pattern",
        default=defaults["pattern"],
        help="Recursive glob pattern under --predict-root",
    )
    parser.add_argument("--min-levels", type=int, default=defaults["min_levels"])
    parser.add_argument(
        "--good-qc-only",
        action=argparse.BooleanOptionalAction,
        default=defaults["good_qc_only"],
        help="Apply QC masking when reading NetCDF inputs",
    )
    parser.add_argument(
        "--profile-type",
        choices=("adjusted", "raw"),
        default=defaults["profile_type"],
        help="Select adjusted values when available, or raw values only.",
    )
    parser.add_argument(
        "--raw-fallback",
        action=argparse.BooleanOptionalAction,
        default=defaults["raw_fallback"],
        help="When profile-type is adjusted, allow raw values if adjusted values are missing.",
    )
    parser.add_argument("--profile-limit", type=int, default=defaults["profile_limit"])
    parser.add_argument("--device", type=str, default=defaults["device"])
    parser.add_argument(
        "--run-root",
        type=Path,
        default=defaults["run_root"],
        help="Directory under which prediction JSON files and plots are written",
    )
    parser.add_argument("--seed", type=int, default=defaults["seed"])
    _add_input_toggle_arguments(parser, defaults)
    _add_sigma_heave_argument(parser, defaults["sigma_heave_source"])
    return parser


def _load_prediction_profiles(
    root: str | Path,
    *,
    pattern: str = "**/*.nc",
    good_qc_only: bool = False,
    min_levels: int = 5,
    profile_limit: int | None = None,
    profile_type: str = "adjusted",
    raw_fallback: bool = False,
) -> list[ArgoProfile]:
    root_path = Path(root)
    profiles = iter_argo_files(
        root_path,
        pattern=pattern,
        good_qc_only=good_qc_only,
        min_levels=min_levels,
        profile_type=profile_type,
        raw_fallback=raw_fallback,
    )
    profile_list = list(profiles)
    if profile_limit is not None:
        profile_list = profile_list[: max(int(profile_limit), 0)]
    return profile_list


def _predict_model_factory(metadata: dict[str, object]):
    from .model import Net, NetConfig

    input_dim = metadata.get("input_dim")
    if input_dim is None:
        feature_names = metadata.get("feature_names")
        if isinstance(feature_names, list):
            input_dim = len(feature_names)
    if input_dim is None:
        raise KeyError("Checkpoint metadata is missing input_dim or feature_names.")

    grid_size = int(metadata.get("grid_size", 80))
    return Net(NetConfig(input_dim=int(input_dim), grid_size=grid_size))


def predict_main(argv: list[str] | None = None) -> None:
    """Run a trained profile QC model on raw Argo profiles."""
    argv_list = sys.argv[1:] if argv is None else list(argv)
    if any(arg in {"-h", "--help"} for arg in argv_list):
        _build_predict_parser().print_help()
        return
    config = load_app_config(extract_config_path(argv_list))
    parser = _build_predict_parser(config)
    args = parser.parse_args(argv_list)
    if args.predict_root is None:
        parser.error("--predict-root is required")
    args.config = _resolve_path_arg(args.config)
    args.predict_root = _resolve_path_arg(args.predict_root)
    args.checkpoint = _resolve_path_arg(args.checkpoint)
    args.run_root = _resolve_path_arg(args.run_root)
    args.sigma_heave_source = _resolve_sigma_heave_source(args.sigma_heave_source)
    reference_source = resolve_reference_source_mode(config.train.reference)

    profiles = _load_prediction_profiles(
        args.predict_root,
        pattern=args.pattern,
        good_qc_only=args.good_qc_only,
        min_levels=args.min_levels,
        profile_limit=args.profile_limit,
        profile_type=args.profile_type,
        raw_fallback=args.raw_fallback,
    )
    if not profiles:
        raise RuntimeError("No Argo profiles were selected for prediction.")

    examples: list[ProfileExample] = []
    skipped = 0
    for profile in profiles:
        try:
            profile_input = resolve_profile_input(
                profile.to_profile_input(),
                use_residual_t=args.use_residual_t,
                use_residual_s=args.use_residual_s,
                use_sigma_t=args.use_sigma_t,
                use_sigma_s=args.use_sigma_s,
                use_sigma_vert=args.use_sigma_vert,
                use_sigma_heave_t=args.use_sigma_heave_t,
                use_sigma_heave_s=args.use_sigma_heave_s,
                use_rho_ts=args.use_rho_ts,
                use_day_of_year=args.use_day_of_year,
                sigma_heave_source=args.sigma_heave_source,
                reference_source=reference_source,
            )
        except ValueError as exc:
            skipped += 1
            warnings.warn(f"Skipping {profile.profile_id}: {exc}", stacklevel=2)
            continue
        examples.append(ProfileExample(profile=profile_input))
    if not examples:
        raise RuntimeError("No valid profiles remained after cleaning.")

    model, metadata = load_model_from_checkpoint(
        args.checkpoint,
        _predict_model_factory,
        map_location=args.device,
    )
    normalization = metadata.get("normalization") or metadata.get("norm_stats")
    predictor = Neural(
        model,
        config=Config(reconstruction_grid_size=int(metadata.get("grid_size", 80))),
        device=args.device,
        normalization=normalization,
    )

    writer = TrainingRunWriter(
        args.run_root,
        examples,
        norm=normalization,
        plot_count=len(examples),
        seed=args.seed,
    )
    print(f"Writing prediction artifacts to {writer.run_dir}")
    _save_run_config(writer.run_dir, resolved_run_config_dict(config=config, command="predict", args=args))
    progress = writer.record_predictions(predictor=predictor)
    writer.finalize(history=[], checkpoint_path=args.checkpoint)

    payload = {
        "run_dir": str(writer.run_dir),
        "progress_file": str(writer.progress_path),
        "checkpoint": str(Path(args.checkpoint)),
        "n_profiles": len(profiles),
        "n_selected": len(examples),
        "n_skipped": skipped,
        "n_predicted": int(progress.get("n_predicted", len(examples))),
        "n_failed": int(progress.get("n_failed", 0)),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def _build_train_parser(config: AppConfig | None = None) -> argparse.ArgumentParser:
    config = config or load_app_config(None)
    defaults = train_parser_defaults(config)
    parser = argparse.ArgumentParser(description="Train the profile QC model")
    parser.add_argument("--config", type=Path, default=None, help="TOML config file")
    parser.add_argument(
        "--train-root",
        default=defaults["train_root"],
        type=Path,
        help="Directory or file with Argo or EN4 profiles",
    )
    parser.add_argument(
        "--argo-root",
        dest="train_root",
        type=Path,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--test-root",
        type=Path,
        default=defaults["test_root"],
        help=(
            "Optional held-out directory or file with Argo or EN4 profiles. "
            "When set, the held-out side follows the configured profile source and can be augmented. "
            "If omitted, the training root is split into train and validation subsets."
        ),
    )
    parser.add_argument(
        "--test-augment",
        action=argparse.BooleanOptionalAction,
        default=defaults["test_augment"],
        help=(
            "Apply the synthetic corruption pipeline to --test-root and use adjusted/corrected values. "
            "If disabled, the test root is loaded raw and fed to the model without synthetic augmentation."
        ),
    )
    parser.add_argument(
        "--data-source",
        choices=("argo", "en4"),
        default=defaults["data_source"],
        help="Select the clean-profile source used to build synthetic training examples",
    )
    parser.add_argument(
        "--training-mode",
        choices=("preload", "stream"),
        default=defaults["training_mode"],
        help=(
            "Choose whether to materialize all examples before training or to open source files lazily "
            "inside the dataset/dataloader."
        ),
    )
    parser.add_argument(
        "--profile-type",
        choices=("adjusted", "raw"),
        default=defaults["profile_type"],
        help="Select adjusted values when available, or raw values only.",
    )
    parser.add_argument(
        "--raw-fallback",
        action=argparse.BooleanOptionalAction,
        default=defaults["raw_fallback"],
        help="When profile-type is adjusted, allow raw values if adjusted values are missing.",
    )
    parser.add_argument("--output", type=Path, default=defaults["output"], help="Optional checkpoint output path")
    parser.add_argument("--seed", type=int, default=defaults["seed"])
    parser.add_argument("--profile-limit", type=int, default=defaults["profile_limit"])
    parser.add_argument("--n-examples-per-profile", type=int, default=defaults["n_examples_per_profile"])
    parser.add_argument("--n-levels", type=int, default=defaults["n_levels"])
    parser.add_argument("--min-levels", type=int, default=defaults["min_levels"])
    parser.add_argument("--grid-size", type=int, default=defaults["grid_size"])
    parser.add_argument("--upper-ocean-bias", type=float, default=defaults["upper_ocean_bias"])
    parser.add_argument("--epochs", type=int, default=defaults["epochs"])
    parser.add_argument("--batch-size", type=int, default=defaults["batch_size"])
    parser.add_argument("--learning-rate", type=float, default=defaults["learning_rate"])
    parser.add_argument("--val-fraction", type=float, default=defaults["val_fraction"])
    parser.add_argument("--device", type=str, default=defaults["device"])
    parser.add_argument(
        "--run-root",
        type=Path,
        default=defaults["run_root"],
        help="Directory under which per-run progress JSON and plots are written",
    )
    parser.add_argument(
        "--epoch-plot-count",
        type=int,
        default=defaults["epoch_plot_count"],
        help="Number of random reconstruction plots to save after each epoch",
    )
    parser.add_argument(
        "--good-qc-only",
        action=argparse.BooleanOptionalAction,
        default=defaults["good_qc_only"],
        help="Apply QC masking when reading NetCDF inputs",
    )
    _add_input_toggle_arguments(parser, defaults)
    _add_sigma_heave_argument(parser, defaults["sigma_heave_source"])
    return parser


def train_main(argv: list[str] | None = None) -> None:
    """Train the profile QC model on synthetic examples."""
    argv_list = sys.argv[1:] if argv is None else list(argv)
    if any(arg in {"-h", "--help"} for arg in argv_list):
        _build_train_parser().print_help()
        return
    config = load_app_config(extract_config_path(argv_list))
    parser = _build_train_parser(config)
    args = parser.parse_args(argv_list)
    if args.train_root is None:
        parser.error("--train-root is required")
    args.config = _resolve_path_arg(args.config)
    args.train_root = _resolve_path_arg(args.train_root)
    args.test_root = _resolve_path_arg(args.test_root)
    args.output = _resolve_path_arg(args.output)
    args.run_root = _resolve_path_arg(args.run_root)
    args.sigma_heave_source = _resolve_sigma_heave_source(args.sigma_heave_source)
    reference_source = resolve_reference_source_mode(config.train.reference)

    try:
        from torch.utils.data import DataLoader

        from .model import Net, NetConfig
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("Training requires PyTorch. Install the train extra.") from exc

    builder = {
        "argo": build_argo_synthetic_examples,
        "en4": build_en4_synthetic_examples,
    }[args.data_source]

    train_profile_source_kwargs = {
        "profile_type": args.profile_type,
        "raw_fallback": args.raw_fallback,
    }
    test_profile_source_kwargs = {
        "profile_type": "raw",
        "raw_fallback": False,
    }

    def _materialize_examples(
        root: str | Path,
        *,
        role: str,
        augment: bool = True,
        profile_source_kwargs: dict[str, object] | None = None,
    ) -> list[ProfileExample]:
        profile_source_kwargs = {} if profile_source_kwargs is None else dict(profile_source_kwargs)
        if not augment:
            root_path = Path(root)
            if args.data_source == "argo":
                profiles = list(
                    iter_argo_files(
                        root_path,
                        good_qc_only=False,
                        min_levels=args.min_levels,
                        **profile_source_kwargs,
                    )
                )
            else:
                profiles = list(
                    iter_en4_files(
                        root_path,
                        good_qc_only=False,
                        min_levels=args.min_levels,
                        **profile_source_kwargs,
                    )
                )
            if args.profile_limit is not None:
                profiles = profiles[: max(int(args.profile_limit), 0)]
            if not profiles:
                raise RuntimeError(
                    f"No {args.data_source.upper()} profiles were converted into {role} examples."
                )
            return [
                ProfileExample(
                    profile=resolve_profile_input(
                        profile.to_profile_input(),
                        use_residual_t=args.use_residual_t,
                        use_residual_s=args.use_residual_s,
                        use_sigma_t=args.use_sigma_t,
                        use_sigma_s=args.use_sigma_s,
                        use_sigma_vert=args.use_sigma_vert,
                        use_sigma_heave_t=args.use_sigma_heave_t,
                        use_sigma_heave_s=args.use_sigma_heave_s,
                        use_rho_ts=args.use_rho_ts,
                        use_day_of_year=args.use_day_of_year,
                        sigma_heave_source=args.sigma_heave_source,
                        reference_source=reference_source,
                    )
                )
                for profile in profiles
            ]

        synthetic = builder(
            root,
            n_examples_per_profile=args.n_examples_per_profile,
            n_levels=args.n_levels,
            grid_size=args.grid_size,
            profile_limit=args.profile_limit,
            min_levels=args.min_levels,
            good_qc_only=args.good_qc_only,
            seed=args.seed,
            upper_ocean_bias=args.upper_ocean_bias,
            use_raw_values=False,
            reference_source=reference_source,
            **profile_source_kwargs,
        )
        if not synthetic:
            raise RuntimeError(
                f"No {args.data_source.upper()} profiles were converted into {role} examples."
            )
        return [
            ProfileExample(
                profile=resolve_profile_input(
                    item.example.profile,
                    use_residual_t=args.use_residual_t,
                    use_residual_s=args.use_residual_s,
                    use_sigma_t=args.use_sigma_t,
                    use_sigma_s=args.use_sigma_s,
                    use_sigma_vert=args.use_sigma_vert,
                    use_sigma_heave_t=args.use_sigma_heave_t,
                    use_sigma_heave_s=args.use_sigma_heave_s,
                    use_rho_ts=args.use_rho_ts,
                    use_day_of_year=args.use_day_of_year,
                    sigma_heave_source=args.sigma_heave_source,
                    reference_source=reference_source,
                ),
                labels=item.example.labels,
            )
            for item in synthetic
        ]

    def _make_stream_factory(
        root: str | Path,
        *,
        augment: bool,
        source_files: list[Path] | None = None,
    ):
        if augment:
            profile_type = args.profile_type
            raw_fallback = args.raw_fallback
            good_qc_only = args.good_qc_only
        else:
            profile_type = "raw"
            raw_fallback = False
            good_qc_only = False
        return make_training_example_factory(
            root,
            data_source=args.data_source,
            augment=augment,
            source_files=source_files,
            profile_type=profile_type,
            raw_fallback=raw_fallback,
            good_qc_only=good_qc_only,
            min_levels=args.min_levels,
            profile_limit=args.profile_limit,
            n_examples_per_profile=args.n_examples_per_profile,
            n_levels=args.n_levels,
            grid_size=args.grid_size,
            seed=args.seed,
            upper_ocean_bias=args.upper_ocean_bias,
            use_raw_values=False,
            reference_source=reference_source,
            use_residual_t=args.use_residual_t,
            use_residual_s=args.use_residual_s,
            use_sigma_t=args.use_sigma_t,
            use_sigma_s=args.use_sigma_s,
            use_sigma_vert=args.use_sigma_vert,
            use_sigma_heave_t=args.use_sigma_heave_t,
            use_sigma_heave_s=args.use_sigma_heave_s,
            use_rho_ts=args.use_rho_ts,
            use_day_of_year=args.use_day_of_year,
            sigma_heave_source=args.sigma_heave_source,
            source_name=args.data_source,
        )

    training_mode = str(args.training_mode).strip().lower()
    preview_limit = max(int(args.epoch_plot_count), 1)
    norm: object | None = None
    total_examples: int | None = None

    if training_mode == "stream":
        train_source_files = list_source_files(args.train_root)
        if not train_source_files:
            raise RuntimeError(f"No {args.data_source.upper()} files were found under {args.train_root}.")
        train_source_files, _ = split_source_files(train_source_files, 0.0, args.seed)

        if args.test_root is None:
            train_source_files, eval_source_files = split_source_files(train_source_files, args.val_fraction, args.seed)
            eval_root = args.train_root
            eval_augment = True
            eval_label = "val"
        else:
            eval_root = args.test_root
            eval_source_files = list_source_files(args.test_root)
            eval_augment = bool(args.test_augment)
            eval_label = "test"

        train_factory = _make_stream_factory(args.train_root, augment=True, source_files=train_source_files)
        train_preview = preview_stream_examples(train_factory, limit=preview_limit)
        if not train_preview:
            raise RuntimeError(f"No {args.data_source.upper()} profiles were converted into training examples.")

        eval_factory = None
        eval_preview: list[ProfileExample] = []
        if eval_source_files:
            eval_factory = _make_stream_factory(eval_root, augment=eval_augment, source_files=eval_source_files)
            eval_preview = preview_stream_examples(eval_factory, limit=preview_limit)
            if not eval_preview and args.test_root is not None:
                raise RuntimeError(
                    f"No {args.data_source.upper()} profiles were converted into held-out evaluation examples."
                )

        train_dataset = StreamingProfileDataset(train_factory, norm=None)
        val_dataset = StreamingProfileDataset(eval_factory, norm=None) if eval_factory is not None else None
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_profiles)
        val_loader = (
            DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_profiles)
            if val_dataset is not None
            else None
        )
        monitor_examples = eval_preview if eval_preview else train_preview
        writer = TrainingRunWriter(
            args.run_root,
            monitor_examples,
            norm=None,
            plot_count=max(int(args.epoch_plot_count), 0),
            seed=args.seed,
        )
        print(f"Training mode: {training_mode} (lazy file loading)")
        print(f"Writing training artifacts to {writer.run_dir}")
        _save_run_config(writer.run_dir, resolved_run_config_dict(config=config, command="train", args=args))

        sample = ProfileDataset([train_preview[0]], norm=None)[0]
        model = Net(NetConfig(input_dim=sample["features"].shape[1], grid_size=args.grid_size))
        normalization_accumulator = NormalizationAccumulator()
        history = fit_model(
            model,
            train_loader,
            val_loader,
            eval_label=eval_label,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            device=args.device,
            progress=True,
            normalization_accumulator=normalization_accumulator,
            normalization_callback=writer.set_normalization,
            epoch_callback=lambda epoch, model, history: writer.record_epoch(
                epoch=epoch,
                model=model,
                history=history,
                device=args.device,
                n_train_examples=None,
                n_val_examples=None,
                eval_label=eval_label,
            ),
        )

        if args.output is not None:
            save_checkpoint(
                args.output,
                model,
                metadata={
                    "input_dim": int(sample["features"].shape[1]),
                    "grid_size": int(args.grid_size),
                    "min_levels": int(args.min_levels),
                    "n_examples": total_examples,
                    "seed": int(args.seed),
                    "upper_ocean_bias": float(args.upper_ocean_bias),
                    "feature_names": list(sample["feature_names"]),
                    "normalization": None if normalization_accumulator is None else normalization_accumulator.to_stats().as_dict(),
                    "config": resolved_run_config_dict(config=config, command="train", args=args),
                },
            )
        writer.finalize(history=history, checkpoint_path=args.output)

        summary = {
            "run_dir": str(writer.run_dir),
            "progress_file": str(writer.progress_path),
            "training_mode": training_mode,
            "n_examples": total_examples,
            "n_train": None,
            "n_eval_examples": None,
            "eval_label": eval_label,
            "history": history,
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    train_examples = _materialize_examples(
        args.train_root,
        role="training",
        profile_source_kwargs=train_profile_source_kwargs,
    )
    if args.test_root is None:
        train_examples, eval_examples = _split_examples(train_examples, args.val_fraction, args.seed)
        eval_label = "val"
        val_loader_enabled = True
    else:
        eval_examples = _materialize_examples(
            args.test_root,
            role="held-out evaluation",
            augment=bool(args.test_augment),
            profile_source_kwargs=test_profile_source_kwargs,
        )
        eval_label = "test"
        val_loader_enabled = bool(args.test_augment)

    total_examples = len(train_examples) + len(eval_examples)
    norm = compute_normalization_stats(train_examples)
    train_dataset = ProfileDataset(train_examples, norm=norm)
    val_dataset = ProfileDataset(eval_examples, norm=norm) if (eval_examples and val_loader_enabled) else None
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_profiles)
    val_loader = (
        DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_profiles)
        if val_dataset is not None
        else None
    )
    monitor_examples = eval_examples if eval_examples else train_examples
    writer = TrainingRunWriter(
        args.run_root,
        monitor_examples,
        norm=norm,
        plot_count=max(int(args.epoch_plot_count), 0),
        seed=args.seed,
    )
    print(f"Training mode: {training_mode} (preload into RAM)")
    print(f"Writing training artifacts to {writer.run_dir}")
    _save_run_config(writer.run_dir, resolved_run_config_dict(config=config, command="train", args=args))

    sample = train_dataset[0]
    model = Net(NetConfig(input_dim=sample["features"].shape[1], grid_size=args.grid_size))
    history = fit_model(
        model,
        train_loader,
        val_loader,
        eval_label=eval_label,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=args.device,
        progress=True,
        epoch_callback=lambda epoch, model, history: writer.record_epoch(
            epoch=epoch,
            model=model,
            history=history,
            device=args.device,
            n_train_examples=len(train_examples),
            n_val_examples=len(eval_examples),
            eval_label=eval_label,
        ),
    )

    if args.output is not None:
        save_checkpoint(
            args.output,
            model,
            metadata={
                "input_dim": int(sample["features"].shape[1]),
                "grid_size": int(args.grid_size),
                "min_levels": int(args.min_levels),
                "n_examples": total_examples,
                "seed": int(args.seed),
                "upper_ocean_bias": float(args.upper_ocean_bias),
                "feature_names": list(sample["feature_names"]),
                "normalization": norm.as_dict(),
                "config": resolved_run_config_dict(config=config, command="train", args=args),
            },
        )
    writer.finalize(history=history, checkpoint_path=args.output)

    summary = {
        "run_dir": str(writer.run_dir),
        "progress_file": str(writer.progress_path),
        "training_mode": training_mode,
        "n_examples": total_examples,
        "n_train": len(train_examples),
        "n_eval_examples": len(eval_examples),
        "eval_label": eval_label,
        "history": history,
    }
    summary[f"n_{eval_label}"] = len(eval_examples)
    print(json.dumps(summary, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> None:
    """Dispatch the top-level OutlierDetect CLI."""
    argv_list = sys.argv[1:] if argv is None else list(argv)
    if argv_list and argv_list[0] == "config":
        config_main(argv_list[1:])
        return
    main_parser = _build_main_parser()

    if any(arg in {"-h", "--help"} for arg in argv_list):
        if "--predict" in argv_list:
            _build_predict_parser().print_help()
        else:
            main_parser.print_help()
        return

    args, remaining = main_parser.parse_known_args(argv_list)

    if args.predict:
        predict_main(remaining)
        return

    main_parser.error("Specify --predict or the config subcommand.")


if __name__ == "__main__":
    main()
