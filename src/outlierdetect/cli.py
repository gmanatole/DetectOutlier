"""Small command-line entry points."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import warnings

import numpy as np

from .argo import ArgoProfile, iter_argo_files
from .parquet import iter_argo_parquet_profiles
from .tool import Config, Neural
from .training.argo import build_argo_synthetic_examples
from .training.artifacts import TrainingRunWriter
from .training.dataset import ProfileDataset, ProfileExample, collate_profiles, compute_normalization_stats
from .training.train import fit_model, load_model_from_checkpoint, save_checkpoint


def _add_raw_to_parquet_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True, help="Directory or file containing Argo NetCDF profiles")
    parser.add_argument("--output", required=True, type=Path, help="Output parquet file")
    parser.add_argument("--pattern", default="**/*.nc", help="Recursive glob pattern under --input")
    parser.add_argument("--min-levels", type=int, default=5)
    parser.add_argument(
        "--good-qc-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep only Argo levels with good QC flags",
    )


def _write_raw_to_parquet(args: argparse.Namespace) -> None:
    from .parquet import write_argo_parquet

    summary = write_argo_parquet(
        args.input,
        args.output,
        pattern=args.pattern,
        good_qc_only=args.good_qc_only,
        min_levels=args.min_levels,
    )
    print(json.dumps(summary.as_dict(), indent=2, sort_keys=True))


def _build_main_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OutlierDetect command line interface")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--raw-to-parquet",
        action="store_true",
        help="Export raw Argo NetCDF profiles to parquet",
    )
    group.add_argument(
        "--predict",
        action="store_true",
        help="Run a trained model on raw Argo or parquet-backed profiles",
    )
    return parser


def _build_raw_to_parquet_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export Argo NetCDF profiles to parquet")
    _add_raw_to_parquet_arguments(parser)
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


def _default_train_run_root() -> Path:
    return Path(__file__).resolve().parent / "train" / "data" / "run"


def _default_predict_run_root() -> Path:
    return Path("artifacts") / "predict_runs"


def _build_predict_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a trained profile QC model on Argo data")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Model checkpoint to load")
    parser.add_argument(
        "--argo-root",
        required=True,
        help="Directory or file containing Argo NetCDF profiles, or a parquet dataset exported by outlier-detect",
    )
    parser.add_argument("--pattern", default="**/*.nc", help="Recursive glob pattern under --argo-root")
    parser.add_argument("--min-levels", type=int, default=5)
    parser.add_argument(
        "--good-qc-only",
        action="store_true",
        default=False,
        help="Apply Argo QC masking when reading NetCDF inputs (off by default)",
    )
    parser.add_argument("--profile-limit", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--run-root",
        type=Path,
        default=_default_predict_run_root(),
        help="Directory under which prediction JSON files and plots are written",
    )
    parser.add_argument("--seed", type=int, default=4)
    return parser


def _load_prediction_profiles(
    root: str | Path,
    *,
    pattern: str = "**/*.nc",
    good_qc_only: bool = False,
    min_levels: int = 5,
    profile_limit: int | None = None,
) -> list[ArgoProfile]:
    root_path = Path(root)
    if root_path.suffix.lower() in {".parquet", ".pq"}:
        profiles = iter_argo_parquet_profiles(root_path, min_levels=min_levels)
    else:
        profiles = iter_argo_files(
            root_path,
            pattern=pattern,
            good_qc_only=good_qc_only,
            min_levels=min_levels,
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
    """Run a trained profile QC model on raw Argo or parquet-backed profiles."""
    parser = _build_predict_parser()
    args = parser.parse_args(argv)

    profiles = _load_prediction_profiles(
        args.argo_root,
        pattern=args.pattern,
        good_qc_only=args.good_qc_only,
        min_levels=args.min_levels,
        profile_limit=args.profile_limit,
    )
    if not profiles:
        raise RuntimeError("No Argo profiles were selected for prediction.")

    examples: list[ProfileExample] = []
    skipped = 0
    for profile in profiles:
        try:
            profile_input = profile.to_profile_input()
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

def train_main(argv: list[str] | None = None) -> None:
    """Train the profile QC model on Argo-backed synthetic examples."""
    parser = argparse.ArgumentParser(description="Train the profile QC model")
    parser.add_argument(
        "--argo-root",
        required=True,
        help="Directory or file with Argo profiles, or a parquet dataset exported by outlier-detect",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional checkpoint output path")
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--profile-limit", type=int, default=None)
    parser.add_argument("--n-examples-per-profile", type=int, default=1)
    parser.add_argument("--n-levels", type=int, default=20)
    parser.add_argument("--min-levels", type=int, default=5)
    parser.add_argument("--grid-size", type=int, default=80)
    parser.add_argument("--upper-ocean-bias", type=float, default=1.7)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--run-root",
        type=Path,
        default=_default_train_run_root(),
        help="Directory under which per-run progress JSON and plots are written",
    )
    parser.add_argument(
        "--epoch-plot-count",
        type=int,
        default=10,
        help="Number of random reconstruction plots to save after each epoch",
    )
    args = parser.parse_args(argv)

    try:
        from torch.utils.data import DataLoader

        from .model import Net, NetConfig
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("Training requires PyTorch. Install the train extra.") from exc

    synthetic = build_argo_synthetic_examples(
        args.argo_root,
        n_examples_per_profile=args.n_examples_per_profile,
        n_levels=args.n_levels,
        grid_size=args.grid_size,
        profile_limit=args.profile_limit,
        min_levels=args.min_levels,
        seed=args.seed,
        upper_ocean_bias=args.upper_ocean_bias,
    )
    if not synthetic:
        raise RuntimeError("No Argo profiles were converted into training examples.")

    examples = [item.example for item in synthetic]
    train_examples, val_examples = _split_examples(examples, args.val_fraction, args.seed)
    norm = compute_normalization_stats(train_examples)
    train_dataset = ProfileDataset(train_examples, norm=norm)
    val_dataset = ProfileDataset(val_examples, norm=norm) if val_examples else None
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_profiles)
    val_loader = (
        DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_profiles)
        if val_dataset is not None
        else None
    )
    monitor_examples = val_examples if val_examples else train_examples
    writer = TrainingRunWriter(
        args.run_root,
        monitor_examples,
        norm=norm,
        plot_count=max(int(args.epoch_plot_count), 0),
        seed=args.seed,
    )
    print(f"Writing training artifacts to {writer.run_dir}")

    sample = train_dataset[0]
    model = Net(NetConfig(input_dim=sample["features"].shape[1], grid_size=args.grid_size))
    history = fit_model(
        model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=args.device,
        epoch_callback=lambda epoch, model, history: writer.record_epoch(
            epoch=epoch,
            model=model,
            history=history,
            device=args.device,
            n_train_examples=len(train_examples),
            n_val_examples=len(val_examples),
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
                "n_examples": len(examples),
                "seed": int(args.seed),
                "upper_ocean_bias": float(args.upper_ocean_bias),
                "feature_names": list(sample["feature_names"]),
                "normalization": norm.as_dict(),
            },
        )
    writer.finalize(history=history, checkpoint_path=args.output)

    summary = {
        "run_dir": str(writer.run_dir),
        "progress_file": str(writer.progress_path),
        "n_examples": len(examples),
        "n_train": len(train_examples),
        "n_val": len(val_examples),
        "history": history,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


def raw_to_parquet_main(argv: list[str] | None = None) -> None:
    """Export all Argo NetCDF profiles under a directory to parquet."""
    parser = _build_raw_to_parquet_parser()
    args = parser.parse_args(argv)
    try:
        _write_raw_to_parquet(args)
    except (IsADirectoryError, ValueError) as exc:
        parser.error(str(exc))


def main(argv: list[str] | None = None) -> None:
    """Dispatch the top-level OutlierDetect CLI."""
    argv_list = sys.argv[1:] if argv is None else list(argv)
    main_parser = _build_main_parser()

    if any(arg in {"-h", "--help"} for arg in argv_list):
        if "--raw-to-parquet" in argv_list:
            _build_raw_to_parquet_parser().print_help()
        elif "--predict" in argv_list:
            _build_predict_parser().print_help()
        else:
            main_parser.print_help()
        return

    args, remaining = main_parser.parse_known_args(argv_list)

    if args.raw_to_parquet:
        raw_to_parquet_main(remaining)
        return

    if args.predict:
        predict_main(remaining)
        return

    main_parser.error("Specify either --raw-to-parquet or --predict.")


def argo_parquet_main(argv: list[str] | None = None) -> None:
    """Backward-compatible alias for the parquet export CLI."""
    raw_to_parquet_main(argv)


if __name__ == "__main__":
    main()
