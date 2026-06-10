"""Helpers for writing per-epoch training artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import warnings
from typing import Any, Sequence
import uuid

import numpy as np

from outlierdetect.data import NormalizationStats, probability_payload
from outlierdetect.density import sigma0_from_ts

from .dataset import ProfileDataset, ProfileExample, collate_profiles

try:
    import torch
except Exception:  # pragma: no cover - torch is optional
    torch = None  # type: ignore[assignment]


@dataclass(slots=True)
class TrainingRunWriter:
    """Persist progress JSON and reconstruction plots during training."""

    run_root: str | Path
    examples: Sequence[ProfileExample]
    norm: NormalizationStats | dict[str, float] | None = None
    plot_count: int = 10
    seed: int = 4
    dpi: int = 160
    run_id: str = field(init=False)
    run_dir: Path = field(init=False)
    progress_path: Path = field(init=False)
    plots_dir: Path = field(init=False)
    _progress: dict[str, Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.run_root = Path(self.run_root)
        self.norm = NormalizationStats.from_mapping(self.norm)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        self.run_id = f"{stamp}_{os.getpid()}_{uuid.uuid4().hex[:8]}"
        self.run_dir = self.run_root / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.plots_dir = self.run_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.progress_path = self.run_dir / "progress.json"
        self._progress = {
            "run_id": self.run_id,
            "run_root": str(self.run_root),
            "run_dir": str(self.run_dir),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "running",
            "epochs": [],
            "plot_count": int(self.plot_count),
            "seed": int(self.seed),
            "dpi": int(self.dpi),
            "norm_stats": None if self.norm is None else self.norm.as_dict(),
        }
        self._write_progress()

    def record_epoch(
        self,
        *,
        epoch: int,
        model: Any,
        history: list[dict[str, float]],
        device: str | Any = "cpu",
        n_train_examples: int | None = None,
        n_val_examples: int | None = None,
    ) -> dict[str, Any]:
        """Write progress JSON and 10 sampled reconstruction plots for one epoch."""
        if torch is None:  # pragma: no cover - guarded by training dependency
            raise ImportError("Writing training plots requires PyTorch.")

        epoch_dir = self.plots_dir / f"epoch_{epoch:03d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)

        selected = self._select_indices(epoch)
        was_training = bool(getattr(model, "training", False))
        model.eval()

        plot_files: list[str] = []
        selected_profile_ids: list[str] = []
        selected_profiles: list[dict[str, Any]] = []
        try:
            for rank, example_index in enumerate(selected, start=1):
                example = self.examples[example_index]
                sample = ProfileDataset([example], norm=self.norm)[0]
                batch = collate_profiles([sample])
                features = batch["features"].to(device)
                mask = batch["mask"].to(device)
                pressure_grid = batch.get("pressure_grid")
                if pressure_grid is not None:
                    pressure_grid = pressure_grid.to(device)
                with torch.no_grad():
                    outputs = model(features, mask=mask, recon_pressure=pressure_grid)
                plot_name = self._plot_name(rank, example.profile.profile_id, example_index)
                plot_path = epoch_dir / plot_name
                output_profile_id = example.profile.profile_id or f"profile_{example_index:04d}"
                point_probs = torch.sigmoid(outputs["point_logits"][0]).detach().cpu().numpy()
                if point_probs.shape[1] < 2:
                    raise ValueError("point_logits must have at least two channels for T/S probabilities.")
                profile_prob = float(torch.sigmoid(outputs["profile_logit"]).reshape(-1)[0].detach().cpu().numpy())
                point_bad_t = point_probs[:, 0]
                point_bad_s = point_probs[:, 1]
                point_density = point_probs[:, 2] if point_probs.shape[1] > 2 else None
                recon_temperature, recon_salinity = self._denormalize_reconstruction(
                    outputs["recon_mean"][0].detach().cpu().numpy()
                )
                self._save_reconstruction_plot(
                    plot_path,
                    profile_id=output_profile_id,
                    pressure=example.profile.pressure,
                    temperature=example.profile.temperature,
                    salinity=example.profile.salinity,
                    recon_temperature=recon_temperature,
                    recon_salinity=recon_salinity,
                    truth_temperature=self._extract_truth(sample, "truth_t"),
                    truth_salinity=self._extract_truth(sample, "truth_s"),
                    epoch=epoch,
                    rank=rank,
                )
                plot_rel = str(plot_path.relative_to(self.run_dir))
                prediction_path = plot_path.with_suffix(".json")
                prediction_rel = str(prediction_path.relative_to(self.run_dir))
                prediction_payload = probability_payload(
                    profile_id=output_profile_id,
                    profile_bad_probability=profile_prob,
                    point_bad_t=point_bad_t,
                    point_bad_s=point_bad_s,
                    point_density_inconsistent=point_density,
                    plot_file=plot_rel,
                    epoch=epoch,
                    rank=rank,
                )
                self._write_json(prediction_path, prediction_payload)
                plot_files.append(plot_rel)
                selected_profile_ids.append(output_profile_id)
                selected_profiles.append(
                    {
                        "profile_id": output_profile_id,
                        "plot_file": plot_rel,
                        "prediction_file": prediction_rel,
                        "profile_bad_probability": profile_prob,
                    }
                )
        finally:
            if was_training:
                model.train()

        epochs = _group_history(history)
        current = epochs[-1] if epochs else {}
        self._progress.update(
            {
                "status": "running",
                "updated_utc": datetime.now(timezone.utc).isoformat(),
                "current_epoch": int(epoch),
                "epochs": epochs,
                "latest": current,
                "plot_files": plot_files,
                "selected_profile_ids": selected_profile_ids,
                "selected_profiles": selected_profiles,
            }
        )
        if n_train_examples is not None:
            self._progress["n_train_examples"] = int(n_train_examples)
        if n_val_examples is not None:
            self._progress["n_val_examples"] = int(n_val_examples)

        self._write_progress()
        return dict(self._progress)

    def finalize(
        self,
        *,
        history: list[dict[str, float]],
        checkpoint_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Mark the run as complete and persist the final progress snapshot."""
        self._progress.update(
            {
                "status": "complete",
                "finished_utc": datetime.now(timezone.utc).isoformat(),
                "epochs": _group_history(history),
            }
        )
        if checkpoint_path is not None:
            self._progress["checkpoint_path"] = str(Path(checkpoint_path))
        self._write_progress()
        return dict(self._progress)

    def record_predictions(
        self,
        *,
        predictor: Any,
    ) -> dict[str, Any]:
        """Write prediction JSON sidecars and plots for each profile."""
        if torch is None:  # pragma: no cover - guarded by prediction dependency
            raise ImportError("Writing prediction artifacts requires PyTorch.")

        prediction_dir = self.run_dir / "predictions"
        prediction_dir.mkdir(parents=True, exist_ok=True)
        prediction_plot_dir = self.plots_dir / "predict"
        prediction_plot_dir.mkdir(parents=True, exist_ok=True)

        plot_files: list[str] = []
        prediction_files: list[str] = []
        selected_profile_ids: list[str] = []
        selected_profiles: list[dict[str, Any]] = []
        n_failed = 0

        model = getattr(predictor, "model", None)
        was_training = bool(getattr(model, "training", False))
        if model is not None and hasattr(model, "eval"):
            model.eval()

        try:
            for index, example in enumerate(self.examples):
                profile = example.profile
                profile_id = profile.profile_id or f"profile_{index:04d}"
                rank = index + 1
                base_name = self._plot_name(rank, profile_id, index)
                prediction_path = prediction_dir / base_name.replace(".png", ".json")
                plot_path = prediction_plot_dir / base_name
                try:
                    result = predictor.predict(profile)
                except Exception as exc:
                    n_failed += 1
                    warnings.warn(f"Skipping {profile_id}: {exc}", stacklevel=2)
                    continue

                payload = result.probability_dict()
                if profile.attrs:
                    payload["attrs"] = dict(profile.attrs)
                self._write_json(prediction_path, payload)
                prediction_file_rel = str(prediction_path.relative_to(self.run_dir))
                prediction_files.append(prediction_file_rel)

                summary = result.summary()
                summary.update(
                    {
                        "prediction_file": prediction_file_rel,
                        "profile_attrs": dict(profile.attrs),
                    }
                )
                if result.temperature_reconstructed is None or result.salinity_reconstructed is None:
                    raise ValueError("predict() must return reconstructed temperature and salinity to save plots.")
                self._save_reconstruction_plot(
                    plot_path,
                    profile_id=profile_id,
                    pressure=profile.pressure,
                    temperature=profile.temperature,
                    salinity=profile.salinity,
                    recon_temperature=result.temperature_reconstructed,
                    recon_salinity=result.salinity_reconstructed,
                    truth_temperature=None,
                    truth_salinity=None,
                    epoch=0,
                    rank=rank,
                )
                plot_file_rel = str(plot_path.relative_to(self.run_dir))
                plot_files.append(plot_file_rel)
                summary["plot_file"] = plot_file_rel

                selected_profile_ids.append(profile_id)
                selected_profiles.append(summary)
        finally:
            if was_training and model is not None and hasattr(model, "train"):
                model.train()

        self._progress.update(
            {
                "status": "running",
                "updated_utc": datetime.now(timezone.utc).isoformat(),
                "mode": "predict",
                "n_profiles": len(self.examples),
                "n_predicted": len(selected_profiles),
                "n_failed": n_failed,
                "plot_count": len(plot_files),
                "plot_files": plot_files,
                "prediction_files": prediction_files,
                "selected_profile_ids": selected_profile_ids,
                "selected_profiles": selected_profiles,
            }
        )
        self._write_progress()
        return dict(self._progress)

    def _select_indices(self, epoch: int) -> list[int]:
        n_examples = len(self.examples)
        if n_examples == 0:
            return []
        count = min(int(self.plot_count), n_examples)
        rng = np.random.default_rng(int(self.seed) + int(epoch))
        return sorted(rng.choice(n_examples, size=count, replace=False).tolist())

    def _plot_name(self, rank: int, profile_id: str | None, example_index: int) -> str:
        base = profile_id or f"profile_{example_index:04d}"
        base = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._-") or "profile"
        return f"{rank:02d}_{base}.png"

    def _save_reconstruction_plot(
        self,
        path: Path,
        *,
        profile_id: str | None,
        pressure: np.ndarray,
        temperature: np.ndarray,
        salinity: np.ndarray,
        recon_temperature: np.ndarray,
        recon_salinity: np.ndarray,
        truth_temperature: np.ndarray | None,
        truth_salinity: np.ndarray | None,
        epoch: int,
        rank: int,
    ) -> None:
        try:
            import matplotlib.pyplot as plt
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Saving epoch reconstruction plots requires matplotlib. "
                "Install with: pip install -e '.[train]'"
            ) from exc

        s_arrays = [salinity, recon_salinity]
        t_arrays = [temperature, recon_temperature]
        if truth_temperature is not None and truth_salinity is not None:
            s_arrays.append(truth_salinity)
            t_arrays.append(truth_temperature)

        s_min, s_max = _finite_limits(*s_arrays)
        t_min, t_max = _finite_limits(*t_arrays)
        s_pad = max((s_max - s_min) * 0.1, 0.05)
        t_pad = max((t_max - t_min) * 0.1, 0.05)
        s_grid = np.linspace(s_min - s_pad, s_max + s_pad, 120)
        t_grid = np.linspace(t_min - t_pad, t_max + t_pad, 120)
        s_mesh, t_mesh = np.meshgrid(s_grid, t_grid)
        rho_mesh = sigma0_from_ts(s_mesh, t_mesh)
        levels = np.linspace(float(np.nanmin(rho_mesh)), float(np.nanmax(rho_mesh)), 10)

        fig, ax = plt.subplots(figsize=(5.8, 5.2), constrained_layout=True)
        contours = ax.contour(s_mesh, t_mesh, rho_mesh, levels=levels, colors="0.84", linewidths=0.8)
        ax.clabel(contours, inline=True, fontsize=7, fmt="%.1f")
        ax.plot(salinity, temperature, color="#1f77b4", marker="o", ms=3.5, lw=1.4, label="input")
        if truth_temperature is not None and truth_salinity is not None:
            ax.plot(
                truth_salinity,
                truth_temperature,
                color="0.45",
                lw=1.2,
                ls="--",
                alpha=0.75,
                label="truth",
            )
        ax.plot(
            recon_salinity,
            recon_temperature,
            color="#d62728",
            marker="o",
            ms=2.8,
            lw=1.8,
            label="reconstruction",
        )
        ax.set_xlabel("Salinity")
        ax.set_ylabel("Temperature")
        ax.set_title(f"{profile_id or 'profile'} | epoch {epoch:03d} | sample {rank:02d}")
        ax.legend(frameon=False, loc="best")
        fig.savefig(path, dpi=self.dpi)
        plt.close(fig)

    def _extract_truth(self, sample: dict[str, Any], name: str) -> np.ndarray | None:
        recon = sample.get("recon_truth_physical")
        if recon is None:
            recon = sample.get("recon_truth")
        if recon is None:
            return None
        index = 0 if name == "truth_t" else 1
        arr = np.asarray(recon, dtype=float)
        if arr.ndim == 3:
            arr = arr[0]
        if arr.ndim != 2 or arr.shape[1] <= index:
            raise ValueError(f"Unexpected reconstruction truth shape: {arr.shape}")
        if sample.get("recon_truth_physical") is None and self.norm is not None:
            if index == 0:
                return self.norm.denormalize_temperature(arr[:, index])
            return self.norm.denormalize_salinity(arr[:, index])
        return arr[:, index]

    def _denormalize_reconstruction(self, recon_mean: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.norm is None:
            return recon_mean[:, 0], recon_mean[:, 1]
        return (
            self.norm.denormalize_temperature(recon_mean[:, 0]),
            self.norm.denormalize_salinity(recon_mean[:, 1]),
        )

    def _write_progress(self) -> None:
        self._write_json(self.progress_path, self._progress)

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(path)


def _group_history(history: list[dict[str, float]]) -> list[dict[str, Any]]:
    grouped: dict[int, dict[str, Any]] = {}
    for entry in history:
        epoch = int(entry.get("epoch", len(grouped) + 1))
        record = grouped.setdefault(epoch, {"epoch": epoch})
        train_metrics = {k.removeprefix("train_"): float(v) for k, v in entry.items() if k.startswith("train_")}
        val_metrics = {k.removeprefix("val_"): float(v) for k, v in entry.items() if k.startswith("val_")}
        if train_metrics:
            record["train"] = train_metrics
        if val_metrics:
            record["val"] = val_metrics
    return [grouped[key] for key in sorted(grouped)]


def _finite_limits(*arrays: np.ndarray) -> tuple[float, float]:
    values: list[np.ndarray] = []
    for arr in arrays:
        arr = np.asarray(arr, dtype=float).ravel()
        finite = arr[np.isfinite(arr)]
        if finite.size:
            values.append(finite)
    if not values:
        return 0.0, 1.0
    merged = np.concatenate(values)
    lo = float(np.nanmin(merged))
    hi = float(np.nanmax(merged))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return 0.0, 1.0
    if hi <= lo:
        span = max(abs(lo), 1.0)
        return lo - 0.5 * span, hi + 0.5 * span
    return lo, hi
