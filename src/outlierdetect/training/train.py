"""Minimal training loop for neural model.

This is deliberately lightweight. It defines losses and an epoch loop, but leaves
experiment management, logging, and data ingestion to project-specific scripts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
except Exception:  # pragma: no cover - torch is optional
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    nn = Any  # type: ignore[assignment]

from ..corrections import CorrectionPrior


@dataclass(slots=True)
class LossWeights:
    profile_qc: float = 1.0
    point_qc: float = 1.0
    nuisance: float = 0.2
    nuisance_prior_kl: float = 0.02
    nuisance_prior: CorrectionPrior = field(default_factory=CorrectionPrior.default)
    reconstruction: float = 1.0
    uncertainty: float = 0.02


if torch is not None:

    def compute_loss(
        outputs: dict[str, torch.Tensor],
        batch: dict[str, Any],
        weights: LossWeights | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the MVP multitask loss.

        Handles missing labels using masks. Reconstruction loss uses a Gaussian
        negative log-likelihood when reconstruction targets are present.
        """
        weights = weights or LossWeights()
        device = outputs["profile_logit"].device
        total = torch.zeros((), device=device)
        logs: dict[str, float] = {}

        profile_mask = batch["profile_bad_mask"].to(device)
        if profile_mask.any():
            target = batch["profile_bad"].to(device)[profile_mask]
            pred = outputs["profile_logit"][profile_mask]
            loss = F.binary_cross_entropy_with_logits(pred, target)
            total = total + weights.profile_qc * loss
            logs["profile_qc"] = float(loss.detach().cpu())

        point_mask = batch["point_label_mask"].to(device)
        if point_mask.any():
            labels = batch["point_labels"].to(device)
            pred = outputs["point_logits"]
            valid = point_mask.unsqueeze(-1) & torch.isfinite(labels)
            loss = F.binary_cross_entropy_with_logits(pred[valid], labels[valid])
            total = total + weights.point_qc * loss
            logs["point_qc"] = float(loss.detach().cpu())

        nuisance_mask = batch["nuisance_mask"].to(device)
        if nuisance_mask.any():
            target = batch["nuisance_mean"].to(device)[nuisance_mask]
            mean = outputs["nuisance_mean"][nuisance_mask]
            log_std = outputs["nuisance_log_std"][nuisance_mask]
            loss = gaussian_nll(target, mean, log_std)
            total = total + weights.nuisance * loss
            logs["nuisance"] = float(loss.detach().cpu())

        if weights.nuisance_prior_kl > 0 and "nuisance_mean" in outputs:
            kl = gaussian_diag_kl_to_full_prior(
                outputs["nuisance_mean"],
                outputs["nuisance_log_std"],
                weights.nuisance_prior,
            )
            total = total + weights.nuisance_prior_kl * kl
            logs["nuisance_prior_kl"] = float(kl.detach().cpu())

        recon_truth = batch.get("recon_truth")
        if recon_truth is not None:
            target = recon_truth.to(device)
            mean = outputs["recon_mean"]
            log_std = outputs["recon_log_std"]
            if target.shape == mean.shape:
                loss = gaussian_nll(target, mean, log_std)
                total = total + weights.reconstruction * loss
                logs["reconstruction"] = float(loss.detach().cpu())
                if weights.uncertainty > 0:
                    uncertainty = torch.mean(torch.relu(log_std) ** 2)
                    total = total + weights.uncertainty * uncertainty
                    logs["uncertainty"] = float(uncertainty.detach().cpu())

        logs["total"] = float(total.detach().cpu())
        return total, logs


    def gaussian_nll(target: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
        var = torch.exp(2.0 * log_std).clamp_min(1e-10)
        return torch.mean(0.5 * ((target - mean) ** 2 / var + torch.log(var)))


    def gaussian_diag_kl_to_full_prior(
        mean_q: torch.Tensor,
        log_std_q: torch.Tensor,
        prior: CorrectionPrior,
    ) -> torch.Tensor:
        prior_mean = torch.as_tensor(prior.mean, dtype=mean_q.dtype, device=mean_q.device)
        prior_cov = torch.as_tensor(prior.covariance, dtype=mean_q.dtype, device=mean_q.device)
        prior_prec = torch.linalg.inv(prior_cov)
        prior_logdet = torch.logdet(prior_cov)

        diff = mean_q - prior_mean
        var_q = torch.exp(2.0 * log_std_q).clamp_min(1e-10)
        trace_term = torch.sum(var_q * torch.diagonal(prior_prec), dim=-1)
        quad_term = torch.einsum("bi,ij,bj->b", diff, prior_prec, diff)
        logdet_q = torch.sum(2.0 * log_std_q, dim=-1)
        k = float(mean_q.shape[-1])
        kl = 0.5 * (trace_term + quad_term - k + prior_logdet - logdet_q)
        return torch.mean(torch.clamp(kl, min=0.0))


    def train_epoch(
        model: nn.Module,
        loader: Any,
        optimizer: torch.optim.Optimizer,
        *,
        device: str | torch.device = "cpu",
        weights: LossWeights | None = None,
        grad_clip: float | None = 1.0,
    ) -> dict[str, float]:
        """Train one epoch and return averaged losses."""
        model.train()
        model.to(device)
        totals: dict[str, float] = {}
        n_batches = 0
        for batch in loader:
            features = batch["features"].to(device)
            mask = batch["mask"].to(device)
            pressure_grid = batch.get("pressure_grid")
            if pressure_grid is not None:
                pressure_grid = pressure_grid.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(features, mask=mask, recon_pressure=pressure_grid)
            batch_loss, logs = compute_loss(outputs, batch, weights=weights)
            batch_loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            for key, value in logs.items():
                totals[key] = totals.get(key, 0.0) + float(value)
            n_batches += 1
        return {key: value / max(n_batches, 1) for key, value in totals.items()}


    def eval_epoch(
        model: nn.Module,
        loader: Any,
        *,
        device: str | torch.device = "cpu",
        weights: LossWeights | None = None,
    ) -> dict[str, float]:
        """Evaluate one epoch and return averaged losses."""
        model.eval()
        model.to(device)
        totals: dict[str, float] = {}
        n_batches = 0
        with torch.no_grad():
            for batch in loader:
                features = batch["features"].to(device)
                mask = batch["mask"].to(device)
                pressure_grid = batch.get("pressure_grid")
                if pressure_grid is not None:
                    pressure_grid = pressure_grid.to(device)
                outputs = model(features, mask=mask, recon_pressure=pressure_grid)
                _, logs = compute_loss(outputs, batch, weights=weights)
                for key, value in logs.items():
                    totals[key] = totals.get(key, 0.0) + float(value)
                n_batches += 1
        return {key: value / max(n_batches, 1) for key, value in totals.items()}


    def fit_model(
        model: nn.Module,
        train_loader: Any,
        val_loader: Any | None = None,
        *,
        eval_label: str = "val",
        device: str | torch.device = "cpu",
        weights: LossWeights | None = None,
        grad_clip: float | None = 1.0,
        epochs: int = 1,
        optimizer: torch.optim.Optimizer | None = None,
        learning_rate: float = 1e-3,
        epoch_callback: Callable[[int, nn.Module, list[dict[str, float]]], None] | None = None,
    ) -> list[dict[str, float]]:
        """Fit a model for a small number of epochs and return per-epoch logs."""
        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        history: list[dict[str, float]] = []
        for epoch in range(1, max(int(epochs), 1) + 1):
            train_logs = train_epoch(
                model,
                train_loader,
                optimizer,
                device=device,
                weights=weights,
                grad_clip=grad_clip,
            )
            train_logs = {f"train_{key}": value for key, value in train_logs.items()}
            train_logs["epoch"] = float(epoch)
            history.append(train_logs)

            if val_loader is not None:
                val_logs = eval_epoch(
                    model,
                    val_loader,
                    device=device,
                    weights=weights,
                )
                label = str(eval_label).strip() or "val"
                val_logs = {f"{label}_{key}": value for key, value in val_logs.items()}
                val_logs["epoch"] = float(epoch)
                history.append(val_logs)

            if epoch_callback is not None:
                epoch_callback(epoch, model, history)

        return history


    def save_checkpoint(
        path: str | Path,
        model: nn.Module,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save a model checkpoint together with small metadata."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state_dict": model.state_dict(),
            "metadata": metadata or {},
        }
        torch.save(payload, path)


    def load_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
        """Load a checkpoint saved by :func:`save_checkpoint`."""
        payload = torch.load(Path(path), map_location=map_location)
        if not isinstance(payload, dict):
            raise TypeError("Checkpoint did not contain a dictionary payload.")
        return payload


    def load_model_from_checkpoint(
        path: str | Path,
        model_factory: Any,
        *,
        map_location: str | torch.device = "cpu",
    ) -> tuple[nn.Module, dict[str, Any]]:
        """Load a checkpoint and reconstruct the model with a factory.

        ``model_factory`` should accept a metadata dictionary and return an
        uninitialized model instance with the same architecture as the one used
        during training.
        """
        payload = load_checkpoint(path, map_location=map_location)
        metadata = dict(payload.get("metadata", {}))
        model = model_factory(metadata)
        state_dict = payload.get("model_state_dict")
        if state_dict is None:
            raise KeyError("Checkpoint did not contain a model_state_dict entry.")
        model.load_state_dict(state_dict)
        return model, metadata


    # Backwards-compatible alias.
    loss = compute_loss

else:

    def loss(*_: Any, **__: Any) -> Any:  # type: ignore[misc]
        raise ImportError("Training requires PyTorch. Install with the train extra.")

    def train_epoch(*_: Any, **__: Any) -> Any:  # type: ignore[misc]
        raise ImportError("Training requires PyTorch. Install with the train extra.")

    def eval_epoch(*_: Any, **__: Any) -> Any:  # type: ignore[misc]
        raise ImportError("Training requires PyTorch. Install with the train extra.")

    def fit_model(*_: Any, **__: Any) -> Any:  # type: ignore[misc]
        raise ImportError("Training requires PyTorch. Install with the train extra.")

    def save_checkpoint(*_: Any, **__: Any) -> Any:  # type: ignore[misc]
        raise ImportError("Training requires PyTorch. Install with the train extra.")

    def load_checkpoint(*_: Any, **__: Any) -> Any:  # type: ignore[misc]
        raise ImportError("Training requires PyTorch. Install with the train extra.")

    def load_model_from_checkpoint(*_: Any, **__: Any) -> Any:  # type: ignore[misc]
        raise ImportError("Training requires PyTorch. Install with the train extra.")
