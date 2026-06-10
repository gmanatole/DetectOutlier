"""Neural-network skeleton.

The model is intentionally small and explicit. It supports irregular profiles by
using padded level features and a boolean valid-level mask. Reconstruction is
made on a fixed standard pressure grid chosen by the caller/training config.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class NetConfig:
    input_dim: int
    grid_size: int = 80
    hidden_dim: int = 128
    n_layers: int = 3
    n_heads: int = 4
    dropout: float = 0.1
    point_outputs: int = 3  # T bad, S bad, density inconsistent
    nuisance_outputs: int = 8  # mean/log_std for a_t,b_t,a_s,b_s
    recon_pressure_features: int = 2


try:
    import torch
    from torch import Tensor, nn
except Exception:  # pragma: no cover - torch is an optional dependency
    torch = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[misc,assignment]
    nn = Any  # type: ignore[misc,assignment]


if torch is not None:

    class Net(nn.Module):
        """MVP transformer model for local profile QC and reconstruction.

        Inputs
        ------
        features:
            Tensor with shape ``[batch, n_levels, input_dim]``.
        mask:
            Boolean tensor with shape ``[batch, n_levels]``. True means valid.

        Outputs
        -------
        dict
            ``profile_logit``: [batch]
            ``point_logits``: [batch, n_levels, 3]
            ``nuisance_mean``: [batch, 4]
            ``nuisance_log_std``: [batch, 4]
            ``recon_mean``: [batch, grid_size, 2]
            ``recon_log_std``: [batch, grid_size, 2]
        """

        def __init__(self, config: NetConfig):
            super().__init__()
            self.config = config
            self.input = nn.Sequential(
                nn.Linear(config.input_dim, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.GELU(),
            )
            layer = nn.TransformerEncoderLayer(
                d_model=config.hidden_dim,
                nhead=config.n_heads,
                dim_feedforward=4 * config.hidden_dim,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=config.n_layers)
            self.point_head = nn.Linear(config.hidden_dim, config.point_outputs)
            self.profile_head = nn.Sequential(
                nn.LayerNorm(config.hidden_dim),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, 1),
            )
            self.nuisance_head = nn.Sequential(
                nn.LayerNorm(config.hidden_dim),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, config.nuisance_outputs),
            )
            self.recon_pressure_encoder = nn.Sequential(
                nn.Linear(config.recon_pressure_features, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.GELU(),
            )
            self.recon_head = nn.Sequential(
                nn.LayerNorm(2 * config.hidden_dim),
                nn.Linear(2 * config.hidden_dim, config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, 4),
            )

        def forward(
            self,
            features: Tensor,
            mask: Tensor | None = None,
            recon_pressure: Tensor | None = None,
        ) -> dict[str, Tensor]:
            if mask is None:
                mask = torch.isfinite(features).all(dim=-1)
            features = torch.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
            h = self.input(features)
            # PyTorch src_key_padding_mask uses True for padding; our mask uses True for valid.
            padding_mask = ~mask.bool()
            h = self.encoder(h, src_key_padding_mask=padding_mask)
            point_logits = self.point_head(h)
            pooled = masked_mean(h, mask.bool(), dim=1)
            profile_logit = self.profile_head(pooled).squeeze(-1)
            nuisance = self.nuisance_head(pooled)
            nuisance_mean = nuisance[:, :4]
            nuisance_log_std = nuisance[:, 4:8].clamp(-8.0, 4.0)
            recon_pressure = self._prepare_pressure_grid(features, recon_pressure)
            pressure_features = self._pressure_features(recon_pressure)
            pressure_embed = self.recon_pressure_encoder(pressure_features)
            pooled_grid = pooled.unsqueeze(1).expand(-1, pressure_embed.shape[1], -1)
            recon = self.recon_head(torch.cat([pooled_grid, pressure_embed], dim=-1))
            recon_mean = recon[..., :2]
            recon_log_std = recon[..., 2:].clamp(-8.0, 2.5)
            return {
                "profile_logit": profile_logit,
                "point_logits": point_logits,
                "nuisance_mean": nuisance_mean,
                "nuisance_log_std": nuisance_log_std,
                "recon_mean": recon_mean,
                "recon_log_std": recon_log_std,
            }

        def _prepare_pressure_grid(self, features: Tensor, recon_pressure: Tensor | None) -> Tensor:
            if recon_pressure is None:
                grid = torch.linspace(
                    0.0,
                    1.0,
                    self.config.grid_size,
                    device=features.device,
                    dtype=features.dtype,
                )
                return grid.unsqueeze(0).expand(features.shape[0], -1)

            grid = torch.as_tensor(recon_pressure, dtype=features.dtype, device=features.device)
            if grid.ndim == 1:
                grid = grid.unsqueeze(0)
            if grid.ndim != 2:
                raise ValueError("recon_pressure must have shape [batch, grid_size] or [grid_size].")
            return grid

        def _pressure_features(self, recon_pressure: Tensor) -> Tensor:
            p_min = recon_pressure.amin(dim=1, keepdim=True)
            p_max = recon_pressure.amax(dim=1, keepdim=True)
            p_span = (p_max - p_min).clamp_min(1.0)
            p_norm = (recon_pressure - p_min) / p_span
            p_scaled = recon_pressure / p_max.clamp_min(1.0)
            return torch.stack([p_norm, p_scaled], dim=-1)


    def masked_mean(x: Tensor, mask: Tensor, dim: int) -> Tensor:
        mask_f = mask.to(dtype=x.dtype)
        while mask_f.ndim < x.ndim:
            mask_f = mask_f.unsqueeze(-1)
        numerator = torch.sum(x * mask_f, dim=dim)
        denominator = torch.sum(mask_f, dim=dim).clamp_min(1.0)
        return numerator / denominator

else:

    class Net:  # type: ignore[no-redef]
        """Placeholder when torch is not installed."""

        def __init__(self, *_: Any, **__: Any) -> None:
            raise ImportError(
                "Net requires PyTorch. Install with: pip install 'OutlierDetect[train]'"
            )


# Public aliases.
ProfileNetConfig = NetConfig
ProfileNet = Net
