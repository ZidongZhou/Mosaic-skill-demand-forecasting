from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


def history_change_features(x: torch.Tensor, short_window: int) -> torch.Tensor:
    """History-only indicators of a recent change in local demand.

    All statistics are computed from the pre-forecast input window.
    """
    short_window = min(short_window, x.size(-1))
    short = x[..., -short_window:]
    long = x[..., :-short_window] if x.size(-1) > short_window else x
    eps = 1e-5
    mean_shift = (short.mean(-1) - long.mean(-1)) / (
        long.std(-1, unbiased=False) + eps
    )
    variance_shift = torch.log(
        (short.var(-1, unbiased=False) + eps)
        / (long.var(-1, unbiased=False) + eps)
    )
    slope = (short[..., -1] - short[..., 0]) / max(short_window - 1, 1)
    jump = (
        short[..., -1] - short[..., -2]
        if short_window > 1
        else torch.zeros_like(short[..., -1])
    )
    return torch.stack([mean_shift, variance_shift, slope, jump], dim=-1)


class DualScaleEncoder(nn.Module):
    """Combine full-window and recent-window representations with an adaptive gate."""

    def __init__(
        self, lookback: int, short_window: int, hidden: int, dropout: float
    ) -> None:
        super().__init__()
        self.short_window = short_window
        self.long = nn.Sequential(
            nn.LayerNorm(lookback),
            nn.Linear(lookback, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        self.short = nn.Sequential(
            nn.LayerNorm(short_window),
            nn.Linear(short_window, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(4),
            nn.Linear(4, max(4, hidden // 2)),
            nn.GELU(),
            nn.Linear(max(4, hidden // 2), 1),
        )

    def forward(
        self, x: torch.Tensor, use_gate: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.log1p(x.clamp_min(0))
        long_state = self.long(z)
        short_state = self.short(z[..., -self.short_window :])
        if use_gate:
            gate = torch.sigmoid(
                self.gate(history_change_features(z, self.short_window))
            )
        else:
            gate = torch.full_like(long_state[..., :1], 0.5)
        state = (1 - gate) * long_state + gate * short_state
        return state, gate


@dataclass
class CellForecast:
    active_logits: torch.Tensor
    event_logit: torch.Tensor
    positive_log: torch.Tensor
    mean: torch.Tensor
    change_gate: torch.Tensor


@dataclass
class LatticeForecast:
    active_logits: torch.Tensor
    event_logit: torch.Tensor
    positive_log: torch.Tensor
    bottom_mean: torch.Tensor
    region_mean: torch.Tensor
    occupation_mean: torch.Tensor
    market_mean: torch.Tensor
    change_gate: torch.Tensor


class MOSAIC(nn.Module):
    """Cross-scale forecaster for a region-by-occupation demand lattice.

    The model predicts bottom cells. Higher-level forecasts are exact sums of the
    bottom surface. Cross-scale context is exclusive by default: regional and
    occupational contexts exclude the focal cell, while residual-market context
    excludes the focal region and focal occupation (M - R - O + Y).
    """

    def __init__(
        self,
        lookback: int,
        horizon: int,
        n_regions: int,
        n_occupations: int,
        short_window: int = 4,
        hidden: int = 28,
        embedding_dim: int = 5,
        dropout: float = 0.08,
        use_context: bool = True,
        use_gate: bool = True,
        exclusive_context: bool = True,
        context_mode: str | None = None,
        capacity_matched: bool = False,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.n_regions = n_regions
        self.n_occupations = n_occupations
        self.use_context = use_context
        self.use_gate = use_gate
        self.exclusive_context = exclusive_context
        if context_mode is None:
            context_mode = "full" if use_context else "none"
        allowed = {"full", "none", "region", "occupation", "market", "region_occupation"}
        if context_mode not in allowed:
            raise ValueError(f"Unknown context_mode={context_mode!r}; expected one of {sorted(allowed)}")
        self.context_mode = context_mode
        self.capacity_matched = capacity_matched

        self.encoder = DualScaleEncoder(
            lookback, short_window, hidden, dropout
        )
        self.region_embedding = nn.Embedding(n_regions, embedding_dim)
        self.occupation_embedding = nn.Embedding(n_occupations, embedding_dim)
        # The default reproduces the compact local-only ablation used in the
        # original experiment. Setting capacity_matched=True retains all four
        # context slots and fills unused slots with zeros, yielding a
        # parameter-matched local-only control.
        input_dim = hidden * (4 if (use_context or capacity_matched) else 1) + 2 * embedding_dim + 4
        self.fusion = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
        )
        self.head = nn.Linear(hidden, horizon * 3)
        # Direct quarterly activity head. It learns whether the cell becomes active
        # at least once over the three-month forecast block. This avoids deriving
        # a quarterly event score from a conditional-independence approximation.
        self.event_head = nn.Linear(hidden, 1)

    def _encode(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shape = x.shape[:-1]
        state, gate = self.encoder(
            x.reshape(-1, x.size(-1)), self.use_gate
        )
        return state.reshape(*shape, -1), gate.reshape(*shape, 1)

    def forward_cells(
        self,
        market_history: torch.Tensor,
        region_history: torch.Tensor,
        occupation_history: torch.Tensor,
        bottom_history: torch.Tensor,
        region_ids: torch.Tensor,
        occupation_ids: torch.Tensor,
    ) -> CellForecast:
        if market_history.ndim == 2:
            market_history = market_history[:, None, :].expand_as(bottom_history)
        if self.use_context and self.exclusive_context:
            # Preserve the original margins before converting them to exclusive
            # context. The residual market is disjoint from the focal region and
            # focal occupation: M - R - O + Y.
            region_margin = region_history
            occupation_margin = occupation_history
            market_margin = market_history
            region_history = (region_margin - bottom_history).clamp_min(0)
            occupation_history = (occupation_margin - bottom_history).clamp_min(0)
            market_history = (
                market_margin - region_margin - occupation_margin + bottom_history
            ).clamp_min(0)

        market_state, _ = self._encode(market_history)
        region_state, _ = self._encode(region_history)
        occupation_state, _ = self._encode(occupation_history)
        bottom_state, bottom_gate = self._encode(bottom_history)

        zero = torch.zeros_like(bottom_state)
        use_region = self.context_mode in {"full", "region", "region_occupation"}
        use_occupation = self.context_mode in {"full", "occupation", "region_occupation"}
        use_market = self.context_mode in {"full", "market"}
        if self.use_context or self.capacity_matched:
            parts = [
                bottom_state,
                region_state if use_region else zero,
                occupation_state if use_occupation else zero,
                market_state if use_market else zero,
            ]
        else:
            parts = [bottom_state]

        region_embedding = self.region_embedding(region_ids)
        occupation_embedding = self.occupation_embedding(occupation_ids)
        change_statistics = history_change_features(
            torch.log1p(bottom_history.clamp_min(0)),
            min(4, bottom_history.size(-1)),
        )
        hidden = self.fusion(
            torch.cat(
                parts
                + [
                    region_embedding,
                    occupation_embedding,
                    change_statistics,
                ],
                dim=-1,
            )
        )
        raw = self.head(hidden).reshape(*hidden.shape[:-1], self.horizon, 3)

        active_logits = raw[..., 0]
        event_logit = self.event_head(hidden).squeeze(-1)
        bounded_growth = 1.6 * torch.tanh(raw[..., 1])
        cold_start_log = F.softplus(raw[..., 2])

        last = bottom_history[..., -1, None]
        persistent_log = torch.log1p(last) + bounded_growth
        positive_log = torch.where(last > 0, persistent_log, cold_start_log)
        positive_mean = (torch.exp(positive_log.clamp(max=12)) - 1).clamp_min(0)
        mean = torch.sigmoid(active_logits) * positive_mean

        return CellForecast(
            active_logits=active_logits,
            event_logit=event_logit,
            positive_log=positive_log,
            mean=mean,
            change_gate=bottom_gate,
        )

    def forward(
        self,
        market_history: torch.Tensor,
        region_history: torch.Tensor,
        occupation_history: torch.Tensor,
        bottom_history: torch.Tensor,
    ) -> LatticeForecast:
        batch, regions, occupations, lookback = bottom_history.shape
        region_ids = (
            torch.arange(regions, device=bottom_history.device)[None, :, None]
            .expand(batch, regions, occupations)
            .reshape(batch, -1)
        )
        occupation_ids = (
            torch.arange(occupations, device=bottom_history.device)[None, None, :]
            .expand(batch, regions, occupations)
            .reshape(batch, -1)
        )
        bottom_flat = bottom_history.reshape(batch, regions * occupations, lookback)
        region_flat = (
            region_history[:, :, None, :]
            .expand(batch, regions, occupations, lookback)
            .reshape(batch, regions * occupations, lookback)
        )
        occupation_flat = (
            occupation_history[:, None, :, :]
            .expand(batch, regions, occupations, lookback)
            .reshape(batch, regions * occupations, lookback)
        )
        forecast = self.forward_cells(
            market_history,
            region_flat,
            occupation_flat,
            bottom_flat,
            region_ids,
            occupation_ids,
        )

        def lattice_shape(x: torch.Tensor) -> torch.Tensor:
            return x.reshape(batch, regions, occupations, *x.shape[2:])

        bottom = lattice_shape(forecast.mean)
        return LatticeForecast(
            active_logits=lattice_shape(forecast.active_logits),
            event_logit=lattice_shape(forecast.event_logit),
            positive_log=lattice_shape(forecast.positive_log),
            bottom_mean=bottom,
            region_mean=bottom.sum(2),
            occupation_mean=bottom.sum(1),
            market_mean=bottom.sum((1, 2)),
            change_gate=forecast.change_gate.reshape(batch, regions, occupations, 1),
        )
