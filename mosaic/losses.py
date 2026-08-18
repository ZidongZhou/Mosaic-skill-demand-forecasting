from __future__ import annotations

import torch
import torch.nn.functional as F


def hurdle_loss(
    y: torch.Tensor,
    active_logits: torch.Tensor,
    positive_log: torch.Tensor,
    mean: torch.Tensor,
    pos_weight: float = 1.0,
) -> torch.Tensor:
    """Activation plus positive-magnitude loss on log counts."""
    active = (y > 0).to(y.dtype)
    positive_weight = torch.as_tensor(pos_weight, device=y.device, dtype=y.dtype)
    activation_loss = F.binary_cross_entropy_with_logits(
        active_logits, active, pos_weight=positive_weight, reduction="mean"
    )
    transformed = torch.log1p(y)
    positive = active > 0
    if positive.any():
        magnitude_loss = F.smooth_l1_loss(
            positive_log[positive], transformed[positive], beta=0.25
        )
    else:
        magnitude_loss = torch.zeros((), device=y.device)
    point_loss = F.smooth_l1_loss(torch.log1p(mean), transformed, beta=0.25)
    return 0.8 * activation_loss + 1.1 * magnitude_loss + 0.55 * point_loss


def hurdle_nb_loss(
    y: torch.Tensor,
    active_logits: torch.Tensor,
    log_mu: torch.Tensor,
    log_dispersion: torch.Tensor,
    pos_weight: float = 1.0,
) -> torch.Tensor:
    """Hurdle negative-binomial likelihood for intermittent counts."""
    active = (y > 0).to(y.dtype)
    pw = torch.as_tensor(pos_weight, device=y.device, dtype=y.dtype)
    activation_loss = F.binary_cross_entropy_with_logits(
        active_logits, active, pos_weight=pw, reduction="mean"
    )
    positive = y > 0
    if positive.any():
        mu = torch.exp(log_mu[positive].clamp(-8, 12)).clamp_min(1e-5)
        dispersion = F.softplus(log_dispersion[positive]).clamp_min(1e-3)
        probs = (mu / (mu + dispersion)).clamp(1e-6, 1 - 1e-6)
        dist = torch.distributions.NegativeBinomial(
            total_count=dispersion, probs=probs
        )
        # For a hurdle model the positive component is conditioned on Y>0.
        log_p0 = dist.log_prob(torch.zeros_like(mu))
        log_norm = torch.log1p(-torch.exp(log_p0).clamp(max=1 - 1e-7))
        magnitude_nll = -(dist.log_prob(y[positive]) - log_norm).mean()
    else:
        magnitude_nll = torch.zeros((), device=y.device)
    return activation_loss + 0.45 * magnitude_nll
