#!/usr/bin/env python3
"""Secondary capacity-matched neural information-set comparison.

This compact script retains only the two neural variants reported in Online
Resource 1: full cross-scale Mosaic and a capacity-matched local model with no
cross-scale histories. It uses the same 2022 training/validation blocks and
2023 test origins as the manuscript.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score

from mosaic.data import DemandLattice, SkillWindowDataset
from mosaic.losses import hurdle_loss
from mosaic.model import MOSAIC
from mosaic.protocol import HORIZON, LOOKBACK, TEST_STARTS, TRAIN_STARTS, VALIDATION_STARTS


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def estimate_pos_weight(lattice: DemandLattice) -> float:
    positives = 0
    total = 0
    for start in TRAIN_STARTS:
        y = lattice.bottom[..., start:start + HORIZON]
        positives += int((y > 0).sum())
        total += y.size
    negatives = total - positives
    return float(min(8.0, max(1.0, negatives / max(positives, 1))))


def sample_uniform_cells(raw: dict[str, torch.Tensor], cells: int, generator: torch.Generator):
    target = raw["bottom_target"].float()
    history = raw["bottom_history"].float()
    batch_size, regions, occupations, horizon = target.shape
    lookback = history.size(-1)
    n_cells = regions * occupations
    index = torch.randint(0, n_cells, (batch_size, cells), generator=generator)
    region_ids = index // occupations
    occupation_ids = index % occupations
    batch_index = torch.arange(batch_size)[:, None]
    flat_target = target.reshape(batch_size, n_cells, horizon)
    flat_history = history.reshape(batch_size, n_cells, lookback)
    return {
        "market_history": raw["market_history"].float(),
        "region_history": raw["region_history"].float()[batch_index, region_ids],
        "occupation_history": raw["occupation_history"].float()[batch_index, occupation_ids],
        "bottom_history": flat_history[batch_index, index],
        "bottom_target": flat_target[batch_index, index],
        "region_ids": region_ids,
        "occupation_ids": occupation_ids,
    }


def build_model(kind: str, lattice: DemandLattice) -> MOSAIC:
    if kind == "mosaic":
        return MOSAIC(
            LOOKBACK, HORIZON, lattice.n_regions, lattice.n_occupations,
            use_context=True, use_gate=True, exclusive_context=True, context_mode="full",
        )
    if kind == "local_matched":
        return MOSAIC(
            LOOKBACK, HORIZON, lattice.n_regions, lattice.n_occupations,
            use_context=False, use_gate=True, exclusive_context=True,
            context_mode="none", capacity_matched=True,
        )
    raise ValueError(kind)


def _loss(model: MOSAIC, batch: dict[str, torch.Tensor], pos_weight: float) -> torch.Tensor:
    history = batch["bottom_history"]
    target = batch["bottom_target"]
    forecast = model.forward_cells(
        batch["market_history"], batch["region_history"], batch["occupation_history"],
        history, batch["region_ids"], batch["occupation_ids"],
    )
    base = hurdle_loss(target, forecast.active_logits, forecast.positive_log, forecast.mean, pos_weight)
    event_target = (target > 0).any(dim=-1).float()
    active_months = (history > 0).sum(dim=-1)
    rare_mask = (history[..., -1] == 0) & (active_months <= 2)
    event_loss = F.binary_cross_entropy_with_logits(forecast.event_logit, event_target)
    if rare_mask.any():
        y_rare = event_target[rare_mask]
        pos = y_rare.sum()
        neg = y_rare.numel() - pos
        rare_pos_weight = torch.clamp(neg / (pos + 1e-6), 1.0, 12.0)
        rare_loss = F.binary_cross_entropy_with_logits(
            forecast.event_logit[rare_mask], y_rare, pos_weight=rare_pos_weight
        )
    else:
        rare_loss = event_loss.new_zeros(())
    return base + 0.25 * event_loss + rare_loss


def _full_validation_loss(model: MOSAIC, raw: dict[str, torch.Tensor], pos_weight: float) -> float:
    history = raw["bottom_history"].float()
    target = raw["bottom_target"].float()
    forecast = model(
        raw["market_history"].float(), raw["region_history"].float(),
        raw["occupation_history"].float(), history,
    )
    base = hurdle_loss(target, forecast.active_logits, forecast.positive_log, forecast.bottom_mean, pos_weight)
    event_target = (target > 0).any(dim=-1).float()
    active_months = (history > 0).sum(dim=-1)
    rare_mask = (history[..., -1] == 0) & (active_months <= 2)
    event_loss = F.binary_cross_entropy_with_logits(forecast.event_logit, event_target)
    if rare_mask.any():
        y_rare = event_target[rare_mask]
        pos = y_rare.sum()
        neg = y_rare.numel() - pos
        rare_pos_weight = torch.clamp(neg / (pos + 1e-6), 1.0, 12.0)
        rare_loss = F.binary_cross_entropy_with_logits(
            forecast.event_logit[rare_mask], y_rare, pos_weight=rare_pos_weight
        )
    else:
        rare_loss = event_loss.new_zeros(())
    return float((base + 0.25 * event_loss + rare_loss).item())


def predict_full(model: MOSAIC, lattice: DemandLattice):
    dataset = SkillWindowDataset(lattice, LOOKBACK, HORIZON, TEST_STARTS)
    loader = DataLoader(dataset, batch_size=256 if lattice.n_occupations <= 14 else 32, shuffle=False, num_workers=0)
    event_chunks, true_chunks, history_chunks, skill_chunks, start_chunks = [], [], [], [], []
    model.eval()
    with torch.no_grad():
        for raw in loader:
            history = raw["bottom_history"].float()
            fc = model(
                raw["market_history"].float(), raw["region_history"].float(),
                raw["occupation_history"].float(), history,
            )
            event_chunks.append(torch.sigmoid(fc.event_logit).cpu().numpy().astype(np.float32))
            true_chunks.append(raw["bottom_target"].numpy().astype(np.float32))
            history_chunks.append(raw["bottom_history"].numpy().astype(np.float32))
            skill_chunks.append(raw["skill_index"].numpy())
            start_chunks.append(raw["target_start"].numpy())
    return {
        "event_score": np.concatenate(event_chunks),
        "true": np.concatenate(true_chunks),
        "history": np.concatenate(history_chunks),
        "skill_index": np.concatenate(skill_chunks),
        "target_start": np.concatenate(start_chunks),
    }



def rare_dormant_ap(pred: dict[str, np.ndarray]) -> float:
    """Average precision on the manuscript's rare-dormant evaluation subset."""
    history = pred["history"]
    target = pred["true"]
    score = pred["event_score"]
    active_months = (history > 0).sum(axis=-1)
    mask = (history[..., -1] == 0) & (active_months <= 2)
    event = (target > 0).any(axis=-1).astype(np.int8)
    return float(average_precision_score(event[mask].reshape(-1), score[mask].reshape(-1)))

def train(kind: str, lattice: DemandLattice, seed: int, epochs: int):
    seed_all(seed)
    model = build_model(kind, lattice)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    train_loader = DataLoader(
        SkillWindowDataset(lattice, LOOKBACK, HORIZON, TRAIN_STARTS),
        batch_size=256 if lattice.n_occupations <= 14 else 96,
        shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        SkillWindowDataset(lattice, LOOKBACK, HORIZON, VALIDATION_STARTS),
        batch_size=256 if lattice.n_occupations <= 14 else 8,
        shuffle=False, num_workers=0,
    )
    pos_weight = estimate_pos_weight(lattice)
    generator = torch.Generator().manual_seed(seed + 1000)
    cells = 28 if lattice.n_occupations <= 14 else 36
    best = math.inf
    best_state = None
    stale = 0
    history = []
    started = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for raw in train_loader:
            batch = sample_uniform_cells(raw, cells, generator)
            optimizer.zero_grad(set_to_none=True)
            loss = _loss(model, batch, pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            val_losses = [_full_validation_loss(model, raw, pos_weight) for raw in val_loader]
        val_loss = float(np.mean(val_losses))
        history.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)), "validation_loss": val_loss})
        if val_loss < best - 1e-4:
            best = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= 3:
                break
    if best_state is None:
        raise RuntimeError("No checkpoint produced")
    model.load_state_dict(best_state)
    return model, predict_full(model, lattice), {
        "kind": kind, "seed": seed, "best_validation": best,
        "seconds": time.time() - started, "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lattice", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--epochs", type=int, default=4)
    args = parser.parse_args()
    lattice = DemandLattice.from_npz(args.lattice)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    summaries = []
    ap_by_kind = {}
    for kind in ("local_matched", "mosaic"):
        model, pred, summary = train(kind, lattice, args.seed, args.epochs)
        sub = out / kind
        sub.mkdir(exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "summary": summary}, sub / f"seed_{args.seed}.pt")
        np.savez_compressed(sub / f"seed_{args.seed}_test.npz", **pred)
        ap_by_kind[kind] = rare_dormant_ap(pred)
        summaries.append(summary)
    (out / "training_summary.json").write_text(json.dumps(summaries, indent=2))
    with (out / "neural_information_check_l1.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["variant", "information_set", "rare_dormant_ap"])
        writer.writerow(["Secondary neural", "Cross-scale histories + static IDs", ap_by_kind["mosaic"]])
        writer.writerow(["Secondary neural", "Local histories + static IDs", ap_by_kind["local_matched"]])


if __name__ == "__main__":
    main()
