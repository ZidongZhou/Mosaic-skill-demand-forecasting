#!/usr/bin/env python3
"""Reproduce the 2,324-ID sensitivity using the current strong-local primary specification."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from mosaic.data import DemandLattice
from strong_matched_cross_scale import fit_model, predict_blocks, ap


def truncate_skills(lat: DemandLattice, n_skills: int) -> DemandLattice:
    if n_skills <= 0 or n_skills > lat.n_skills:
        raise ValueError(f"n_skills must be in 1..{lat.n_skills}")
    out = DemandLattice(
        market=lat.market[:n_skills].copy(),
        region=lat.region[:n_skills].copy(),
        occupation=lat.occupation[:n_skills].copy(),
        bottom=lat.bottom[:n_skills].copy(),
        months=list(lat.months),
        occupation_level=lat.occupation_level,
    )
    out.validate()
    return out


def evaluate_level(path: str, level: str, n_skills: int, n_jobs: int) -> dict:
    lat = truncate_skills(DemandLattice.from_npz(path), n_skills)
    local = fit_model(lat, False, n_jobs)
    full = fit_model(lat, True, n_jobs)
    ls, y, d, a, skill, block = predict_blocks(lat, local, False)
    fs, y2, d2, a2, skill2, block2 = predict_blocks(lat, full, True)
    assert np.array_equal(y, y2) and np.array_equal(d, d2) and np.array_equal(a, a2)
    rare = d & (a <= 2)
    la = ap(y, ls, rare); fa = ap(y, fs, rare)
    return {
        'level': level.upper(), 'skills': lat.n_skills, 'n': int(rare.sum()),
        'positive': int(y[rare].sum()), 'strong_local_ap': la,
        'full_mosaic_ap': fa, 'gain': fa-la,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--l1', required=True)
    p.add_argument('--l2', required=True)
    p.add_argument('--skills', type=int, default=2324)
    p.add_argument('--output', required=True)
    p.add_argument('--n-jobs', type=int, default=4)
    a=p.parse_args()
    rows=[evaluate_level(a.l1,'L1',a.skills,a.n_jobs), evaluate_level(a.l2,'L2',a.skills,a.n_jobs)]
    out=Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    df=pd.DataFrame(rows); df.to_csv(out,index=False); print(df.to_string(index=False))

if __name__=='__main__':
    main()
