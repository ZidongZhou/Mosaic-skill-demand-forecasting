#!/usr/bin/env python3
"""Run the primary matched strong-local versus cross-scale GBM comparison.

The two models use the same learner and the same focal-cell information. The
cross-scale model adds disjoint regional, occupational and residual-market
histories. The script writes the rare-dormant summary, the active-month
observability gradient, quarterly results and an optional score archive for
paired cluster bootstrap.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import average_precision_score
from mosaic.data import DemandLattice
from mosaic.protocol import LOOKBACK, HORIZON, TRAIN_STARTS, TEST_STARTS


def memory_features(lat: DemandLattice, start: int) -> np.ndarray:
    hist = lat.bottom[..., :start]
    active = hist > 0
    cumulative_active = active.sum(-1).astype(np.float32)
    cumulative_active_ratio = cumulative_active / float(start)
    any_positive = active.any(-1)
    months_since = np.where(any_positive, active[..., ::-1].argmax(-1), start).astype(np.float32)
    pre_end = max(0, start - LOOKBACK)
    prior_before_window = ((lat.bottom[..., :pre_end] > 0).any(-1).astype(np.float32)
                           if pre_end else np.zeros_like(cumulative_active, dtype=np.float32))
    return np.stack([
        cumulative_active,
        cumulative_active_ratio,
        np.log1p(months_since),
        prior_before_window,
    ], axis=-1).reshape(-1, 4).astype(np.float32)


def feature_matrix(lat: DemandLattice, start: int, cross_scale: bool):
    b = lat.bottom[..., start-LOOKBACK:start].astype(np.float32)
    parts = [np.log1p(b).reshape(-1, LOOKBACK)]
    if cross_scale:
        reg = np.maximum(lat.region[:, :, start-LOOKBACK:start][:, :, None, :] - b, 0)
        occ = np.maximum(lat.occupation[:, :, start-LOOKBACK:start][:, None, :, :] - b, 0)
        market = lat.market[:, start-LOOKBACK:start][:, None, None, :]
        res = np.maximum(
            market
            - lat.region[:, :, start-LOOKBACK:start][:, :, None, :]
            - lat.occupation[:, :, start-LOOKBACK:start][:, None, :, :]
            + b,
            0,
        )
        parts.extend([
            np.log1p(reg).reshape(-1, LOOKBACK),
            np.log1p(occ).reshape(-1, LOOKBACK),
            np.log1p(res).reshape(-1, LOOKBACK),
        ])
    parts.extend([
        (b > 0).sum(-1).reshape(-1, 1).astype(np.float32),
        np.log1p(b[..., -1]).reshape(-1, 1).astype(np.float32),
        memory_features(lat, start),
    ])
    X = np.concatenate(parts, axis=1)
    y = (lat.bottom[..., start:start+HORIZON] > 0).any(-1).reshape(-1).astype(np.uint8)
    return X, y, b


def fit_model(lat: DemandLattice, cross_scale: bool, n_jobs: int):
    Xs, ys = [], []
    for start in TRAIN_STARTS:
        X, y, _ = feature_matrix(lat, start, cross_scale)
        Xs.append(X); ys.append(y)
    model = lgb.LGBMClassifier(
        n_estimators=45, num_leaves=25, learning_rate=0.07,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        reg_lambda=1.0, class_weight='balanced', random_state=2026,
        n_jobs=n_jobs, verbosity=-1,
    )
    model.fit(np.concatenate(Xs), np.concatenate(ys))
    return model


def predict_blocks(lat: DemandLattice, model, cross_scale: bool):
    scores=[]; targets=[]; dormant=[]; active=[]; skills=[]; blocks=[]
    ncell = lat.n_regions * lat.n_occupations
    skill_ids = np.repeat(np.arange(lat.n_skills), ncell)
    for block_id, start in enumerate(TEST_STARTS):
        X, y, b = feature_matrix(lat, start, cross_scale)
        scores.append(model.predict_proba(X)[:, 1])
        targets.append(y)
        dormant.append((b[..., -1] == 0).reshape(-1))
        active.append((b > 0).sum(-1).reshape(-1))
        skills.append(skill_ids)
        blocks.append(np.full(y.shape, block_id, dtype=np.int8))
    return tuple(map(np.concatenate, (scores, targets, dormant, active, skills, blocks)))


def ap(y, s, mask):
    return float(average_precision_score(y[mask], s[mask]))


def main():
    apx = argparse.ArgumentParser()
    apx.add_argument('--lattice', required=True)
    apx.add_argument('--level', required=True, choices=['L1','L2','l1','l2'])
    apx.add_argument('--output', required=True)
    apx.add_argument('--n-jobs', type=int, default=4)
    apx.add_argument('--save-scores', action='store_true')
    args = apx.parse_args()
    level=args.level.upper(); out=Path(args.output); out.mkdir(parents=True, exist_ok=True)
    lat=DemandLattice.from_npz(args.lattice)
    local=fit_model(lat, False, args.n_jobs); full=fit_model(lat, True, args.n_jobs)
    ls,y,d,a,skill,block=predict_blocks(lat,local,False)
    fs,y2,d2,a2,skill2,block2=predict_blocks(lat,full,True)
    assert np.array_equal(y,y2) and np.array_equal(d,d2) and np.array_equal(a,a2)
    assert np.array_equal(skill,skill2) and np.array_equal(block,block2)

    bins=[('0',a==0),('1',a==1),('2',a==2),('3-5',(a>=3)&(a<=5)),('6-11',(a>=6)&(a<=11))]
    rows=[]
    for label,bmask in bins:
        mask=d & bmask
        la=ap(y,ls,mask); fa=ap(y,fs,mask)
        rows.append(dict(level=level,active_months=label,n=int(mask.sum()),positive=int(y[mask].sum()),
                         prevalence=float(y[mask].mean()),strong_local_ap=la,strong_full_ap=fa,gain=fa-la))
    grad=pd.DataFrame(rows)
    grad.to_csv(out/f'strong_matched_gradient_{level}.csv',index=False)
    norm=grad[['level','active_months','n','positive','prevalence','strong_local_ap','strong_full_ap','gain']].copy()
    den=1.0-norm['prevalence']
    norm['strong_local_normalized_ap']=(norm['strong_local_ap']-norm['prevalence'])/den
    norm['strong_full_normalized_ap']=(norm['strong_full_ap']-norm['prevalence'])/den
    norm['normalized_gain']=norm['strong_full_normalized_ap']-norm['strong_local_normalized_ap']
    norm.to_csv(out/f'strong_observability_normalized_{level}.csv',index=False)

    rare=d & (a<=2)
    la=ap(y,ls,rare); fa=ap(y,fs,rare)
    pd.DataFrame([dict(level=level,n=int(rare.sum()),positive=int(y[rare].sum()),strong_local_ap=la,
                       strong_full_ap=fa,gain=fa-la)]).to_csv(out/f'strong_matched_summary_{level}.csv',index=False)

    quarter_labels=['2023-01 to 2023-03','2023-04 to 2023-06','2023-07 to 2023-09','2023-10 to 2023-12']
    qrows=[]
    for bid,label in enumerate(quarter_labels):
        mask=rare & (block==bid); ql=ap(y,ls,mask); qf=ap(y,fs,mask)
        qrows.append(dict(level=level,forecast_block=label,strong_local_ap=ql,strong_full_ap=qf,gain=qf-ql,
                          n=int(mask.sum()),positive=int(y[mask].sum())))
    pd.DataFrame(qrows).to_csv(out/f'strong_quarterly_{level}.csv',index=False)

    if args.save_scores:
        np.savez_compressed(out/f'strong_matched_scores_{level}.npz',local=ls,full=fs,target=y,
                            dormant=d,active=a,skill=skill,block=block)

if __name__ == '__main__':
    main()
