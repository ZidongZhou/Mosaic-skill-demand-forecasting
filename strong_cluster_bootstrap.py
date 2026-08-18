#!/usr/bin/env python3
"""Paired skill-cluster bootstrap for the strong matched comparison.

Each observability subset resamples only the skill clusters represented in that
subset. Scores are sorted once per model, and a Numba implementation computes
weighted average precision exactly, including score ties. This keeps 1,000
cluster-bootstrap replicates practical for the large L2 evaluation set.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from numba import njit, prange


@njit
def _ap_one(y_sorted, cluster_sorted, group_end, counts, positives_by_cluster):
    total_positive = 0.0
    for k in range(len(positives_by_cluster)):
        total_positive += counts[k] * positives_by_cluster[k]
    if total_positive <= 0:
        return np.nan
    cumulative_total = 0.0
    cumulative_positive = 0.0
    group_positive = 0.0
    ap = 0.0
    for i in range(len(y_sorted)):
        w = counts[cluster_sorted[i]]
        if w:
            cumulative_total += w
            if y_sorted[i]:
                cumulative_positive += w
                group_positive += w
        if group_end[i]:
            if group_positive > 0 and cumulative_total > 0:
                ap += (group_positive / total_positive) * (cumulative_positive / cumulative_total)
            group_positive = 0.0
    return ap


@njit(parallel=True)
def _ap_many(y_sorted, cluster_sorted, group_end, counts_matrix, positives_by_cluster):
    out = np.empty(counts_matrix.shape[0], dtype=np.float64)
    for b in prange(counts_matrix.shape[0]):
        out[b] = _ap_one(y_sorted, cluster_sorted, group_end, counts_matrix[b], positives_by_cluster)
    return out


def _prepare(y, scores, cluster_inverse, n_clusters):
    order = np.argsort(-scores, kind='mergesort')
    sorted_scores = scores[order]
    y_sorted = y[order].astype(np.uint8)
    cluster_sorted = cluster_inverse[order].astype(np.int32)
    group_end = np.empty(len(sorted_scores), dtype=np.bool_)
    if len(sorted_scores) > 1:
        group_end[:-1] = sorted_scores[:-1] != sorted_scores[1:]
    group_end[-1] = True
    positives = np.bincount(cluster_inverse, weights=y, minlength=n_clusters).astype(np.int32)
    return y_sorted, cluster_sorted, group_end, positives


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--scores',required=True)
    p.add_argument('--level',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--replicates',type=int,default=1000)
    p.add_argument('--seed',type=int,default=2026)
    a=p.parse_args()
    z=np.load(a.scores)
    y=z['target'].astype(np.uint8); ls=z['local']; fs=z['full']; d=z['dormant']; active=z['active']; skill=z['skill'].astype(int)
    rng=np.random.default_rng(a.seed)
    bins=[('0',active==0),('1',active==1),('2',active==2),('3-5',(active>=3)&(active<=5)),('6-11',(active>=6)&(active<=11)),('rare_dormant',active<=2)]
    rows=[]
    for label,bmask in bins:
        mask=d & bmask
        ym=y[mask]; l=ls[mask]; f=fs[mask]; sk=skill[mask]
        clusters, cluster_inverse=np.unique(sk, return_inverse=True)
        n_clusters=len(clusters)
        observed=float(average_precision_score(ym,f)-average_precision_score(ym,l))
        # Sampling n_clusters clusters with replacement is multinomial with equal probabilities.
        probs=np.full(n_clusters,1.0/n_clusters,dtype=np.float64)
        counts=np.vstack([rng.multinomial(n_clusters,probs) for _ in range(a.replicates)]).astype(np.int16)
        yl,cl,el,pl=_prepare(ym,l,cluster_inverse,n_clusters)
        yf,cf,ef,pf=_prepare(ym,f,cluster_inverse,n_clusters)
        local_ap=_ap_many(yl,cl,el,counts,pl)
        full_ap=_ap_many(yf,cf,ef,counts,pf)
        vals=full_ap-local_ap
        vals=vals[np.isfinite(vals)]
        lo,hi=np.quantile(vals,[.025,.975])
        rows.append(dict(level=a.level.upper(),active_months=label,n=int(mask.sum()),positive=int(ym.sum()),
                         clusters=n_clusters,observed=observed,ci_low=float(lo),ci_high=float(hi),
                         boot_mean=float(np.mean(vals)),replicates=int(len(vals))))
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(a.output,index=False)

if __name__=='__main__':
    main()
