#!/usr/bin/env python3
"""Paired skill-cluster bootstrap for L1 context attribution contrasts."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

def main():
    p=argparse.ArgumentParser();p.add_argument('--scores',required=True);p.add_argument('--output',required=True);p.add_argument('--replicates',type=int,default=500);p.add_argument('--seed',type=int,default=2026)
    a=p.parse_args();z=np.load(a.scores); y=z['target'];skill=z['skill'].astype(int);unique=np.unique(skill);rng=np.random.default_rng(a.seed)
    ctx=['local','region','occupation','market','region_occupation','occupation_market','region_market','full']
    point={k:float(average_precision_score(y,z[k])) for k in ctx}; boots={k:[] for k in ctx}
    for _ in range(a.replicates):
        sampled=rng.choice(unique,size=len(unique),replace=True); counts=np.bincount(sampled,minlength=int(unique.max())+1);w=counts[skill]; keep=w>0
        for k in ctx: boots[k].append(float(average_precision_score(y[keep],z[k][keep],sample_weight=w[keep])))
    pairs={'region_single_gain':('region','local'),'occupation_single_gain':('occupation','local'),'market_single_gain':('market','local'),'drop_region':('full','occupation_market'),'drop_occupation':('full','region_market'),'drop_market':('full','region_occupation'),'full_gain_vs_local':('full','local')}
    rows=[]
    for name,(x0,x1) in pairs.items():
        arr=np.asarray(boots[x0])-np.asarray(boots[x1]);lo,hi=np.quantile(arr,[.025,.975]);rows.append(dict(contrast=name,estimate=point[x0]-point[x1],ci_low=float(lo),ci_high=float(hi)))
    Path(a.output).parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(a.output,index=False)
if __name__=='__main__':main()
