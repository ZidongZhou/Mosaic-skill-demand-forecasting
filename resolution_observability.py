"""Compute recent-local-event-observability statistics used in Table 1 and Fig. 1a.

The four 2023 forecast origins are pooled. A is the number of positive focal
months in the preceding 12 months. All reported columns therefore use the same
forecast-origin population and the manuscript's explicit observability measure.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from mosaic.data import DemandLattice
from mosaic.protocol import LOOKBACK, TEST_STARTS


def level_stats(name, arr):
    vals=[]
    for start in TEST_STARTS:
        h=arr[..., start-LOOKBACK:start]
        vals.append((h>0).sum(axis=-1).reshape(-1))
    a=np.concatenate(vals)
    return dict(resolution=name, series=int(np.prod(arr.shape[:-1])),
                median_active_months_prev12=float(np.median(a)),
                share_A0_pct=float(100*np.mean(a==0)),
                share_Ale2_pct=float(100*np.mean(a<=2)),
                mean_active_months_prev12=float(np.mean(a)))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--l1',required=True); ap.add_argument('--l2',required=True)
    ap.add_argument('--out',default='frozen_results/resolution_observability_2023.csv'); args=ap.parse_args()
    l1=DemandLattice.from_npz(args.l1); l2=DemandLattice.from_npz(args.l2)
    rows=[level_stats('Market',l1.market), level_stats('Region',l1.region),
          level_stats('L1 occupation',l1.occupation), level_stats('L2 occupation',l2.occupation),
          level_stats('Region x L1',l1.bottom), level_stats('Region x L2',l2.bottom)]
    out=pd.DataFrame(rows); Path(args.out).parent.mkdir(parents=True,exist_ok=True); out.to_csv(args.out,index=False)
    print(out.to_string(index=False))
if __name__=='__main__': main()
