#!/usr/bin/env python3
"""Complete L1 context attribution under the matched strong-local specification."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import average_precision_score
from mosaic.data import DemandLattice
from mosaic.protocol import LOOKBACK,HORIZON,TRAIN_STARTS,TEST_STARTS
from strong_matched_cross_scale import memory_features

CONTEXTS={
 'local':(), 'region':('region',), 'occupation':('occupation',), 'market':('market',),
 'region_occupation':('region','occupation'), 'occupation_market':('occupation','market'),
 'region_market':('region','market'), 'full':('region','occupation','market')}

def feat(lat,start,name):
    b=lat.bottom[...,start-LOOKBACK:start].astype(np.float32); parts=[np.log1p(b).reshape(-1,LOOKBACK)]
    use=CONTEXTS[name]
    if 'region' in use:
        x=np.maximum(lat.region[:,:,start-LOOKBACK:start][:,:,None,:]-b,0); parts.append(np.log1p(x).reshape(-1,LOOKBACK))
    if 'occupation' in use:
        x=np.maximum(lat.occupation[:,:,start-LOOKBACK:start][:,None,:,:]-b,0); parts.append(np.log1p(x).reshape(-1,LOOKBACK))
    if 'market' in use:
        M=lat.market[:,start-LOOKBACK:start][:,None,None,:]; R=lat.region[:,:,start-LOOKBACK:start][:,:,None,:]; O=lat.occupation[:,:,start-LOOKBACK:start][:,None,:,:]
        x=np.maximum(M-R-O+b,0); parts.append(np.log1p(x).reshape(-1,LOOKBACK))
    parts.extend([(b>0).sum(-1).reshape(-1,1).astype(np.float32),np.log1p(b[...,-1]).reshape(-1,1).astype(np.float32),memory_features(lat,start)])
    X=np.concatenate(parts,1); y=(lat.bottom[...,start:start+HORIZON]>0).any(-1).reshape(-1).astype(np.uint8)
    return X,y,b

def main():
    p=argparse.ArgumentParser(); p.add_argument('--lattice',required=True); p.add_argument('--output',required=True); p.add_argument('--scores'); p.add_argument('--n-jobs',type=int,default=4)
    a=p.parse_args(); lat=DemandLattice.from_npz(a.lattice); rows=[]; saved={}; first=True; ncell=lat.n_regions*lat.n_occupations; skill_ids=np.repeat(np.arange(lat.n_skills),ncell)
    for ctx in CONTEXTS:
        Xs=[];ys=[]
        for st in TRAIN_STARTS:
            X,y,_=feat(lat,st,ctx); Xs.append(X);ys.append(y)
        m=lgb.LGBMClassifier(n_estimators=45,num_leaves=25,learning_rate=.07,subsample=.8,subsample_freq=1,colsample_bytree=.8,reg_lambda=1.0,class_weight='balanced',random_state=2026,n_jobs=a.n_jobs,verbosity=-1)
        m.fit(np.concatenate(Xs),np.concatenate(ys))
        ss=[];tt=[];mm=[];sk=[]
        for st in TEST_STARTS:
            X,y,b=feat(lat,st,ctx); ss.append(m.predict_proba(X)[:,1]);tt.append(y);act=(b>0).sum(-1).reshape(-1);d=(b[...,-1]==0).reshape(-1);mm.append(d&(act<=2));sk.append(skill_ids)
        s=np.concatenate(ss); yall=np.concatenate(tt); mask=np.concatenate(mm); skills=np.concatenate(sk)
        score=s[mask].astype(np.float32); target=yall[mask].astype(np.uint8); skills=skills[mask].astype(np.int16)
        val=float(average_precision_score(target,score)); rows.append(dict(context=ctx,ap=val,n=int(mask.sum()),positive=int(target.sum())))
        saved[ctx]=score
        if first: saved['target']=target;saved['skill']=skills; first=False
        else: assert np.array_equal(saved['target'],target) and np.array_equal(saved['skill'],skills)
    df=pd.DataFrame(rows); full=float(df.loc[df.context=='full','ap'].iloc[0]); local=float(df.loc[df.context=='local','ap'].iloc[0]); df['gain_vs_local']=df.ap-local;df['drop_from_full']=full-df.ap
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); df.to_csv(a.output,index=False)
    if a.scores: np.savez_compressed(a.scores,**saved)

if __name__=='__main__': main()
