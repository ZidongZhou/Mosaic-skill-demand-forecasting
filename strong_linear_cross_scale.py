import numpy as np, pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
from mosaic.data import DemandLattice
from mosaic.protocol import LOOKBACK,HORIZON,TRAIN_STARTS,TEST_STARTS
from strong_matched_cross_scale import memory_features
import argparse
ap=argparse.ArgumentParser(); ap.add_argument('--lattice',required=True); ap.add_argument('--out',default='frozen_results/strong_linear_l1.csv'); args=ap.parse_args()
lat=DemandLattice.from_npz(args.lattice)
def features(start,full):
 b=lat.bottom[...,start-LOOKBACK:start].astype(np.float32); parts=[np.log1p(b).reshape(-1,LOOKBACK)]
 if full:
  reg=np.maximum(lat.region[:,:,start-LOOKBACK:start][:,:,None,:]-b,0)
  occ=np.maximum(lat.occupation[:,:,start-LOOKBACK:start][:,None,:,:]-b,0)
  market=lat.market[:,start-LOOKBACK:start][:,None,None,:]
  res=np.maximum(market-lat.region[:,:,start-LOOKBACK:start][:,:,None,:]-lat.occupation[:,:,start-LOOKBACK:start][:,None,:,:]+b,0)
  parts += [np.log1p(reg).reshape(-1,LOOKBACK),np.log1p(occ).reshape(-1,LOOKBACK),np.log1p(res).reshape(-1,LOOKBACK)]
 parts += [(b>0).sum(-1).reshape(-1,1).astype(np.float32),np.log1p(b[...,-1]).reshape(-1,1).astype(np.float32),memory_features(lat,start)]
 X=np.concatenate(parts,axis=1); y=(lat.bottom[...,start:start+HORIZON]>0).any(-1).reshape(-1).astype(np.uint8)
 return X,y,b

def run(full):
 Xs=[];ys=[]
 for s in TRAIN_STARTS:
  X,y,_=features(s,full); Xs.append(X);ys.append(y)
 X=np.concatenate(Xs); y=np.concatenate(ys); sc=StandardScaler(); X=sc.fit_transform(X)
 clf=SGDClassifier(loss='log_loss',penalty='l2',alpha=1e-4,class_weight='balanced',max_iter=100,tol=1e-4,random_state=2026,average=True,n_jobs=-1)
 clf.fit(X,y)
 ss=[];yy=[];rr=[]
 for s in TEST_STARTS:
  X,y,b=features(s,full); score=clf.decision_function(sc.transform(X)); a=(b>0).sum(-1).reshape(-1);d=(b[...,-1]==0).reshape(-1)
  ss.append(score);yy.append(y);rr.append(d&(a<=2))
 ss=np.concatenate(ss);yy=np.concatenate(yy);rr=np.concatenate(rr)
 return average_precision_score(yy[rr],ss[rr])
local=run(False); full=run(True)
pd.DataFrame([{'variant':'Strong logistic-linear','information_set':'Strong local','rare_dormant_ap':local},{'variant':'Strong logistic-linear','information_set':'Strong local + cross-scale','rare_dormant_ap':full}]).to_csv(args.out,index=False)
print('local',local,'full',full,'gain',full-local)
