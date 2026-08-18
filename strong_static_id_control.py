import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import average_precision_score
from mosaic.data import DemandLattice
from mosaic.protocol import LOOKBACK,HORIZON,TRAIN_STARTS,TEST_STARTS
from strong_matched_cross_scale import memory_features
import argparse
ap=argparse.ArgumentParser(); ap.add_argument('--lattice',required=True); ap.add_argument('--out',default='frozen_results/strong_static_id_control_l1.csv'); args=ap.parse_args()
lat=DemandLattice.from_npz(args.lattice)
S,R,O=lat.n_skills,lat.n_regions,lat.n_occupations
ncell=R*O
ids=np.stack([
    np.repeat(np.arange(S,dtype=np.int32),ncell),
    np.tile(np.repeat(np.arange(R,dtype=np.int32),O),S),
    np.tile(np.arange(O,dtype=np.int32),S*R)
],axis=1)
def feat(start,cross):
    b=lat.bottom[...,start-LOOKBACK:start].astype(np.float32)
    parts=[np.log1p(b).reshape(-1,LOOKBACK)]
    if cross:
      reg=np.maximum(lat.region[:,:,start-LOOKBACK:start][:,:,None,:]-b,0)
      occ=np.maximum(lat.occupation[:,:,start-LOOKBACK:start][:,None,:,:]-b,0)
      market=lat.market[:,start-LOOKBACK:start][:,None,None,:]
      res=np.maximum(market-lat.region[:,:,start-LOOKBACK:start][:,:,None,:]-lat.occupation[:,:,start-LOOKBACK:start][:,None,:,:]+b,0)
      parts += [np.log1p(reg).reshape(-1,LOOKBACK),np.log1p(occ).reshape(-1,LOOKBACK),np.log1p(res).reshape(-1,LOOKBACK)]
    parts += [(b>0).sum(-1).reshape(-1,1).astype(np.float32),np.log1p(b[...,-1]).reshape(-1,1).astype(np.float32),memory_features(lat,start),ids]
    X=np.concatenate(parts,axis=1)
    y=(lat.bottom[...,start:start+HORIZON]>0).any(-1).reshape(-1).astype(np.uint8)
    return X,y,b

def fit(cross):
  Xs=[];ys=[]
  for s in TRAIN_STARTS:
    X,y,_=feat(s,cross);Xs.append(X);ys.append(y)
  X=np.concatenate(Xs); y=np.concatenate(ys)
  cat=[X.shape[1]-3,X.shape[1]-2,X.shape[1]-1]
  m=lgb.LGBMClassifier(n_estimators=45,num_leaves=25,learning_rate=.07,subsample=.8,subsample_freq=1,colsample_bytree=.8,reg_lambda=1.,class_weight='balanced',random_state=2026,n_jobs=4,verbosity=-1)
  m.fit(X,y,categorical_feature=cat)
  return m
ml=fit(False); mf=fit(True)
sl=[];sf=[];yy=[];dd=[];aa=[]
for s in TEST_STARTS:
 X,y,b=feat(s,False); sl.append(ml.predict_proba(X)[:,1]); yy.append(y); dd.append((b[...,-1]==0).reshape(-1)); aa.append((b>0).sum(-1).reshape(-1))
 X2,y2,b2=feat(s,True); sf.append(mf.predict_proba(X2)[:,1]); assert np.array_equal(y,y2)
sl=np.concatenate(sl);sf=np.concatenate(sf);y=np.concatenate(yy);d=np.concatenate(dd);a=np.concatenate(aa)
rare=d&(a<=2)
la=average_precision_score(y[rare],sl[rare]); fa=average_precision_score(y[rare],sf[rare])
rows=[{'subset':'rare-dormant','strong_local_ids_ap':la,'strong_local_ids_cross_scale_ap':fa,'gain':fa-la}]
print('strong+IDs',la,fa,fa-la,'n',rare.sum())
for label,mask in [('A=0',a==0),('A=1',a==1),('A=2',a==2),('A=3-5',(a>=3)&(a<=5)),('A=6-11',(a>=6)&(a<=11))]:
 m=mask&d
 lbin=average_precision_score(y[m],sl[m]); fbin=average_precision_score(y[m],sf[m]); print(label,lbin,fbin,fbin-lbin)
 rows.append({'subset':label,'strong_local_ids_ap':lbin,'strong_local_ids_cross_scale_ap':fbin,'gain':fbin-lbin})
pd.DataFrame(rows).to_csv(args.out,index=False)
