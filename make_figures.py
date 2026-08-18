from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Liberation Sans','DejaVu Sans'],
                     'font.size':9,'axes.labelsize':9,'xtick.labelsize':8.5,'ytick.labelsize':8.5,
                     'legend.fontsize':8.5})

def save(fig,out,stem):
    out.mkdir(parents=True,exist_ok=True)
    fig.savefig(out/f'{stem}.png',dpi=300,bbox_inches='tight')
    fig.savefig(out/f'{stem}.eps',format='eps',bbox_inches='tight')
    fig.savefig(out/f'{stem}.pdf',bbox_inches='tight')
    plt.close(fig)

def col(df,*names):
    lower={c.lower():c for c in df.columns}
    for n in names:
        if n.lower() in lower:return lower[n.lower()]
    raise KeyError((names,df.columns.tolist()))

def fig1(root,out):
    res=pd.read_csv(root/'resolution_observability_2023.csv')
    l1=pd.read_csv(root/'strong_matched_gradient_l1.csv'); l2=pd.read_csv(root/'strong_matched_gradient_l2.csv')
    b1=pd.read_csv(root/'strong_bootstrap_l1.csv'); b2=pd.read_csv(root/'strong_bootstrap_l2.csv')
    fig=plt.figure(figsize=(7.15,5.8))
    gs=fig.add_gridspec(2,2,height_ratios=[1,1.05],hspace=.42,wspace=.35)
    ax=fig.add_subplot(gs[0,0])
    x=np.arange(len(res)); vals=res['share_Ale2_pct'].values
    ax.plot(x,vals,marker='o',linewidth=1.6,color='0.15')
    ax.set_xticks(x); ax.set_xticklabels(['Market','Region','L1 occ.','L2 occ.','Region x L1','Region x L2'],rotation=28,ha='right')
    ax.set_ylabel('Forecast origins with A <= 2 (%)')
    ax.set_ylim(0,90); ax.grid(axis='y',alpha=.25)
    ax.text(-.14,1.06,'a',transform=ax.transAxes,fontweight='bold',fontsize=11)
    # Mosaic lattice schematic
    ax=fig.add_subplot(gs[0,1]); ax.set_aspect('equal'); n=5; focal=(2,2)
    for r in range(n):
      for c in range(n):
        if (r,c)==focal: fc='0.82'
        elif r==focal[0]: fc='0.90'
        elif c==focal[1]: fc='0.74'
        else: fc='0.96'
        hatch='' if (r,c)==focal else ('///' if r==focal[0] else ('\\\\' if c==focal[1] else '..'))
        ax.add_patch(Rectangle((c,n-1-r),1,1,facecolor=fc,edgecolor='0.25',linewidth=.7,hatch=hatch))
    ax.text(focal[1]+.5,n-1-focal[0]+.5,'focal',ha='center',va='center',fontsize=8,fontweight='bold')
    ax.annotate('regional',xy=(4.5,n-1-focal[0]+.5),xytext=(5.35,n-1-focal[0]+.5),ha='left',va='center',arrowprops=dict(arrowstyle='-',lw=.8))
    ax.annotate('occupational',xy=(focal[1]+.5,4.5),xytext=(focal[1]+.5,5.35),ha='center',va='bottom',arrowprops=dict(arrowstyle='-',lw=.8))
    ax.text(4.85,.25,'residual\nmarket',ha='right',va='bottom',fontsize=8)
    ax.set_xlim(0,5.8); ax.set_ylim(0,5.8); ax.set_xticks(np.arange(n)+.5); ax.set_xticklabels(['O1','O2','O3','O4','O5']); ax.set_yticks(np.arange(n)+.5); ax.set_yticklabels(['R5','R4','R3','R2','R1'])
    ax.set_xlabel('Occupation'); ax.set_ylabel('Region');
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.text(-.14,1.06,'b',transform=ax.transAxes,fontweight='bold',fontsize=11)
    # Gradient panel across full width
    ax=fig.add_subplot(gs[1,:])
    labels=['0','1','2','3-5','6-11']; x=np.arange(5)
    def vals(df):
      g=col(df,'gain'); return df[g].values.astype(float)
    # bootstrap files have row labels; use their CI if matching, otherwise frozen CI cols in gradient
    def cis(df):
      lo=next((c for c in df.columns if '2.5' in c or 'lower' in c.lower()),None)
      hi=next((c for c in df.columns if '97.5' in c or 'upper' in c.lower()),None)
      if lo and hi:return df[lo].values.astype(float),df[hi].values.astype(float)
      return None
    for df,boot,lab,marker,off,color,ls in [(l1,b1,'L1 broad occupation','o',-.045,'0.15','-'),(l2,b2,'L2 detailed occupation','s',.045,'0.55','--')]:
      y=vals(df)
      bb=boot[boot['active_months'].astype(str).isin(labels)].copy()
      bb['active_months']=pd.Categorical(bb['active_months'].astype(str),categories=labels,ordered=True)
      bb=bb.sort_values('active_months')
      err=np.vstack([y-bb['ci_low'].values,bb['ci_high'].values-y])
      ax.errorbar(x+off,y,yerr=err,marker=marker,linewidth=1.5,capsize=2.5,label=lab,color=color,linestyle=ls)
    ax.axhline(0,linewidth=.7,color='0.35')
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_xlabel('Active months in preceding 12 months, A'); ax.set_ylabel('Gain in activation AP')
    ax.grid(axis='y',alpha=.25); ax.legend(frameon=False,ncol=2,loc='upper right')
    ax.text(-.055,1.06,'c',transform=ax.transAxes,fontweight='bold',fontsize=11)
    fig.subplots_adjust(top=.97,bottom=.11,left=.09,right=.98)
    save(fig,out,'Fig1')

def fig2(root,out):
    ctx=pd.read_csv(root/'strong_context_l1.csv'); boot=pd.read_csv(root/'strong_context_bootstrap_l1.csv'); q1=pd.read_csv(root/'strong_quarterly_l1.csv'); q2=pd.read_csv(root/'strong_quarterly_l2.csv')
    c=ctx.set_index('context')
    single={
      'Regional':float(c.loc['region','gain_vs_local']),
      'Occupational':float(c.loc['occupation','gain_vs_local']),
      'Residual market':float(c.loc['market','gain_vs_local']),
    }
    drops={
      'Regional':float(c.loc['occupation_market','drop_from_full']),
      'Occupational':float(c.loc['region_market','drop_from_full']),
      'Residual market':float(c.loc['region_occupation','drop_from_full']),
    }
    # CI values from bootstrap table: locate rows by labels and use lower/upper.
    def get_ci(kind,name,est):
      tok={'Regional':'region','Occupational':'occupation','Residual market':'market'}[name]
      contrast=(f'drop_{tok}' if kind=='drop' else f'{tok}_single_gain')
      sub=boot[boot['contrast'].astype(str).str.lower().eq(contrast)]
      if len(sub): return float(sub.iloc[0]['ci_low']),float(sub.iloc[0]['ci_high'])
      return est,est
    fig,axs=plt.subplots(1,2,figsize=(7.15,3.15))
    ax=axs[0]; names=list(single); yy=np.arange(len(names))[::-1]
    for i,name in enumerate(names):
      y=yy[i]; e=single[name]; lo,hi=get_ci('single',name,e)
      ax.errorbar(e,y+.10,xerr=[[e-lo],[hi-e]],fmt='o',capsize=2.5,label='Added alone' if i==0 else None,color='0.15')
      d=drops[name]; lo2,hi2=get_ci('drop',name,d)
      ax.errorbar(d,y-.10,xerr=[[d-lo2],[hi2-d]],fmt='s',capsize=2.5,label='Drop when removed' if i==0 else None,color='0.58')
    ax.axvline(0,linewidth=.7,color='0.35'); ax.set_yticks(yy); ax.set_yticklabels(names); ax.set_xlabel('Marginal contribution to AP'); ax.grid(axis='x',alpha=.25); ax.legend(frameon=False,fontsize=8,loc='upper left'); ax.text(-.16,1.04,'a',transform=ax.transAxes,fontweight='bold',fontsize=11)
    ax=axs[1]
    for df,lab,marker,color,ls in [(q1,'L1 broad occupation','o','0.15','-'),(q2,'L2 detailed occupation','s','0.55','--')]:
      # find gain column and use row order
      gc=next(c for c in df.columns if 'gain' in c.lower()); y=df[gc].values.astype(float)
      ax.plot(np.arange(4),y,marker=marker,linewidth=1.5,label=lab,color=color,linestyle=ls)
    ax.axhline(0,linewidth=.7,color='0.35'); ax.set_xticks(range(4)); ax.set_xticklabels(['Q1','Q2','Q3','Q4']); ax.set_xlabel('2023 forecast block'); ax.set_ylabel('Full minus strong-local AP'); ax.grid(axis='y',alpha=.25); ax.legend(frameon=False,fontsize=8); ax.text(-.16,1.04,'b',transform=ax.transAxes,fontweight='bold',fontsize=11)
    fig.tight_layout(); save(fig,out,'Fig2')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--results',default='frozen_results'); ap.add_argument('--output-dir',default='reproduced_figures'); args=ap.parse_args()
    root=Path(args.results); out=Path(args.output_dir)
    fig1(root,out); fig2(root,out)
    print('Generated Fig1 and Fig2 in',out)
if __name__=='__main__': main()
