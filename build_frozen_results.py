"""Check compact frozen-result tables and regenerate manuscript figures separately.

SHA256SUMS.txt verifies the pristine distributed archive. Figure regeneration is
written to reproduced_figures/ so vector-file metadata cannot alter archived hashes.
"""
from pathlib import Path
import subprocess, sys
import pandas as pd
ROOT=Path(__file__).resolve().parent
FR=ROOT/'frozen_results'
required=sorted(FR.glob('*.csv'))
for p in required:
    df=pd.read_csv(p)
    if df.empty: raise RuntimeError(f'Empty frozen result: {p.name}')
print(f'Checked {len(required)} compact result tables for readability.')
out=ROOT/'reproduced_figures'
out.mkdir(exist_ok=True)
subprocess.run([sys.executable,str(ROOT/'make_figures.py'),'--results',str(FR),'--output-dir',str(out)],check=True)
print('Regenerated Fig1 and Fig2 in',out)
