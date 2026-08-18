#!/usr/bin/env python3
from __future__ import annotations
import argparse
from mosaic.data import DemandLattice

p=argparse.ArgumentParser()
p.add_argument('--dataset',required=True)
p.add_argument('--level',choices=['r1','r2'],required=True)
p.add_argument('--output',required=True)
a=p.parse_args()
l=DemandLattice.from_job_sdf(a.dataset,a.level)
l.to_npz(a.output)
print(f'saved {a.output}: skills={l.n_skills}, regions={l.n_regions}, occupations={l.n_occupations}, months={l.n_periods}')
