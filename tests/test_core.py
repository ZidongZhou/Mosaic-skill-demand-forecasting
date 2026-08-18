import numpy as np
import torch
import pytest

from mosaic.data import DemandLattice
from mosaic.model import MOSAIC


def toy_lattice():
    rng=np.random.default_rng(4)
    bottom=rng.poisson(0.8,size=(5,2,3,18)).astype(np.float32)
    return DemandLattice(
        market=bottom.sum((1,2)),
        region=bottom.sum(2),
        occupation=bottom.sum(1),
        bottom=bottom,
        months=[f"2022-{m:02d}" for m in range(1,13)]+[f"2023-{m:02d}" for m in range(1,7)],
        occupation_level='r1',
    )


def test_lattice_coherence():
    l=toy_lattice(); l.validate()
    assert np.max(np.abs(l.bottom.sum((1,2))-l.market))==0


def test_model_outputs_coherent_nonnegative_forecasts():
    l=toy_lattice()
    m=MOSAIC(12,3,l.n_regions,l.n_occupations)
    out=m(torch.from_numpy(l.market[:2,:12]),torch.from_numpy(l.region[:2,:,:12]),torch.from_numpy(l.occupation[:2,:,:12]),torch.from_numpy(l.bottom[:2,:,:,:12]))
    assert torch.all(out.bottom_mean>=0)
    assert torch.max(torch.abs(out.bottom_mean.sum((1,2))-out.market_mean)).item()==0



def test_lattice_rejects_negative_values():
    l = toy_lattice()
    l.bottom[0, 0, 0, 0] = -1
    with pytest.raises(ValueError, match="Negative values"):
        l.validate()


def test_lattice_rejects_nonfinite_values():
    l = toy_lattice()
    l.market[0, 0] = np.nan
    with pytest.raises(ValueError, match="Non-finite values"):
        l.validate()
