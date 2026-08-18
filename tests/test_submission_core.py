import numpy as np
from mosaic.protocol import LOOKBACK,HORIZON,TRAIN_STARTS,VALIDATION_STARTS,TEST_STARTS

def test_protocol_blocks_do_not_overlap():
    blocks=[set(range(s,s+HORIZON)) for s in TEST_STARTS]
    assert all(blocks[i].isdisjoint(blocks[j]) for i in range(len(blocks)) for j in range(i+1,len(blocks)))
    assert max(TRAIN_STARTS)+HORIZON <= VALIDATION_STARTS[0]
    assert VALIDATION_STARTS[0]+HORIZON <= TEST_STARTS[0]

def test_rare_dormant_is_history_only():
    hist=np.array([[0,0,1,0,0,0,0,0,0,0,0,0],[0,1,0,1,0,0,0,0,0,0,0,0]],dtype=float)
    active=(hist>0).sum(1); dormant=hist[:,-1]==0; rare=dormant&(active<=2)
    assert rare.tolist()==[True,True]

def test_exclusive_context_arithmetic():
    focal=np.array([2.,1.,0.]); region=np.array([5.,4.,1.]); occupation=np.array([7.,2.,0.]); market=np.array([10.,8.,3.])
    np.testing.assert_array_equal(region-focal,[3.,3.,1.])
    np.testing.assert_array_equal(occupation-focal,[5.,1.,0.])
    np.testing.assert_array_equal(market-region-occupation+focal,[0.,3.,2.])
