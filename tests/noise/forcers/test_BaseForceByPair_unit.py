from math import prod

import numpy as np
import pytest
from scipy.stats import binom

from localuf.noise.forcers import _BaseForceByPair
from localuf.type_aliases import FourInts, FourFloats

@pytest.fixture
def base_force_by_pair():
    toy_m: FourInts = tuple(range(4)) # type: ignore
    bfbp = _BaseForceByPair(edges={
        toy_m: [((0, -1, 0), (0, 0, 0)), ((0, 0, 0), (0, 0, 1))],
    })
    bfbp.PAIR_POPULATIONS = (
        bfbp._EDGES[toy_m],
        bfbp._EDGES[toy_m],
    )
    return bfbp


def test_subset_probability(base_force_by_pair: _BaseForceByPair):
    weights = ((0, 0), (1, 0), (0, 1), (1, 1))
    pi: FourFloats = (0.1, 0.2, 0.3, 0.4)
    subset_prob: np.ndarray = base_force_by_pair.subset_probability(
        weights=weights,
        pi=pi,
    ) # type: ignore
    assert (subset_prob == np.array([
        prod(binom.pmf(w, 2, p) for w, p in zip(weight, pi))
        for weight in weights
    ])).all()