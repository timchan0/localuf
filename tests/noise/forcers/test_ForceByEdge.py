import itertools
import re

import pytest

from localuf.noise import CircuitLevel
from localuf.noise.forcers import ForceByEdge
from localuf.type_aliases import Edge


def test_force_error(toy_cl: CircuitLevel, e_westmost: tuple[Edge, Edge]):
    fbe = ForceByEdge(toy_cl._EDGES)
    for weight in itertools.product((0, 1), repeat=2):
        error = fbe.force_error(weight)
        assert error == {e for e, w in zip(e_westmost, weight) if w}
    with pytest.raises(
        ValueError,
        match='Sample larger than population or is negative',
    ):
        fbe.force_error((2, 0))
    with pytest.raises(
        ValueError,
        match=re.escape("len(weight)=3 != 2=number of edge subsets")
    ):
        fbe.force_error((0, 0, 0))