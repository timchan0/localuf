import numpy as np
import pytest

from localuf import Surface
from localuf.noise import CircuitLevel
from localuf.type_aliases import Edge, FourInts


@pytest.mark.parametrize('d', range(3, 7, 2), ids=lambda d: f'd{d}')
def test_init(d: int):

    sf1 = Surface(d, 'circuit-level', _merge_redundant_edges=False)
    sf2 = Surface(d, 'circuit-level')
    sf2_edges = set(sf2.EDGES)

    # manually merge edges of sf1
    dc = {}
    for m, es in sf1.NOISE._EDGES.items(): # type: ignore
        for e in es:
            if e not in sf2_edges:
                sub = sf1._substitute(e)
                if sub in dc:
                    dc[sub].append(m)
                else:
                    dc[sub] = [m]
            else:
                if e in dc:
                    dc[e].append(m)
                else:
                    dc[e] = [m]
    edges = {e: sum(np.array(m) for m in ls) for e, ls in dc.items()}

    for e, m in edges.items():
        key: FourInts = tuple(m) # type: ignore
        assert e in sf2.NOISE._EDGES[key] # type: ignore


def test_EDGES(toy_cl: CircuitLevel, e_westmost: tuple[Edge, Edge]):
    e0, e1 = e_westmost
    assert len(toy_cl._EDGES) == 2
    assert toy_cl._EDGES[1, 0, 2, 0] == [e0]
    assert toy_cl._EDGES[5, 0, 3, 0] == [e1]