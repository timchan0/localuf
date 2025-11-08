from localuf import Repetition, Surface


def test_make_error(
        sf3F: Surface,
        sf3T: Surface,
        rp3_forward: Repetition,
        rp3_frugal: Repetition,
):
    assert sf3F.make_error(0) == set()
    assert sf3F.make_error(1) == set(sf3F.EDGES)
    assert sf3T.make_error(0) == set()
    assert sf3T.make_error(1) == set(sf3T.EDGES)
    assert rp3_forward.make_error(0) == set()
    assert rp3_forward.make_error(1) == set(rp3_forward.NOISE.EDGES) # type: ignore
    assert rp3_frugal.make_error(0) == set()
    assert rp3_frugal.make_error(1) == set(rp3_frugal.NOISE.EDGES) # type: ignore


def test_INCIDENT_EDGES_circuit_level():
    sf = Surface(3, 'circuit-level', _merge_redundant_edges=False)
    # boundary node
    assert sf.INCIDENT_EDGES[0, -1, 0] == {
        # cardinals
        ((0, -1, 0), (0, 0, 0)),
        # diagonals
        ((0, -1, 0), (0, 0, 1)),
        ((0, -1, 0), (1, 0, 1)),
    }
    # degree-8 node
    assert sf.INCIDENT_EDGES[0, 1, 1] == {
        # cardinals
        ((0, 1, 0), (0, 1, 1)),
        ((0, 0, 1), (0, 1, 1)),
        ((0, 1, 1), (0, 2, 1)),
        ((0, 1, 1), (1, 1, 1)),
        ((0, 1, 1), (0, 1, 2)),
        # diagonals
        ((0, 0, 0), (0, 1, 1)),
        ((0, 1, 1), (1, 1, 0)),
        ((0, 1, 1), (0, 2, 2)),
        ((0, 1, 1), (1, 2, 2)),
    }
    # weight-4 stabilizer
    assert sf.INCIDENT_EDGES[1, 1, 1] == {
        # cardinals
        ((1, 1, 0), (1, 1, 1)),
        ((1, 0, 1), (1, 1, 1)),
        ((1, 1, 1), (1, 2, 1)),
        ((0, 1, 1), (1, 1, 1)),
        ((1, 1, 1), (2, 1, 1)),
        ((1, 1, 1), (1, 1, 2)),
        # diagonals
        # SD
        ((0, 1, 2), (1, 1, 1)),
        ((1, 1, 1), (2, 1, 0)),
        # EU
        ((1, 0, 0), (1, 1, 1)),
        ((1, 1, 1), (1, 2, 2)),
        # SEU
        ((0, 0, 0), (1, 1, 1)),
        ((1, 1, 1), (2, 2, 2)),
    }


def test_INCIDENT_EDGES_circuit_level_merge():
    sf = Surface(3, 'circuit-level')
    # boundary node
    assert sf.INCIDENT_EDGES[0, -1, 0] == {
        # cardinals
        ((0, -1, 0), (0, 0, 0)),
    }
    # degree-7 node
    assert sf.INCIDENT_EDGES[0, 1, 1] == {
        # cardinals
        ((0, 1, 0), (0, 1, 1)),
        ((0, 0, 1), (0, 1, 1)),
        ((0, 1, 1), (0, 2, 1)),
        ((0, 1, 1), (1, 1, 1)),
        ((0, 1, 1), (0, 1, 2)),
        # diagonals
        ((0, 0, 0), (0, 1, 1)),
        ((0, 1, 1), (1, 1, 0)),
    }
    # weight-4 stabilizer
    assert sf.INCIDENT_EDGES[1, 1, 1] == {
        # cardinals
        ((1, 1, 0), (1, 1, 1)),
        ((1, 0, 1), (1, 1, 1)),
        ((1, 1, 1), (1, 2, 1)),
        ((0, 1, 1), (1, 1, 1)),
        ((1, 1, 1), (2, 1, 1)),
        ((1, 1, 1), (1, 1, 2)),
        # diagonals
        # SD
        ((0, 1, 2), (1, 1, 1)),
        ((1, 1, 1), (2, 1, 0)),
        # EU
        ((1, 0, 0), (1, 1, 1)),
        # SEU
        ((0, 0, 0), (1, 1, 1)),
    }