import pytest
import networkx as nx

from localuf import Surface
from localuf.type_aliases import Node
from localuf.decoders.luf import LUF
from localuf.constants import Growth
from localuf.decoders.luf.constants import Stage

@pytest.mark.parametrize("luf, syndrome", [
    ("astris5F", "syndrome5F"),
    ("astris5T", "syndrome5T"),
])
def test_not_any_busy_after_growing(luf: LUF, syndrome: set[Node], request):
    luf = request.getfixturevalue(luf)
    syndrome = request.getfixturevalue(syndrome)
    luf.NODES.load(syndrome)
    luf.NODES.advance()
    assert not any(node.busy for node in luf.NODES.dc.values())

@pytest.mark.parametrize("luf, syndrome", [
    ("astris5F", "syndrome5F"),
    ("astris5T", "syndrome5T"),
])
def test_not_any_busy_after_presyncing(luf: LUF, syndrome: set[Node], request):
    luf = request.getfixturevalue(luf)
    syndrome = request.getfixturevalue(syndrome)
    luf.NODES.load(syndrome)
    while not luf.NODES.valid:
        check = luf.CONTROLLER.stage is Stage.PRESYNCING
        luf._advance()
        if check:
            assert not any(node.busy for node in luf.NODES.dc.values())
            assert luf.CONTROLLER.stage is Stage.SYNCING

def test_presyncing_astris(astris: LUF):
    for _ in range(2):
        astris._advance()
    # controller stage now PRESYNC
    for node in astris.NODES.dc.values():
        node.active = True
    astris._advance()
    assert all(not node.active for node in astris.NODES.dc.values())

def test_presyncing_actis(actis: LUF):
    actis.CONTROLLER.stage = Stage.PRESYNCING
    for node in actis.NODES.dc.values():
        node.stage = Stage.PRESYNCING # type: ignore
        node.active = True
    actis._advance()
    assert all(not node.active for node in actis.NODES.dc.values())

def test_validate_single_isolated_error(astris3F: LUF, actis3Fu: LUF):
    syndrome = {(0, 0), (0, 1)}
    erasure = {((0, 0), (0, 1))}

    n_steps = astris3F.validate(syndrome)
    assert n_steps == 6
    assert astris3F.erasure == erasure

    n_steps = actis3Fu.validate(syndrome)
    assert n_steps == 28
    assert actis3Fu.erasure == erasure

def test_validate_vdash_shape(astris3F: LUF, actis3Fu: LUF):
    syndrome = {
        (0, 0),
        (1, 0),
        (1, 1),
        (2, 0),
    }
    erasure = {
        ((0, 0), (1, 0)),
        ((1, 0), (1, 1)),
        ((1, 0), (2, 0)),
    }

    n_steps = astris3F.validate(syndrome)
    assert n_steps == 6
    assert astris3F.erasure == erasure

    n_steps = actis3Fu.validate(syndrome)
    assert n_steps == 29
    assert actis3Fu.erasure == erasure

def test_validate_sixpack(
        sf3F: Surface,
        astris3F: LUF,
        actis3Fu: LUF,
):
    syndrome = {v for v in sf3F.NODES if not sf3F.is_boundary(v)}
    erasure = {e for e in sf3F.EDGES if not any(
        sf3F.is_boundary(v) for v in e
    )}

    n_steps = astris3F.validate(syndrome)
    assert n_steps == 8
    assert astris3F.erasure == erasure

    n_steps = actis3Fu.validate(syndrome)
    assert n_steps == 31
    assert actis3Fu.erasure == erasure

def test_validate_perp_shape(astris5F: LUF, actis5Fu: LUF):
    syndrome = {
        (0, 1),
        (1, 0),
        (1, 1),
        (1, 2),
    }
    erasure = {
        ((0, 1), (1, 1)),
        ((1, 0), (1, 1)),
        ((1, 1), (1, 2)),
    }

    n_steps = astris5F.validate(syndrome)
    assert n_steps == 8
    assert astris5F.erasure == erasure

    n_steps = actis5Fu.validate(syndrome)
    assert n_steps == 43
    assert actis5Fu.erasure == erasure

def test_validate_diagonal(sf5F: Surface):
    luf = LUF(sf5F)
    syndrome = {
        (1, 1),
        (2, 2),
    }
    n_steps = luf.validate(syndrome)
    assert n_steps == 12
    assert luf.erasure == {
        ((0, 1), (1, 1)),
        ((1, 0), (1, 1)),
        ((1, 1), (1, 2)),
        ((1, 1), (2, 1)),
        ((1, 2), (2, 2)),
        ((2, 1), (2, 2)),
        ((2, 2), (2, 3)),
        ((2, 2), (3, 2)),
    }

def test_validate_J_shape(sf5F: Surface):
    luf = LUF(sf5F)
    syndrome = {
        (0, 1),
        (0, 2),
        (1, 2),
        (2, 2),
        (3, 0),
        (3, 1),
        (3, 2),
    }
    n_steps = luf.validate(syndrome)
    assert n_steps == 36
    assert luf.erasure == {
        ((0, 0), (0, 1)),
        ((0, 1), (0, 2)),
        ((0, 1), (1, 1)),
        ((0, 2), (0, 3)),
        ((0, 2), (1, 2)),
        ((1, 1), (1, 2)),
        ((1, 2), (1, 3)),
        ((1, 2), (2, 2)),
        ((2, 0), (3, 0)),
        ((2, 1), (2, 2)),
        ((2, 1), (3, 1)),
        ((2, 2), (2, 3)),
        ((2, 2), (3, 2)),
        ((3, -1), (3, 0)),
        ((3, 0), (3, 1)),
        ((3, 0), (4, 0)),
        ((3, 1), (3, 2)),
        ((3, 1), (4, 1)),
        ((3, 2), (3, 3)),
        ((3, 2), (4, 2))
    }

def test_cid_and_active_causality(sf5F: Surface):
    """Ensure causality i.e. info travels at 1 edge per timestep.
    
    Specifically, info is `cid` & `active`.
    This is ensured via `next_cid` & `busy`, respectively.
    """
    luf = LUF(sf5F)
    syndrome_3_row = {
        (0, 0),
        (0, 1),
        (0, 2),
    }
    n_steps = luf.validate(syndrome_3_row)
    assert n_steps == 17
    assert luf.erasure == {
        ((0, -1), (0, 0)),
        ((0, 0), (0, 1)),
        ((0, 0), (1, 0)),
        ((0, 1), (0, 2)),
        ((0, 1), (1, 1)),
        ((0, 2), (0, 3)),
        ((0, 2), (1, 2))
    }

def test_anyon_causality(sf5F: Surface):
    """Ensure causality of anyon info.
    
    Ensured via `next_anyon`.
    Cluster touches both boundaries."""
    luf = LUF(sf5F)
    syndrome_rightharpoonup = {
        (0, 3),
        (1, 0),
        (1, 1),
        (1, 2),
        (1, 3),
    }
    n_steps = luf.validate(syndrome_rightharpoonup)
    assert n_steps == 26
    assert luf.erasure == {
        ((0, 0), (1, 0)),
        ((0, 1), (1, 1)),
        ((0, 2), (0, 3)),
        ((0, 2), (1, 2)),
        ((0, 3), (0, 4)),
        ((0, 3), (1, 3)),
        ((1, -1), (1, 0)),
        ((1, 0), (1, 1)),
        ((1, 0), (2, 0)),
        ((1, 1), (1, 2)),
        ((1, 1), (2, 1)),
        ((1, 2), (1, 3)),
        ((1, 2), (2, 2)),
        ((1, 3), (1, 4)),
        ((1, 3), (2, 3))
    }

def test_busy_between_syncing(sf5F: Surface, syndrome5F: set[Node]):
    """Ensure `busy` from one syncing stage affects not `active` for next.

    If `busy` remains `True` from previous syncing stage to next,
    upper cluster grows even after touching boundary.
    Both `n_steps` and `luf.erasure` will then be incorrect.
    """
    luf = LUF(sf5F)
    n_steps = luf.validate(syndrome5F)
    assert n_steps == 37
    assert luf.erasure == {
        ((0, -1), (0, 0)),
        ((0, 0), (0, 1)),
        ((0, 1), (0, 2)),
        ((0, 2), (0, 3)),

        ((0, 0), (1, 0)),
        ((0, 1), (1, 1)),
        ((0, 2), (1, 2)),


        ((2, 1), (3, 1)),

        ((3, 0), (3, 1)),
        ((3, 1), (3, 2)),

        ((3, 0), (4, 0)),
        ((3, 1), (4, 1)),
        ((3, 2), (4, 2)),

        ((4, -1), (4, 0)),
        ((4, 0), (4, 1)),
        ((4, 1), (4, 2)),
        ((4, 2), (4, 3)),
    }

def test_validate_big_P_shape(sf7F: Surface):
    luf = LUF(sf7F)
    syndrome = {
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 3),
    (2, 1),
    (2, 2),
    (2, 3),
    (4, 1),
    }
    n_steps = luf.validate(syndrome)
    assert n_steps == 26
    assert luf.erasure == {
        ((0, 0), (0, 1)),
        ((0, 1), (0, 2)),
        ((0, 1), (1, 1)),
        ((0, 2), (0, 3)),
        ((0, 2), (1, 2)),
        ((0, 3), (0, 4)),
        ((0, 3), (1, 3)),
        ((1, 1), (2, 1)),
        ((1, 2), (1, 3)),
        ((1, 2), (2, 2)),
        ((1, 3), (1, 4)),
        ((1, 3), (2, 3)),
        ((2, 0), (2, 1)),
        ((2, 1), (2, 2)),
        ((2, 1), (3, 1)),
        ((2, 2), (2, 3)),
        ((2, 2), (3, 2)),
        ((2, 3), (2, 4)),
        ((2, 3), (3, 3)),
        ((3, 1), (4, 1)),
        ((4, 0), (4, 1)),
        ((4, 1), (4, 2)),
        ((4, 1), (5, 1))
    }

def test_validate_globe_shape(sf9F: Surface):
    luf = LUF(sf9F)
    syndrome = {
        (0, 0),
        (0, 2),
        (0, 4),
        (0, 6),
        (2, 6),
        (3, 2),
        (4, 2),
        (4, 6),
        (5, 2),
        (6, 6),
        (8, 2),
        (8, 4),
        (8, 6),
    }
    n_steps = luf.validate(syndrome)
    assert n_steps == 62
    assert luf.erasure == {
        ((0, -1), (0, 0)),
        ((0, 0), (0, 1)),
        ((0, 0), (1, 0)),
        ((0, 1), (0, 2)),
        ((0, 2), (0, 3)),
        ((0, 2), (1, 2)),
        ((0, 3), (0, 4)),
        ((0, 4), (0, 5)),
        ((0, 4), (1, 4)),
        ((0, 5), (0, 6)),
        ((0, 6), (0, 7)),
        ((0, 6), (1, 6)),
        ((1, 2), (2, 2)),
        ((1, 6), (2, 6)),
        ((2, 1), (2, 2)),
        ((2, 1), (3, 1)),
        ((2, 2), (2, 3)),
        ((2, 2), (3, 2)),
        ((2, 3), (3, 3)),
        ((2, 5), (2, 6)),
        ((2, 6), (2, 7)),
        ((2, 6), (3, 6)),
        ((3, 0), (3, 1)),
        ((3, 1), (3, 2)),
        ((3, 1), (4, 1)),
        ((3, 2), (3, 3)),
        ((3, 2), (4, 2)),
        ((3, 3), (3, 4)),
        ((3, 3), (4, 3)),
        ((3, 6), (4, 6)),
        ((4, 0), (4, 1)),
        ((4, 1), (4, 2)),
        ((4, 1), (5, 1)),
        ((4, 2), (4, 3)),
        ((4, 2), (5, 2)),
        ((4, 3), (4, 4)),
        ((4, 3), (5, 3)),
        ((4, 5), (4, 6)),
        ((4, 6), (4, 7)),
        ((4, 6), (5, 6)),
        ((5, 0), (5, 1)),
        ((5, 1), (5, 2)),
        ((5, 1), (6, 1)),
        ((5, 2), (5, 3)),
        ((5, 2), (6, 2)),
        ((5, 3), (5, 4)),
        ((5, 3), (6, 3)),
        ((5, 6), (6, 6)),
        ((6, 1), (6, 2)),
        ((6, 2), (6, 3)),
        ((6, 2), (7, 2)),
        ((6, 5), (6, 6)),
        ((6, 6), (6, 7)),
        ((6, 6), (7, 6)),
        ((7, 2), (8, 2)),
        ((7, 4), (8, 4)),
        ((7, 6), (8, 6)),
        ((8, 1), (8, 2)),
        ((8, 2), (8, 3)),
        ((8, 3), (8, 4)),
        ((8, 4), (8, 5)),
        ((8, 5), (8, 6)),
        ((8, 6), (8, 7))
    }

def test_validate_single_defect(astris3T: LUF, actis3Tu: LUF):
    syndrome: set[Node] = {
        (0, 0, 0),
    }
    erasure = {
        ((0, -1, 0), (0, 0, 0)),
        ((0, 0, 0), (0, 0, 1)),
        ((0, 0, 0), (0, 1, 0)),
        ((0, 0, 0), (1, 0, 0))
    }

    n_steps = astris3T.validate(syndrome)
    assert n_steps == 10
    assert astris3T.erasure == erasure

    n_steps = actis3Tu.validate(syndrome)
    assert n_steps == 71
    assert actis3Tu.erasure == erasure

def test_validate_square(astris3T: LUF, actis3Tu: LUF):
    syndrome = {
        (0, 0, 0),
        (1, 0, 0),
        (0, 0, 1),
        (1, 0, 1),
    }
    erasure = {
        ((0, 0, 0), (0, 0, 1)),
        ((0, 0, 0), (1, 0, 0)),
        ((0, 0, 1), (1, 0, 1)),
        ((1, 0, 0), (1, 0, 1))
    }

    n_steps = astris3T.validate(syndrome)
    assert n_steps == 7
    assert astris3T.erasure == erasure

    n_steps = actis3Tu.validate(syndrome)
    assert n_steps == 35
    assert actis3Tu.erasure == erasure


def test_pointer_digraph(astris3F: LUF, uvw):
    u, v, _ = uvw
    astris3F.NODES.dc[u].pointer = 'W'
    astris3F.NODES.dc[v].pointer = 'W'
    astris3F.growth[u, v] = Growth.FULL
    dig, dig_diedges, dig_edges = astris3F._pointer_digraph
    assert type(dig) is nx.DiGraph
    assert set(dig.nodes) == set(astris3F.CODE.NODES)
    assert set(dig.edges) == {(v, u)}
    assert dig_diedges == [(v, u)]
    assert dig_edges == [(u, v)]