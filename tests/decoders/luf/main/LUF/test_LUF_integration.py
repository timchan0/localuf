import itertools

import pytest
import networkx as nx

from localuf import Surface
from localuf.type_aliases import Node
from localuf.decoders.luf import Macar, Actis
from localuf.constants import Growth
from localuf.decoders.luf.constants import Stage

@pytest.mark.parametrize("macar, syndrome", [
    ("macar5F", "syndrome5F"),
    ("macar5T", "syndrome5T"),
])
def test_not_any_busy_after_growing(macar: Macar, syndrome: set[Node], request):
    macar = request.getfixturevalue(macar)
    syndrome = request.getfixturevalue(syndrome)
    macar.NODES.load(syndrome)
    macar.NODES.advance()
    assert not any(node.busy for node in macar.NODES.dc.values())

@pytest.mark.parametrize("macar, syndrome", [
    ("macar5F", "syndrome5F"),
    ("macar5T", "syndrome5T"),
])
def test_not_any_busy_after_presyncing(macar: Macar, syndrome: set[Node], request):
    macar = request.getfixturevalue(macar)
    syndrome = request.getfixturevalue(syndrome)
    macar.NODES.load(syndrome)
    while not macar.NODES.valid:
        check = macar.CONTROLLER.stage is Stage.PRESYNCING
        macar._advance()
        if check:
            assert not any(node.busy for node in macar.NODES.dc.values())
            assert macar.CONTROLLER.stage is Stage.SYNCING

def test_presyncing_macar(macar: Macar):
    for _ in itertools.repeat(None, 2):
        macar._advance()
    # controller stage now PRESYNC
    for node in macar.NODES.dc.values():
        node.active = True
    macar._advance()
    assert all(not node.active for node in macar.NODES.dc.values())

def test_presyncing_actis(actis: Actis):
    actis.CONTROLLER.stage = Stage.PRESYNCING
    for node in actis.NODES.dc.values():
        node.stage = Stage.PRESYNCING # type: ignore
        node.active = True
    actis._advance()
    assert all(not node.active for node in actis.NODES.dc.values())

def test_validate_single_isolated_error(macar3F: Macar, actis3Fu: Actis):
    syndrome = {(0, 0), (0, 1)}
    erasure = {((0, 0), (0, 1))}

    n_steps = macar3F.validate(syndrome)
    assert n_steps == 6
    assert macar3F.erasure == erasure

    n_steps = actis3Fu.validate(syndrome)
    assert n_steps == 28
    assert actis3Fu.erasure == erasure

def test_validate_vdash_shape(macar3F: Macar, actis3Fu: Actis):
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

    n_steps = macar3F.validate(syndrome)
    assert n_steps == 6
    assert macar3F.erasure == erasure

    n_steps = actis3Fu.validate(syndrome)
    assert n_steps == 29
    assert actis3Fu.erasure == erasure

def test_validate_sixpack(
        sf3F: Surface,
        macar3F: Macar,
        actis3Fu: Actis,
):
    syndrome = {v for v in sf3F.NODES if not sf3F.is_boundary(v)}
    erasure = {e for e in sf3F.EDGES if not any(
        sf3F.is_boundary(v) for v in e
    )}

    n_steps = macar3F.validate(syndrome)
    assert n_steps == 8
    assert macar3F.erasure == erasure

    n_steps = actis3Fu.validate(syndrome)
    assert n_steps == 31
    assert actis3Fu.erasure == erasure

def test_validate_perp_shape(macar5F: Macar, actis5Fu: Actis):
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

    n_steps = macar5F.validate(syndrome)
    assert n_steps == 8
    assert macar5F.erasure == erasure

    n_steps = actis5Fu.validate(syndrome)
    assert n_steps == 43
    assert actis5Fu.erasure == erasure

def test_validate_diagonal(sf5F: Surface):
    macar = Macar(sf5F)
    syndrome = {
        (1, 1),
        (2, 2),
    }
    n_steps = macar.validate(syndrome)
    assert n_steps == 12
    assert macar.erasure == {
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
    macar = Macar(sf5F)
    syndrome = {
        (0, 1),
        (0, 2),
        (1, 2),
        (2, 2),
        (3, 0),
        (3, 1),
        (3, 2),
    }
    n_steps = macar.validate(syndrome)
    assert n_steps == 36
    assert macar.erasure == {
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
    macar = Macar(sf5F)
    syndrome_3_row = {
        (0, 0),
        (0, 1),
        (0, 2),
    }
    n_steps = macar.validate(syndrome_3_row)
    assert n_steps == 17
    assert macar.erasure == {
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
    macar = Macar(sf5F)
    syndrome_rightharpoonup = {
        (0, 3),
        (1, 0),
        (1, 1),
        (1, 2),
        (1, 3),
    }
    n_steps = macar.validate(syndrome_rightharpoonup)
    assert n_steps == 26
    assert macar.erasure == {
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
    Both `n_steps` and `decoder.erasure` will then be incorrect.
    """
    decoder = Macar(sf5F)
    n_steps = decoder.validate(syndrome5F)
    assert n_steps == 37
    assert decoder.erasure == {
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
    decoder = Macar(sf7F)
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
    n_steps = decoder.validate(syndrome)
    assert n_steps == 26
    assert decoder.erasure == {
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
    decoder = Macar(sf9F)
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
    n_steps = decoder.validate(syndrome)
    assert n_steps == 62
    assert decoder.erasure == {
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

def test_validate_single_defect(macar3T: Macar, actis3Tu: Actis):
    syndrome: set[Node] = {
        (0, 0, 0),
    }
    erasure = {
        ((0, -1, 0), (0, 0, 0)),
        ((0, 0, 0), (0, 0, 1)),
        ((0, 0, 0), (0, 1, 0)),
        ((0, 0, 0), (1, 0, 0))
    }

    n_steps = macar3T.validate(syndrome)
    assert n_steps == 10
    assert macar3T.erasure == erasure

    n_steps = actis3Tu.validate(syndrome)
    assert n_steps == 71
    assert actis3Tu.erasure == erasure

def test_validate_square(macar3T: Macar, actis3Tu: Actis):
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

    n_steps = macar3T.validate(syndrome)
    assert n_steps == 7
    assert macar3T.erasure == erasure

    n_steps = actis3Tu.validate(syndrome)
    assert n_steps == 35
    assert actis3Tu.erasure == erasure


def test_pointer_digraph(macar3F: Macar, uvw):
    u, v, _ = uvw
    macar3F.NODES.dc[u].pointer = 'W'
    macar3F.NODES.dc[v].pointer = 'W'
    macar3F.growth[u, v] = Growth.FULL
    dig, dig_diedges, dig_edges = macar3F._pointer_digraph
    assert type(dig) is nx.DiGraph
    assert set(dig.nodes) == set(macar3F.CODE.NODES)
    assert set(dig.edges) == {(v, u)}
    assert dig_diedges == [(v, u)]
    assert dig_edges == [(u, v)]