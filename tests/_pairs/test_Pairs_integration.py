from localuf.type_aliases import Node, Edge
from localuf._pairs import Pairs

FourNodes = tuple[Node, Node, Node, Node]


def test_load_annihilates_anyons(pairs: Pairs, uvwx: FourNodes):
    u, v, _, _ = uvwx
    pairs._dc = {u: v, v: u}
    pairs.load((u, v))
    assert pairs._dc == {}


def test_load_joins_paths(pairs: Pairs, uvwx: FourNodes):
    u, v, w, x = uvwx
    pairs._dc = {u: w, w: u, v: x, x: v}
    pairs.load((u, v))
    assert pairs._dc == {w: x, x: w}


def test_load_extends_path(pairs: Pairs, uvwx: FourNodes):
    u, v, w, _ = uvwx
    pairs._dc = {u: w, w: u}
    pairs.load((u, v))
    assert pairs._dc == {v: w, w: v}
    # test symmetry
    pairs.load((u, v))
    assert pairs._dc == {u: w, w: u}


def test_load_adds_path(pairs: Pairs, uvwx: FourNodes):
    u, v, w, x = uvwx
    pairs.load((u, v))
    assert pairs._dc == {u: v, v: u}
    pairs.load((w, x))
    assert pairs._dc == {u: v, v: u, w: x, x: w}


def test_load_ignores_cycle(pairs: Pairs, uvwx: FourNodes):
    u, v, w, x = uvwx
    a_cycle: set[Edge] = {
        (u, v),
        (v, x),
        (w, x),
        (u, w),
    }
    for e in a_cycle:
        pairs.load(e)
    assert pairs._dc == {}