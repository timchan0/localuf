from unittest import mock

import pytest

from localuf.type_aliases import Node
from localuf._pairs import Pairs

FourNodes = tuple[Node, Node, Node, Node]


def test_reset(pairs: Pairs, uvwx: FourNodes):
    u, v, _, _ = uvwx
    pairs._dc = {u: v, v: u}
    pairs.reset()
    assert pairs._dc == {}


def test_contains(pairs: Pairs, uvwx: FourNodes):
    u, v, w, _ = uvwx
    pairs._dc = {u: v, v: u}
    assert u in pairs
    assert v in pairs
    assert w not in pairs


def test_getitem(pairs: Pairs, uvwx: FourNodes):
    u, v, w, _ = uvwx
    pairs._dc = {u: v, v: u}
    assert pairs[u] == v
    assert pairs[v] == u
    with pytest.raises(KeyError):
        pairs[w]


def test_add(pairs: Pairs, uvwx: FourNodes):
    u, v, _, _ = uvwx
    pairs.add(u, v)
    assert pairs._dc == {u: v, v: u}


def test_remove(pairs: Pairs, uvwx: FourNodes):
    u, v, _, _ = uvwx
    pairs._dc = {u: v, v: u}
    pairs.remove(u)
    assert pairs._dc == {}


def test_load_annihilates_anyons(pairs: Pairs, uvwx: FourNodes):
    u, v, _, _ = uvwx
    pairs._dc = {u: v, v: u}
    with mock.patch(
        "localuf._pairs.Pairs.remove",
        side_effect=pairs.remove,
    ) as mock_remove:
        pairs.load((u, v))
        mock_remove.assert_called_once_with(u)


def test_load_joins_paths(pairs: Pairs, uvwx: FourNodes):
    u, v, w, x = uvwx
    pairs._dc = {u: w, w: u, v: x, x: v}
    with (
        mock.patch("localuf._pairs.Pairs.remove") as mock_remove,
        mock.patch("localuf._pairs.Pairs.add") as mock_add,
    ):
        pairs.load((u, v))
        assert mock_remove.call_args_list == [mock.call(u), mock.call(v)]
        mock_add.assert_called_once_with(w, x)


def test_load_extends_path(pairs: Pairs, uvwx: FourNodes):
    u, v, w, _ = uvwx
    pairs._dc = {u: w, w: u}
    with (
        mock.patch("localuf._pairs.Pairs.remove") as mock_remove,
        mock.patch("localuf._pairs.Pairs.add") as mock_add,
    ):
        pairs.load((u, v))
        mock_remove.assert_called_once_with(u)
        mock_add.assert_called_once_with(v, w)
    # test symmetry
    with (
        mock.patch("localuf._pairs.Pairs.remove") as mock_remove,
        mock.patch("localuf._pairs.Pairs.add") as mock_add,
    ):
        pairs.load((v, u))
        mock_remove.assert_called_once_with(u)
        mock_add.assert_called_once_with(v, w)


def test_load_adds_path(pairs: Pairs, uvwx: FourNodes):
    u, v, _, _ = uvwx
    with mock.patch("localuf._pairs.Pairs.add") as mock_add:
        pairs.load((u, v))
        mock_add.assert_called_once_with(u, v)


def test_as_set(pairs: Pairs, uvwx: FourNodes):
    u, v, w, x = uvwx
    pairs._dc = {u: v, v: u, w: x, x: w}
    assert pairs.as_set == {(u, v), (w, x)}