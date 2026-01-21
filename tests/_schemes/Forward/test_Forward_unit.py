from itertools import product
import math
from unittest import mock

import pytest

from localuf import Surface, Repetition
from localuf.decoders import UF
from localuf.type_aliases import Edge
from localuf._determinants import SpaceTimeDeterminant
from localuf._schemes import Forward
from localuf._pairs import LogicalCounter


@pytest.fixture(params=range(3, 11, 2), ids=lambda x: f"d{x}")
def sf_ol(request):
    sf = Surface(
        request.param,
        noise='phenomenological',
        scheme='forward',
    )
    return sf.SCHEME


def test_COMMIT_HEIGHT_attribute(forward_rp: Forward):
    assert forward_rp._COMMIT_HEIGHT == forward_rp._CODE.D


def test_BUFFER_HEIGHT_attribute(forward_rp: Forward):
    assert forward_rp._BUFFER_HEIGHT == forward_rp._CODE.D


def test_DETERMINANT_attribute(forward_rp: Forward):
    assert type(forward_rp._DETERMINANT) is SpaceTimeDeterminant


def test_LOGICAL_COUNTER_attribute(forward3F: Forward):
    assert type(forward3F._LOGICAL_COUNTER) is LogicalCounter


def assert_is_tuple_of_edges(edges: tuple[Edge, ...]):
    assert type(edges) is tuple
    assert type(edges[0]) is tuple
    assert type(edges[0][0]) is tuple
    assert type(edges[0][0][0]) is int


def test_COMMIT_EDGES_attribute(forward_rp: Forward):
    assert_is_tuple_of_edges(forward_rp._COMMIT_EDGES)
    assert set(forward_rp._COMMIT_EDGES).issubset(set(forward_rp._CODE.EDGES))
    assert 2*len(forward_rp._COMMIT_EDGES) == len(forward_rp._CODE.EDGES)


def test_BUFFER_EDGES_attribute(forward_rp: Forward):
    assert_is_tuple_of_edges(forward_rp._BUFFER_EDGES)
    assert set(forward_rp._BUFFER_EDGES).issubset(set(forward_rp._CODE.EDGES))
    assert 2*len(forward_rp._BUFFER_EDGES) == len(forward_rp._CODE.EDGES)
    assert set(forward_rp._COMMIT_EDGES).isdisjoint(set(forward_rp._BUFFER_EDGES))


def test_reset(forward_rp: Forward):
    forward_rp.history = [(set(forward_rp._CODE.EDGES), set(forward_rp._CODE.EDGES), set(forward_rp._CODE.NODES))]
    forward_rp.reset()
    assert not hasattr(forward_rp, 'history')


def test_get_leftover(forward_rp: Forward):
    all_edges = set(forward_rp._CODE.EDGES)
    assert forward_rp._get_leftover(set(), set()) == (set(), set())
    assert forward_rp._get_leftover(set(), all_edges) == (set(forward_rp._COMMIT_EDGES), set())
    assert forward_rp._get_leftover(all_edges, set()) == (set(forward_rp._COMMIT_EDGES), set(forward_rp._BUFFER_EDGES))
    assert forward_rp._get_leftover(all_edges, all_edges) == (set(), set(forward_rp._BUFFER_EDGES))


def test_get_logical_error_repetition(forward_rp: Forward):
    d = forward_rp._CODE.D
    ts = [0, forward_rp._COMMIT_HEIGHT-1]
    paths = [{((j, t), (j+1, t)) for j in range(-1, d-1)} for t in ts]
    assert forward_rp.get_logical_error(set()) == 0
    assert forward_rp.pairs._dc == {}
    assert forward_rp.get_logical_error(paths[0]) == 1
    assert forward_rp.pairs._dc == {}
    assert forward_rp.get_logical_error(paths[1]) == 1
    assert forward_rp.pairs._dc == {}
    assert forward_rp.get_logical_error(paths[0] | paths[1]) == 2
    assert forward_rp.pairs._dc == {}


def test_get_logical_error_surface(sf_ol: Forward):

    d = sf_ol._CODE.D
    h = sf_ol._COMMIT_HEIGHT
    ts = [0, h-1]
    paths = [{((0, j, t), (0, j+1, t)) for j in range(-1, d-1)} for t in ts]
    assert sf_ol.get_logical_error(set()) == 0
    assert sf_ol.pairs._dc == {}
    assert sf_ol.get_logical_error(paths[0]) == 1
    assert sf_ol.pairs._dc == {}
    assert sf_ol.get_logical_error(paths[1]) == 1
    assert sf_ol.pairs._dc == {}
    assert sf_ol.get_logical_error(paths[0] | paths[1]) == 2
    assert sf_ol.pairs._dc == {}

    commit_leftover = {
        ((0, 0, h-1), (0, 0, h)),
        ((0, 0, h-1), (1, 0, h-1)),
        ((1, 0, h-1), (1, 0, h)),
    }
    assert sf_ol.get_logical_error(commit_leftover) == 0
    # test pair with zero j separation not ignored
    assert sf_ol.pairs._dc == {
        (0, 0, 0): (1, 0, 0),
        (1, 0, 0): (0, 0, 0),
    }


def test_get_syndrome(forward_rp: Forward):
    with mock.patch(
        "localuf._schemes.Forward._get_artificial_defects",
        return_value={(1, 0)},
    ) as mock_gad:
        next_syndrome = forward_rp._get_syndrome(set(forward_rp._CODE.EDGES), set())
        mock_gad.assert_called_once_with(set(forward_rp._CODE.EDGES))
        assert next_syndrome == {(1, 0)}


def test_get_artificial_defects(forward_rp: Forward):
    for j in range(forward_rp._CODE.D-1):
        commit_leftover = {((j, forward_rp._COMMIT_HEIGHT-1), (j, forward_rp._COMMIT_HEIGHT))}
        assert forward_rp._get_artificial_defects(commit_leftover) == {(j, 0)}
        commit_leftover = {((j, 0), (j, 1))}
        assert forward_rp._get_artificial_defects(commit_leftover) == set()


def test_get_artificial_defects_circuit_level(sfCL_OL_scheme: Forward):
    h = sfCL_OL_scheme._COMMIT_HEIGHT
    # test artificial defects from previous decoding cycle do not carry over
    commit_leftover: set[Edge] = {((0, -1, 0), (0, 0, 0))}  # this is a valid commit leftover!
    assert sfCL_OL_scheme._get_artificial_defects(commit_leftover) == set()
    # test SD edges considered (this could fail if we assumed v higher than u for each timelike uv)
    commit_leftover = {
        ((1, -1, h-1), (1, 0, h-1)),
        ((0, 0, h), (1, 0, h-1)),
    }
    assert sfCL_OL_scheme._get_artificial_defects(commit_leftover) == {(0, 0, 0)}
    # test artificial defect annihilation
    commit_leftover |= {
        ((0, -1, h-1), (0, 0, h-1)),
        ((0, 0, h-1), (0, 0, h)),
    }
    assert sfCL_OL_scheme._get_artificial_defects(commit_leftover) == set()


def test_make_error_in_buffer_region(forward_rp: Forward):
    p = 0.5
    with mock.patch(
        "localuf.noise.Phenomenological.make_error",
        return_value=set(forward_rp._CODE.EDGES),
    ) as mock_me:
        error_in_buffer_region = forward_rp._make_error_in_buffer_region(p)
        mock_me.assert_called_once_with(p)
        assert error_in_buffer_region == set(forward_rp._BUFFER_EDGES)


class TestRun:

    @pytest.fixture(
            params=product(range(3, 7, 2), range(1, 4), range(1, 4)),
            ids=lambda x: f"d{x[0]} c{x[1]} b{x[2]}",
    )
    def forward(self, request):
        d, c, b = request.param
        return Repetition(
            d,
            'phenomenological',
            scheme='forward',
            commit_height=c,
            buffer_height=b,
        ).SCHEME
    
    def test_zero_cycles(self, forward: Forward):
        decoder = UF(forward._CODE)
        with pytest.raises(ValueError):
            forward.run(decoder, 1, 0)

    @pytest.mark.parametrize("n", range(1, 4), ids=lambda x: f"n{x}")
    def test_commit_equals_buffer(self, forward: Forward, n: int):
        uf = UF(forward._CODE)
        p = 0.5
        cleanse_count = forward.WINDOW_HEIGHT // forward._COMMIT_HEIGHT
        commit_leftover = set()
        decoding_cycle_count = n+1 + cleanse_count
        with (
            mock.patch(
            "localuf._schemes.Forward._make_error_in_buffer_region",
            return_value=set()
        ) as mock_meibr,
            mock.patch("localuf._schemes.Forward._make_error") as mock_me,
            mock.patch("localuf._schemes.Forward._get_artificial_defects") as mock_gad,
            mock.patch("localuf.codes.Repetition.get_syndrome") as mock_gs,
            mock.patch("localuf.decoders.uf.UF.reset") as mock_reset,
            mock.patch("localuf.decoders.uf.UF.decode") as mock_decode,
            mock.patch(
            "localuf._schemes.Forward._get_leftover",
            return_value=(set(), set())
        ) as mock_gl,
            mock.patch(
            "localuf._schemes.Forward.get_logical_error",
            return_value=1
        ) as mock_gle,
        ):
            m, slenderness = forward.run(uf, p, n)
            assert m == decoding_cycle_count
            layer_count = forward._BUFFER_HEIGHT + (n+1)*forward._COMMIT_HEIGHT
            assert slenderness == layer_count / forward._CODE.D
            mock_meibr.assert_called_once_with(p)
            assert mock_me.call_args_list == n*[mock.call(set(), p, exclude_future_boundary=False)] \
                + [mock.call(set(), p, exclude_future_boundary=True)] \
                + cleanse_count * [mock.call(set(), 0, exclude_future_boundary=False)]
            assert mock_gad.call_args_list == [mock.call(commit_leftover)] * decoding_cycle_count
            assert mock_gs.call_args_list == [mock.call(mock_me.return_value)] * decoding_cycle_count
            syndrome = mock_gs.return_value ^ mock_gad.return_value
            assert mock_reset.call_args_list == [mock.call()] * decoding_cycle_count
            assert mock_decode.call_args_list == [mock.call(syndrome)] * decoding_cycle_count
            assert mock_gl.call_args_list == [mock.call(mock_me.return_value, uf.correction)] * decoding_cycle_count
            assert mock_gle.call_args_list == [mock.call(commit_leftover)] * decoding_cycle_count


def test_make_error(forward3F: Forward):
    # test commit made
    error = forward3F._make_error({
        ((0, 3), (0, 4)),
        ((0, 3), (1, 3)),
    }, 0)
    assert error == {
        ((0, 0), (0, 1)),
        ((0, 0), (1, 0)),
    }
    # test buffer made
    with mock.patch(
        "localuf.noise.Phenomenological.make_error",
        return_value=set(),
    ) as mock_me:
        error = forward3F._make_error(set(), 0.5)
        mock_me.assert_called_once_with(0.5)
    assert error == set()