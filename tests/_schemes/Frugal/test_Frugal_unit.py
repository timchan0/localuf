import math
from unittest import mock

import pytest

from localuf import Repetition, Surface
from localuf.type_aliases import Edge
from localuf.decoders import Snowflake
from localuf._schemes import Frugal
from localuf._pairs import LogicalCounter


@pytest.fixture(params=range(3, 11, 2), ids=lambda x: f"d{x}")
def rp_frugal(request):
    rp = Repetition(
        request.param,
        noise='phenomenological',
        scheme='frugal',
    )
    return rp.SCHEME

@pytest.fixture(params=range(3, 11, 2), ids=lambda x: f"d{x}")
def sf_frugal(request):
    sf = Surface(
        request.param,
        noise='phenomenological',
        scheme='frugal',
    )
    return sf.SCHEME

@pytest.fixture
def frugal3(rp3_frugal: Repetition):
    return rp3_frugal.SCHEME


def test_LOGICAL_COUNTER_attribute(frugal3: Frugal):
    assert type(frugal3._LOGICAL_COUNTER) is LogicalCounter


def test_COMMIT_HEIGHT_attribute(rp_frugal: Frugal):
    assert rp_frugal._COMMIT_HEIGHT == 1


def test_BUFFER_HEIGHT_attribute(rp_frugal: Frugal):
    assert rp_frugal._BUFFER_HEIGHT == 2*(rp_frugal._CODE.D//2)


def test_reset(frugal3: Frugal):
    e = ((-1, 0), (0, 0))
    frugal3.error = {e}
    frugal3.step_counts = [2]
    frugal3._future_boundary_syndrome = {(0, 3)}
    with mock.patch("localuf._pairs.Pairs.reset") as mock_reset:
        frugal3.reset()
        mock_reset.assert_called_once_with()
    assert frugal3.error == set()
    assert frugal3.step_counts == []
    assert frugal3._future_boundary_syndrome == set()


def test_get_logical_error_repetition(rp_frugal: Frugal):

    assert not rp_frugal.get_logical_error()
    assert rp_frugal.pairs._dc == {}
    
    u, v = (-1, 0), (0, 0)
    rp_frugal.pairs._dc = {u: v, v: u}
    assert not rp_frugal.get_logical_error()
    uu, vv = (-1, -1), (0, -1)
    assert rp_frugal.pairs._dc == {uu: vv, vv: uu}

    u, v = (-1, 0), (rp_frugal._CODE.D-1, 0)
    rp_frugal.pairs._dc = {u: v, v: u}
    assert rp_frugal.get_logical_error()
    assert rp_frugal.pairs._dc == {}


def test_get_logical_error_surface(sf_frugal: Frugal):
    
    assert not sf_frugal.get_logical_error()
    assert sf_frugal.pairs._dc == {}

    u, v = (0, -1, 0), (0, 0, 0)
    sf_frugal.pairs._dc = {u: v, v: u}
    assert not sf_frugal.get_logical_error()
    uu, vv = (0, -1, -1), (0, 0, -1)
    assert sf_frugal.pairs._dc == {uu: vv, vv: uu}

    u, v = (0, -1, 0), (0, sf_frugal._CODE.D-1, 0)
    sf_frugal.pairs._dc = {u: v, v: u}
    assert sf_frugal.get_logical_error()
    assert sf_frugal.pairs._dc == {}

    # unique to surface: north--south edge
    u, v = (0, 0, 0), (1, 0, 0)
    sf_frugal.pairs._dc = {u: v, v: u}
    assert not sf_frugal.get_logical_error()
    uu, vv = (0, 0, -1), (1, 0, -1)
    assert sf_frugal.pairs._dc == {uu: vv, vv: uu}


def test_advance(frugal3: Frugal):
    p = 1/2
    with (
        mock.patch("localuf._schemes.Frugal._raise_window") as rw,
        mock.patch(
            f"localuf.codes.Repetition.make_error",
            return_value=set(),
        ) as me,
        mock.patch(
            f"localuf._schemes.Frugal._load",
            return_value=set(),
        ) as mock_load,
        mock.patch(
            f"localuf.decoders.snowflake.Snowflake.decode",
            return_value=2,
        ) as mock_decode,
    ):
        assert frugal3.advance(p, Snowflake(frugal3._CODE)) == 2
        rw.assert_called_once_with()
        me.assert_called_once_with(p, exclude_future_boundary=False)
        mock_load.assert_called_once_with(set())
        mock_decode.assert_called_once_with(set())


def test_raise_window(rp_frugal: Frugal):

    raise_window_helper(rp_frugal)
    
    e, f, g = (
        ((-1, 0), (0, 0)),
        ((-1, 1), (0, 1)),
        ((0, 0), (0, 1)),
    )
    rp_frugal.error = {e, f, g}
    rp_frugal._raise_window()
    assert rp_frugal.error == {e}
    assert rp_frugal.pairs._dc == {
        (-1, 0): (0, 1),
        (0, 1): (-1, 0),
    }
    rp_frugal.pairs._dc = {
        (-1, -1): (0, 0),
        (0, 0): (-1, -1),
    }
    rp_frugal._raise_window()
    assert rp_frugal.error == set()
    assert rp_frugal.pairs._dc == {(-1, 0): (-1, -1), (-1, -1): (-1, 0)}

def raise_window_helper(frugal: Frugal):
    frugal._raise_window()
    assert frugal.error == set()
    assert frugal.pairs._dc == {}


def test_load(rp_frugal: Frugal):
    h = rp_frugal.WINDOW_HEIGHT
    edges = (
        ((-1, h-2), (0, h-2)),
        ((-1, h-1), (0, h-1)),
        ((0, h-1), (1, h-1)),
        ((1, h-1), (2, h-1)),
        ((0, h-1), (0, h)),
    )
    error: set[Edge] = set(edges[1:])
    rp_frugal.error = set(edges[:1])
    rp_frugal._future_boundary_syndrome = {(1, h)}
    syndrome = rp_frugal._load(error)
    assert rp_frugal.error == set(edges)
    assert syndrome == {
        (0, h-1),
        (1, h-1),
    } if rp_frugal._CODE.D == 3 else {
        (0, h-1),
        (1, h-1),
        (2, h-1),
    }
    assert rp_frugal._future_boundary_syndrome == {(0, h)}


class TestRun:

    def test_zero_slenderness(self, frugal3: Frugal):
        decoder = Snowflake(frugal3._CODE)
        with pytest.raises(ValueError):
            frugal3.run(decoder, 1, 0)

    @pytest.mark.parametrize("n", range(1, 4))
    def test_run(self, frugal3: Frugal, n: int):
        d = frugal3._CODE.D
        decoder = Snowflake(frugal3._CODE)
        p = 1/2
        transient_count = math.ceil(frugal3.WINDOW_HEIGHT / frugal3._COMMIT_HEIGHT)
        steady_state_raise_count = (n+1) * d
        cleanse_count = 2 * frugal3.WINDOW_HEIGHT
        raise_count = transient_count + steady_state_raise_count + cleanse_count
        with (
            mock.patch("localuf._schemes.Frugal.reset") as mock_reset,
            mock.patch("localuf.decoders.Snowflake.reset") as snow_reset,
            mock.patch("localuf._schemes.Frugal.advance") as mock_advance,
            mock.patch(
                "localuf._schemes.Frugal.get_logical_error",
                return_value=1,
            ) as gle,
            mock.patch("localuf.decoders.snowflake.Snowflake.init_history") as ih,
            mock.patch("localuf.decoders.snowflake.Snowflake.draw_decode") as dd,
        ):
            failure_count, slenderness = frugal3.run(decoder, p, n)
            assert failure_count == raise_count
            assert slenderness == (transient_count + steady_state_raise_count) / d
            mock_reset.assert_called_once_with()
            snow_reset.assert_called_once_with()
            assert mock_advance.call_args_list == \
                (transient_count + d*n + d-1) * [mock.call(p, decoder, exclude_future_boundary=False, log_history=False, time_only='merging')] \
                + [mock.call(p, decoder, exclude_future_boundary=True, log_history=False, time_only='merging')] \
                + cleanse_count * [mock.call(0, decoder, exclude_future_boundary=False, log_history=False, time_only='merging')]
            assert gle.call_args_list == raise_count * [mock.call()]
            ih.assert_not_called()
            dd.assert_not_called()

            frugal3.run(decoder, p, n, draw='fine')
            ih.assert_called_once_with()
            dd.assert_called_once_with()