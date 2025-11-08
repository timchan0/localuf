from unittest import mock

import pytest

from localuf.type_aliases import Node
from localuf.constants import Growth
from localuf.decoders import Snowflake
from localuf.decoders.policies import DecodeDrawer

@pytest.mark.parametrize("unrooter", ['full', 'simple'])
def test_init(snowflake: Snowflake, unrooter):
    with mock.patch("localuf.decoders.uf.BaseUF.__init__") as mock_init:
        snowflake.__init__(snowflake.CODE, unrooter=unrooter)
        mock_init.assert_called_once_with(snowflake.CODE)
    assert type(snowflake._NODES) is dict
    assert type(snowflake._EDGES) is dict
    assert set(snowflake._NODES.keys()) == {
        v for v in snowflake.CODE.NODES
        if v[snowflake.CODE.TIME_AXIS] < snowflake.CODE.SCHEME.WINDOW_HEIGHT
    }
    assert set(snowflake._EDGES.keys()) == {
        e for e in snowflake.CODE.EDGES
        if e[1][snowflake.CODE.TIME_AXIS] < snowflake.CODE.SCHEME.WINDOW_HEIGHT
    }
    assert type(snowflake._DECODE_DRAWER) is DecodeDrawer
    assert snowflake._DECODE_DRAWER._FIG_WIDTH == snowflake._FIG_WIDTH
    assert snowflake._DECODE_DRAWER._FIG_HEIGHT == snowflake._FIG_HEIGHT

@pytest.mark.parametrize("prop", [
    "NODES",
    "EDGES",
    "syndrome",
    "growth",
    "correction",
    "_pointer_digraph",
    "_FIG_WIDTH",
    "_FIG_HEIGHT",
])
def test_property_attributes(test_property, snowflake: Snowflake, prop):
    test_property(snowflake, prop)


def test_syndrome(snowflake: Snowflake,):
    snowflake.NODES[0, 0].defect = True
    assert snowflake.syndrome == {(0, 0)}
    snowflake.NODES[-1, 0].defect = True
    assert snowflake.syndrome == {(0, 0)}


@pytest.mark.parametrize("v", [(0, 0), (-1, 0)])
def test_verbose_syndrome(snowflake: Snowflake, v: Node):
    snowflake.NODES[v].defect = True
    assert snowflake.verbose_syndrome == {v}


def test_growth(snowflake: Snowflake, uvw: tuple[Node, Node, Node]):
    u, v, w = uvw
    snowflake.EDGES[u, v].growth = Growth.HALF
    snowflake.EDGES[v, w].growth = Growth.FULL
    for e in snowflake.EDGES.keys():
        if e == (u, v):
            assert snowflake.growth[e] is Growth.HALF
        elif e == (v, w):
            assert snowflake.growth[e] is Growth.FULL
        else:
            assert snowflake.growth[e] is Growth.UNGROWN


@pytest.fixture
def index_to_id_helper():
    def f(uf: Snowflake):
        d = uf.CODE.D
        boundary_IDs: set[int] = set()
        detector_IDs: set[int] = set()
        for v in uf.NODES.keys():
            id_ = uf.index_to_id(v)
            if uf.CODE.is_boundary(v):
                boundary_IDs.add(id_)
            else:
                detector_IDs.add(id_)
        assert max(boundary_IDs) < min(detector_IDs)
        return d, boundary_IDs, detector_IDs
    return f


def test_index_to_id_2D(snowflake: Snowflake, index_to_id_helper):
    h = snowflake.CODE.SCHEME.WINDOW_HEIGHT
    d, boundary_IDs, detector_IDs = index_to_id_helper(snowflake)
    assert len(boundary_IDs) + len(detector_IDs) == h * (d+1)
    # highest ID boundary node
    assert snowflake.index_to_id((d-1, 0)) == 2*h-1
    # highest ID detector
    assert snowflake.index_to_id((d-2, 0)) == h * (d+1) - 1


def test_reset(snowflake: Snowflake):
    snowflake.history = [snowflake]
    with (
        mock.patch("localuf.decoders.uf.BaseUF.reset") as uf_reset,
        mock.patch("localuf.decoders.snowflake._Node.reset") as node_reset,
        mock.patch("localuf.decoders.snowflake._Edge.reset") as edge_reset,
    ):
        snowflake.reset()
        uf_reset.assert_called_once_with()
        assert node_reset.call_args_list == [mock.call()] * len(snowflake.NODES)
        assert edge_reset.call_args_list == [mock.call()] * len(snowflake.EDGES)
    assert not hasattr(snowflake, 'history')


def test_init_history(snowflake3: Snowflake):
    assert not hasattr(snowflake3, 'history')
    snowflake3.init_history()
    assert snowflake3.history == []


tds = "localuf.decoders.snowflake"


class TestDecode:

    syndrome: set[Node] = {(0, 0)}
    time_only = 'merging'

    def two_one(self, snowflake3: Snowflake):
        log_history = 'fine'
        return_value = 1
        with (
            mock.patch(f"{tds}.Snowflake.append_history") as mock_ah,
            mock.patch(f"{tds}.Snowflake.drop") as mock_drop,
            mock.patch(
                f"{tds}.Snowflake.merge",
                return_value=return_value,
            ) as mock_merge,
        ):
            assert snowflake3.decode(self.syndrome, log_history=log_history) == 2*return_value
            assert mock_ah.call_args_list == [mock.call()] * 3
            mock_drop.assert_called_once_with(self.syndrome)
            assert mock_merge.call_args_list == [
                mock.call(True, log_history, time_only=self.time_only),
                mock.call(False, log_history, time_only=self.time_only),
            ]

    def one_one(self, snowflake3_one_one: Snowflake):
        return_value = 'return_value'
        log_history = 'fine'
        with (
            mock.patch(f"{tds}.Snowflake.append_history") as mock_ah,
            mock.patch(f"{tds}.Snowflake.drop") as mock_drop,
            mock.patch(
                f"{tds}.Snowflake.merge",
                return_value=return_value,
            ) as mock_merge,
        ):
            assert snowflake3_one_one.decode(self.syndrome, log_history=log_history) == return_value
            assert mock_ah.call_args_list == [mock.call()] * 2
            mock_drop.assert_called_once_with(self.syndrome)
            mock_merge.assert_called_once_with(
                whole=True,
                log_history=log_history,
                time_only=self.time_only,
            )


def test_drop(snowflake: Snowflake):
    h = snowflake.CODE.SCHEME.WINDOW_HEIGHT
    syndrome = {
        (0, h-1),
        (2, h-1),
    }
    with (
        mock.patch(f"{tds}.EdgeContact.drop") as ec,
        mock.patch(f"{tds}.FloorContact.drop") as gc,
        mock.patch(f"{tds}._Edge.update_after_drop") as euu,
        mock.patch(f"{tds}.NodeFriendship.drop") as node_f,
        mock.patch(f"{tds}.NothingFriendship.drop") as nothing_f,
        mock.patch(f"{tds}._Node.update_after_drop") as nuu,
        mock.patch(f"{tds}.Snowflake._load") as mock_load,
    ):
        snowflake.drop(syndrome)
        assert ec.call_args_list \
            + gc.call_args_list == [mock.call()] * len(snowflake.EDGES)
        assert euu.call_args_list == [mock.call()] * len(snowflake.EDGES)
        assert node_f.call_args_list \
            + nothing_f.call_args_list == [mock.call()] * len(snowflake.NODES)
        assert nuu.call_args_list == [mock.call()] * len(snowflake.NODES)
        mock_load.assert_called_once_with(syndrome)


def test_load(snowflake3: Snowflake):
    h = snowflake3.CODE.SCHEME.WINDOW_HEIGHT
    nodes = (
        (0, h-1),
        (2, h-1),
    )
    snowflake3.NODES[nodes[0]].next_defect = True
    snowflake3._load(set(nodes))
    for v, node in snowflake3.NODES.items():
        assert node.next_defect is (v==nodes[1])


def test_merge(snowflake: Snowflake):
    whole = True
    node_count = len(snowflake.NODES)
    t = 0

    with (
        mock.patch(f"{tds}.Snowflake.append_history") as mock_ah,
        mock.patch(f"{tds}._Node.merging") as mock_merging,
        mock.patch(f"{tds}._Node.update_after_merging") as mock_uam,
    ):
        assert snowflake.merge(whole, 'fine') == t
        mock_ah.assert_not_called()
        assert mock_merging.call_args_list == node_count * [mock.call(whole)]
        assert mock_uam.call_args_list == node_count * [mock.call()]


def test_append_history(snowflake3: Snowflake):
    snowflake3.history = []
    snowflake3.append_history()
    assert type(snowflake3.history) is list
    assert len(snowflake3.history) == 1
    assert type(snowflake3.history[0]) is Snowflake