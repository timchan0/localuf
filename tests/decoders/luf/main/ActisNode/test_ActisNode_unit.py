from collections import Counter
from unittest import mock

import pytest

from localuf.decoders.luf import ActisNodes, ActisNode, NodeFriendship, Actis
from localuf.decoders.luf.constants import Stage

@pytest.fixture
def in_(actis_nodes: ActisNodes):
    v = (0, 0) if actis_nodes.LUF.CODE.DIMENSION==2 else (0, 0, 0)
    return ActisNode(actis_nodes, v)

@pytest.fixture(name="in3", params=[
    ("sf3F", "v00"),
    ("sf3T", "v000"),
])
def _in3(request):
    sf, v = (request.getfixturevalue(s) for s in request.param)
    actis = Actis(sf)
    ins = ActisNodes(actis)
    return ActisNode(ins, v)

@pytest.fixture
def in3F(sf3F, v00):
    actis = Actis(sf3F)
    ins = ActisNodes(actis)
    return ActisNode(ins, v00)

@pytest.fixture
def in3T(sf3T, v000):
    actis = Actis(sf3T)
    ins = ActisNodes(actis)
    return ActisNode(ins, v000)

@pytest.fixture
def in3_with_E(in3: ActisNode):
    """ActisNode (with correct access) and a pointer string."""
    pointer = 'E'
    e, index = in3.NEIGHBORS[pointer]
    in3.access = {pointer: in3.NODES.dc[e[index]]}
    return in3, pointer

def test_init(in_: ActisNode):

    d = in_.NODES.LUF.CODE.D
    em = in_.NODES.LUF.CODE.NOISE
    if str(em) == 'code capacity':
        assert in_.SPAN == 2 * (d-1)
    elif str(em) == 'phenomenological':
        assert in_.SPAN == 3 * (d-1)
    elif str(em) == 'circuit-level':
        assert in_.SPAN == d-1

    assert type(in_.FRIENDSHIP) is NodeFriendship

    with mock.patch("localuf.decoders.luf._Node.__init__") as mock_init:
        in_.__init__(in_.NODES, in_.INDEX)
        mock_init.assert_called_once_with(in_.NODES, in_.INDEX)


def test_countdown_start(actis_nodes: ActisNodes):
    d = actis_nodes.LUF.CODE.D
    counter = Counter(node.SPAN for node in actis_nodes.dc.values())
    if str(actis_nodes.LUF.CODE.NOISE) == 'code capacity':
        for cs in range(2*d):
            if cs < d:
                assert counter[cs] == cs + 1
            else:
                assert counter[cs] == 2*d - cs
    elif str(actis_nodes.LUF.CODE.NOISE) == 'phenomenological':
        # cs goes up somewhat as triangular numbers
        pass
    elif str(actis_nodes.LUF.CODE.NOISE) == 'circuit-level':
        for cs in range(1, d+1):
            small = d - cs
            assert counter[cs] == (small+1)**3 - small**3
        assert counter[0] == d**2


def test_advance(in_: ActisNode):
    # from https://stackoverflow.com/a/63690318/20887677
    with (
        mock.patch("localuf.decoders.luf.ActisNode.advance_definite") as definite_mock,
        mock.patch("localuf.decoders.luf.ActisNode.advance_indefinite") as indefinite_mock,
    ):
        in_.advance()
        definite_mock.assert_called_once_with('growing')
        in_.stage = Stage.MERGING
        in_.advance()
        indefinite_mock.assert_called_once_with('merging')
        in_.stage = Stage.PRESYNCING
        in_.advance()
        definite_mock.assert_called_with('presyncing')
        in_.stage = Stage.SYNCING
        in_.advance()
        indefinite_mock.assert_called_with('syncing')

def test_advance_definite(in_: ActisNode):
    with mock.patch("localuf.decoders.luf.ActisNode.growing") as growing_mock:
        in_.advance_definite('growing')

        growing_mock.assert_called_once_with()
    assert in_.next_stage is Stage.MERGING

    in_.countdown = 1
    in_.advance_definite('growing')

    assert in_.countdown == 0

def test_advance_indefinite(in_: ActisNode):
    in_.stage = Stage.MERGING
    with mock.patch(
        "localuf.decoders.luf.NodeFriendship.update_stage",
        return_value=True
    ) as us_mock:
        in_.advance_indefinite('merging')

        us_mock.assert_called_once_with()
        assert in_.countdown == in_.SPAN

    with (
        mock.patch(
            "localuf.decoders.luf.NodeFriendship.update_stage",
            return_value=False,
        ) as _,
        mock.patch("localuf.decoders.luf.ActisNode.merging") as merging_mock,
        mock.patch("localuf.decoders.luf.NodeFriendship.relay_signals") as rs_mock,
    ):
        in_.advance()

        merging_mock.assert_called_once_with()
        rs_mock.assert_called_once_with()