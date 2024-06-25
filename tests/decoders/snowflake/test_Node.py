from unittest import mock

import pytest

from localuf.constants import Growth
from localuf.decoders.snowflake.constants import RESET
from localuf.type_aliases import Node
from localuf.decoders import Snowflake
from localuf.decoders.snowflake import _Node, NothingFriendship, NodeFriendship, TopSheetFriendship
from localuf.decoders._base_uf import direction

@pytest.mark.parametrize("prop", [
    "SNOWFLAKE",
    "INDEX",
    "ID",
    "FRIENDSHIP",
    "NEIGHBORS",
    "UNROOTER",
])
def test_property_attributes(test_property, sfn3: _Node, prop):
    test_property(sfn3, prop)


def test_init(sfn3: _Node):
    with mock.patch("localuf.decoders.snowflake._Node.reset") as mock_reset:
        sfn3.__init__(sfn3.SNOWFLAKE, sfn3.INDEX)
        mock_reset.assert_called_once_with()


def test_SNOWFLAKE_property(sfn3: _Node):
    assert type(sfn3.SNOWFLAKE) is Snowflake
    assert sfn3.SNOWFLAKE.NODES[sfn3.INDEX] == sfn3


def test_INDEX_property(sfn3: _Node, fixture_test_INDEX_property):
    fixture_test_INDEX_property(sfn3)


def test_ID_property(sfn3: _Node):
    assert type(sfn3.ID) is int


def test_FRIENDSHIP_property(snowflake: Snowflake):
    for v, node in snowflake.NODES.items():
        if v[snowflake.CODE.TIME_AXIS] == 0:
            assert type(node.FRIENDSHIP) is NothingFriendship
        elif v[snowflake.CODE.TIME_AXIS] == snowflake.CODE.SCHEME.WINDOW_HEIGHT-1:
            assert type(node.FRIENDSHIP) is TopSheetFriendship
        else:
            assert type(node.FRIENDSHIP) is NodeFriendship


def test_NEIGHBORS_property(snowflake3: Snowflake, sfn3: _Node):
    for v, node in snowflake3.NODES.items():
        assert type(node.NEIGHBORS) is dict
        is_top_detector = (
            v[snowflake3.CODE.TIME_AXIS]==snowflake3.CODE.SCHEME.WINDOW_HEIGHT-1
        ) and not node._IS_BOUNDARY
        assert len(node.NEIGHBORS) + is_top_detector == len(snowflake3.CODE.INCIDENT_EDGES[v])
    assert sfn3.NEIGHBORS == {
        'W': (((-1, 0), (0, 0)), 0),
        'E': (((0, 0), (1, 0)), 1),
        'U': (((0, 0), (0, 1)), 1),
    }


def test_busy_attribute(sfn3: _Node):
    assert sfn3.busy is False


def test_reset(sfn3: _Node):

    sfn3.defect = True
    sfn3.active = True
    sfn3.cid = -1
    sfn3.pointer = 'U'
    sfn3.busy = True

    sfn3.next_defect = True
    sfn3.next_active = True
    sfn3.next_cid = -1

    sfn3.access = {'U': sfn3.SNOWFLAKE.NODES[(0, 1)]}

    sfn3.reset()

    assert sfn3.defect is False
    assert sfn3.active is False
    assert sfn3.cid == sfn3.ID
    assert sfn3.pointer == 'C'
    assert sfn3.busy is False

    assert sfn3.next_defect is False
    assert sfn3.next_active is False
    assert sfn3.next_cid == sfn3.ID

    assert sfn3.access == {}


def test_update_after_drop(sfn3: _Node):
    sfn3.next_defect = True
    sfn3.next_active = True
    sfn3.next_cid = RESET
    sfn3.next_pointer = 'U'

    sfn3.update_after_drop()

    assert sfn3.defect
    assert sfn3.active
    assert sfn3.cid == RESET
    assert sfn3.pointer == 'U'
    
    assert sfn3.next_defect
    assert sfn3.next_active
    assert sfn3.next_cid == RESET
    assert sfn3.next_pointer == 'U'


def test_growing_calls_find_broken_pointers(sfn3: _Node):
    with mock.patch("localuf.decoders.snowflake.NothingFriendship.find_broken_pointers") as mock_find:
        sfn3.grow()
        mock_find.assert_called_once_with()


def test_growing_inactive(sfn3: _Node):
    """Test `growing` when node inactive."""
    sfn3.grow()
    assert sfn3.busy is False
    assert all(edge.growth is Growth.UNGROWN
               for edge in sfn3.SNOWFLAKE.EDGES.values())


def test_growing_active(sfn3: _Node):
    """Test `growing` when node active."""
    sfn3.active = True
    for growth in [Growth.HALF] + 2*[Growth.FULL]:
        sfn3.grow()
        assert sfn3.busy is False
        for e, edge in sfn3.SNOWFLAKE.EDGES.items():
            if sfn3.INDEX in e:
                assert edge.growth is growth
            else:
                assert edge.growth is Growth.UNGROWN


def test_update_access(snowflake3: Snowflake):
    w, c, e = (-1, 0), (0, 0), (1, 0)
    center = snowflake3.NODES[c]
    center.update_access()
    assert center.access == {}
    snowflake3.EDGES[w, c].growth = Growth.FULL
    center.update_access()
    assert center.access == {'W': snowflake3.NODES[w]}
    snowflake3.EDGES[c, e].growth = Growth.FULL
    center.update_access()
    assert center.access == {
        'W': snowflake3.NODES[w],
        'E': snowflake3.NODES[e]
    }
    snowflake3.EDGES[w, c].growth = Growth.UNGROWN
    center.update_access()
    assert center.access == {'E': snowflake3.NODES[e]}


def test_merging(sfn3: _Node):
    sfn3.busy = True
    with (
        mock.patch("localuf.decoders.snowflake._Node.syncing") as mock_syncing,
        mock.patch("localuf.decoders.snowflake._Node.flooding") as mock_flooding,
    ):
        sfn3.merging()
        assert sfn3.busy is False
        mock_syncing.assert_called_once_with()
        mock_flooding.assert_called_once_with()


def test_syncing(syncing_flooding_objects: tuple[
    Snowflake, tuple[Node, Node, Node], tuple[_Node, _Node, _Node]
]):
    snowflake3, (w, c, e), (west, center, east) = syncing_flooding_objects

    # TEST PUSH DEFECT
    center.pointer = 'E'
    center.defect = True
    center.next_defect = True
    center.access = {'E': east}
    center.syncing()
    assert center.busy is True
    assert east.next_defect is True
    assert center.defect is True
    assert center.next_defect is False
    assert snowflake3.EDGES[c, e].correction is True
    snowflake3.reset()

    # TEST BOUNDARY PUSHES NOT ITS DEFECT
    west.pointer = 'E'
    west.defect = True
    west.next_defect = True
    west.access = {'E': center}
    west.syncing()
    assert west.busy is False
    assert center.next_defect is False
    assert west.defect is True
    assert west.next_defect is True
    assert snowflake3.EDGES[w, c].correction is False
    snowflake3.reset()

    # TEST UPDATE ACTIVE
    # boundary root never active
    for defect in (False, True):
        west.defect = defect
        west.syncing()
        assert west.next_active is False
        assert west.busy is False
        west.reset()
    # nonboundary root without defect
    center.syncing()
    assert center.next_active is False
    assert center.busy is False
    center.reset()
    # nonboundary root with defect
    center.defect = True
    center.syncing()
    assert center.next_active is True
    assert center.busy is True
    center.reset()
    # else
    center.pointer = 'E'
    center.access = {'E': east}
    east.active = True
    center.syncing()
    assert center.next_active is True
    assert center.busy is True


def test_flooding(sfn3: _Node):
    with mock.patch("localuf.decoders.snowflake.main._FullUnrooter.flooding") as mock_flooding:
        sfn3.flooding()
        mock_flooding.assert_called_once_with()


def test_update_after_merging(sfn3: _Node):
    sfn3.next_defect = True
    sfn3.next_cid = -1
    sfn3.next_defect = True
    sfn3.next_active = True
    sfn3.update_after_merging()
    assert sfn3.cid == -1
    assert sfn3.defect
    assert sfn3.active
    assert sfn3.next_defect
    assert sfn3.next_active