from unittest import mock

import pytest

from localuf._base_classes import Code
from localuf.codes import Repetition, Surface
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


class TestNEIGHBORSProperty:

    def _general_checks(self, code: Code, decoder: Snowflake):
        for v, node in decoder.NODES.items():
            assert type(node.NEIGHBORS) is dict
            is_top_detector = (
                v[code.TIME_AXIS]==code.SCHEME.WINDOW_HEIGHT-1
            ) and not node._IS_BOUNDARY
            if isinstance(code, Repetition):
                assert len(node.NEIGHBORS) + is_top_detector == len(code.INCIDENT_EDGES[v])
            else:
                if is_top_detector:
                    assert len(node.NEIGHBORS) < len(code.INCIDENT_EDGES[v])
                else:
                    assert len(node.NEIGHBORS) == len(code.INCIDENT_EDGES[v])


    def test_default_order(self, snowflake3: Snowflake, sfn3: _Node):
        self._general_checks(snowflake3.CODE, snowflake3)
        assert sfn3.NEIGHBORS == {
            'W': (((-1, 0), (0, 0)), 0),
            'E': (((0, 0), (1, 0)), 1),
            'U': (((0, 0), (0, 1)), 1),
        }


    def test_repetition_custom_order(self, frugal_rep_3: Repetition):
        decoder = Snowflake(
            code=frugal_rep_3,
            _neighbor_order=('U', 'W', 'E', 'D')
        )
        self._general_checks(frugal_rep_3, decoder)
        assert tuple(decoder.NODES[0, 0].NEIGHBORS.items()) == (
            ('U', (((0, 0), (0, 1)), 1)),
            ('W', (((-1, 0), (0, 0)), 0)),
            ('E', (((0, 0), (1, 0)), 1)),
        )

    def test_surface_custom_order(self, surface3_CL_frugal: Surface):
        decoder = Snowflake(
            surface3_CL_frugal,
            _neighbor_order=(
                'U',
                'NU',
                'EU',
                'SEU',
                'N',
                'W',
                'E',
                'S',
                'NWD',
                'WD',
                'SD',
                'D',
            )
        )
        self._general_checks(surface3_CL_frugal, decoder)
        assert tuple(decoder.NODES[0, 0, 0].NEIGHBORS.items()) == (
            ('U', (((0, 0, 0), (0, 0, 1)), 1)),
            ('EU', (((0, 0, 0), (0, 1, 1)), 1)),
            ('SEU', (((0, 0, 0), (1, 1, 1)), 1)),
            ('W', (((0, -1, 0), (0, 0, 0)), 0)),
            ('E', (((0, 0, 0), (0, 1, 0)), 1)),
            ('S', (((0, 0, 0), (1, 0, 0)), 1)),
        )


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


@pytest.mark.parametrize("active", (False, True))
def test_grow(sfn3: _Node, active):
    """Test `grow`."""
    sfn3.active = active
    with (
        mock.patch("localuf.decoders.snowflake._Node._grow") as mock_grow,
        mock.patch("localuf.decoders.snowflake.NothingFriendship.find_broken_pointers") as mock_find,
    ):
        sfn3.grow()
        assert sfn3.busy is False
        if active:
            mock_grow.assert_called_once_with()
        else:
            mock_grow.assert_not_called()
        mock_find.assert_called_once_with()


@pytest.mark.parametrize("active", (False, True))
@pytest.mark.parametrize("whole", (False, True))
def test_grow_whole(sfn3: _Node, active, whole):
    sfn3.active = active
    sfn3.whole = whole
    with (
        mock.patch("localuf.decoders.snowflake._Node._grow") as mock_grow,
        mock.patch("localuf.decoders.snowflake.NothingFriendship.find_broken_pointers") as mock_find,
    ):
        sfn3.grow_whole()
        assert sfn3.busy is False
        if active and whole:
            mock_grow.assert_called_once_with()
            assert sfn3.grown
            assert sfn3.next_grown
        else:
            mock_grow.assert_not_called()
            assert not sfn3.grown
            assert not sfn3.next_grown
        mock_find.assert_called_once_with()


@pytest.mark.parametrize("active", (False, True))
@pytest.mark.parametrize("whole", (False, True))
@pytest.mark.parametrize("grown", (False, True))
def test_grow_half(sfn3: _Node, active, whole, grown):
    sfn3.active = active
    sfn3.whole = whole
    sfn3.grown = grown
    sfn3.unrooted = True
    sfn3.next_unrooted = True
    with (
        mock.patch("localuf.decoders.snowflake._Node._grow") as mock_grow,
        mock.patch("localuf.decoders.snowflake.NothingFriendship.find_broken_pointers") as mock_find,
    ):
        sfn3.grow_half()
        assert not sfn3.unrooted
        assert not sfn3.next_unrooted
        assert sfn3.busy is False
        if active and not whole and not grown:
            mock_grow.assert_called_once_with()
        else:
            mock_grow.assert_not_called()
        assert sfn3.grown is grown
        assert sfn3.next_grown is False
        mock_find.assert_not_called()


def test_grow_subroutine(sfn3: _Node):
    """Test `_grow`."""
    for growth in [Growth.HALF] + 2*[Growth.FULL]:
        sfn3._grow()
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


@pytest.mark.parametrize("whole", (False, True))
def test_merging(sfn3: _Node, whole):
    sfn3.busy = True
    with (
        mock.patch("localuf.decoders.snowflake._Node.syncing") as mock_syncing,
        mock.patch("localuf.decoders.snowflake._Node.flooding") as mock_flooding,
    ):
        sfn3.merging(whole)
        assert sfn3.busy is False
        mock_syncing.assert_called_once_with()
        mock_flooding.assert_called_once_with(whole)


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


@pytest.mark.parametrize("whole", (False, True))
def test_flooding(sfn3: _Node, whole):
    with (
        mock.patch("localuf.decoders.snowflake.main._FullUnrooter.flooding_whole") as mock_whole,
        mock.patch("localuf.decoders.snowflake.main._FullUnrooter.flooding_half") as mock_half,
    ):
        sfn3.flooding(whole)
        if whole:
            mock_whole.assert_called_once_with()
            mock_half.assert_not_called()
        else:
            mock_whole.assert_not_called()
            mock_half.assert_called_once_with()


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