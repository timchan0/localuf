import itertools

import pytest

from localuf import Repetition, Surface
from localuf.type_aliases import Node, Edge
from localuf.constants import Growth
from localuf.decoders import Snowflake
from localuf.decoders.snowflake import _Node

whole = True

@pytest.fixture
def snowflake3tall():
    d = 3
    rp = Repetition(
        d,
        noise='phenomenological',
        scheme='frugal',
        buffer_height=3*(d//2),
    )
    return Snowflake(rp)


@pytest.fixture
def snowflake4():
    d = 4
    rp = Repetition(
        d,
        noise='phenomenological',
        scheme='frugal',
    )
    return Snowflake(rp, merger='slow', schedule='1:1')


@pytest.fixture
def snowflake5():
    d = 5
    rp = Repetition(
        d,
        noise='phenomenological',
        scheme='frugal',
    )
    return Snowflake(rp, schedule='1:1')


@pytest.fixture
def snowflake7():
    d = 7
    rp = Repetition(
        d,
        noise='phenomenological',
        scheme='frugal',
        buffer_height=3*(d//2),
    )
    return Snowflake(rp, merger='slow', schedule='1:1')


def test_nonroot_boundary_defect_kept(snowflake4: Snowflake):
    """Ensure a nonroot boundary node w/ defect loses not its defect."""
    h = snowflake4.CODE.SCHEME.WINDOW_HEIGHT
    for syndrome in [
        {(0, h-1), (2, h-1)},
        set(),
        {(0, h-1)},
        set(),
    ]:
        snowflake4.decode(syndrome)
    snowflake4.drop(set())
    snowflake4._SCHEDULE.grow()
    for _ in itertools.repeat(None, 5):
        for node in snowflake4.NODES.values():
            node.merging(whole)
        for node in snowflake4.NODES.values():
            node.update_after_merging()
        assert snowflake4.NODES[-1, 0].defect


def assert_no_standoff(s: Snowflake):
    """Assert no pointer standoff in the snowflake."""
    for u, v in s.EDGES.keys():
        node_u = s.NODES[u]
        node_v = s.NODES[v]
        assert_no_standoff_between(node_u, node_v)

def assert_no_standoff_between(node_u: _Node, node_v: _Node):
    """Assert no pointer standoff between `node_u` and `node_v`."""
    if node_u.pointer != 'C' and node_v.pointer != 'C':
        assert not all((
                node_u.access[node_u.pointer] == node_v,
                node_v.access[node_v.pointer] == node_u,
            ))

def test_decode_coarse(snowflake3tall: Snowflake):
    """Test adjacent pointers never point toward each other
    after each decode.
    """
    h = snowflake3tall.CODE.SCHEME.WINDOW_HEIGHT
    for syndrome in itertools.chain(
        itertools.repeat({(1, h-1)}, 2),
        itertools.repeat({(0, h-1), (1, h-1)}, 1),
        itertools.repeat(set(), 2*h-1),
    ):
        snowflake3tall.decode(syndrome)
        assert_no_standoff(snowflake3tall)


def test_decode_fine(snowflake5: Snowflake):
    """Test adjacent pointers never point toward each other
    after each merging timestep.
    """
    h = snowflake5.CODE.SCHEME.WINDOW_HEIGHT
    snowflake5.decode({(0, h-1)})
    snowflake5.decode({(1, h-1)})
    snowflake5.decode({(0, h-1)})
    for _ in itertools.repeat(None, 2*h-1):
        # DECODE
        snowflake5.drop(set())
        snowflake5._SCHEDULE.grow()
        while True:
            for node in snowflake5.NODES.values():
                node.merging(whole)
            for node in snowflake5.NODES.values():
                node.update_after_merging()
            assert_no_standoff(snowflake5)
            if not any(node.busy for node in snowflake5.NODES.values()):
                break


def test_snake(snowflake5: Snowflake):
    """Test scenario which 'keep & peel' method fails.
    
    This is because the ((0, 0), (1, 0)) edge incorrectly points west after first drop.
    """
    for e in {
        ((0, 1), (1, 1)),
        ((1, 1), (2, 1)),
        ((0, 2), (1, 2)),
        ((1, 2), (1, 3)),
        ((0, 3), (1, 3)),
        ((-1, 3), (0, 3)),
        ((0, 1), (0, 2)),
        ((2, 1), (2, 2)),
    }:
        snowflake5.EDGES[e].growth = Growth.FULL
    for _ in itertools.repeat(None, 4):
        snowflake5.decode(set())
        assert_no_standoff(snowflake5)


def test_no_lost_defect(snowflake5: Snowflake):
    """Test no defect in bottom layer of detectors after each growth round.
    
    This uses a J-shaped cluster whose root is a boundary node
    at the bottom end of the J.

    The method that fails this test is the one that on window raise preserves...
    * CID structure unless CID is one of the lowest boundary nodes
    (in which case CID resets)
    * pointer structure unless pointer points down to nothing
    (in which case pointer resets)
    """
    h = snowflake5.CODE.SCHEME.WINDOW_HEIGHT
    for syndrome in itertools.chain(
        [
            {(2, h-1), (0, h-1)},
            {(2, h-1)},
            {(2, h-1), (0, h-1)},
        ],
        itertools.repeat({(2, h-1)}, 2),
        itertools.repeat(set(), 2*h),
    ):
        snowflake5.decode(syndrome)
        for j in range(snowflake5.CODE.D-1):
            assert not snowflake5.NODES[j, 0].defect


def assert_lowest_nodes_never_point_down(decoder: Snowflake, syndrome: set[Node]):
    """Call `decoder.decode(syndrome)` then assert lowest nodes never point down."""
    decoder.decode(syndrome)
    for j in range(decoder.CODE.D-1):
        assert decoder.NODES[j, 0].pointer != 'D'


def test_no_pointer_standoff_after_keep(snowflake5: Snowflake):
    """Test no pointer standoff after a cluster merges with a kept cluster (legacy)."""
    h = snowflake5.CODE.SCHEME.WINDOW_HEIGHT
    for syndrome in itertools.chain(
        [
            {(3, h-1)},
            {(2, h-1)},
            {(1, h-1)},
            {(3, h-1)},
        ],
        itertools.repeat(set(), 2*h-1),
    ):
        assert_lowest_nodes_never_point_down(snowflake5, syndrome)


def test_detached_cluster_keep(snowflake5: Snowflake):
    """Test unpeeled nodes have `keep` True when in a detached cluster (legacy).
    
    This uses a U-shaped cluster with no boundary node,
    whose root is at the top left end of the U.

    This test fails if the bottom of the cluster disappears
    before all of the cluster is under the floor.
    """
    h = snowflake5.CODE.SCHEME.WINDOW_HEIGHT
    for syndrome in itertools.chain(
        [
            {(0, h-1), (1, h-1), (2, h-1), (3, h-1)},
            {(0, h-1), (3, h-1)},
        ],
        itertools.repeat(set(), 2*h-1),
    ):
        assert_lowest_nodes_never_point_down(snowflake5, syndrome)


def helper_infinite_due_to_unroot(snowflake5: Snowflake):
    h = snowflake5.CODE.SCHEME.WINDOW_HEIGHT
    for syndrome in [
        {(2, h-1),},
        {(2, h-1), (3, h-1),},
        {(1, h-1),},
        {(2, h-1),},
        set(),
        set(),
    ]:
        snowflake5.decode(syndrome)
    snowflake5.drop(set())
    snowflake5._SCHEDULE.grow()

def test_no_infinite_unroot_cycle(snowflake5: Snowflake):
    """Test unroot wave never cycles infinitely.
    
    This fails if:
    * unroot wave propagates along pointers rather than access,
    * and `unrooted` is not used.
    """
    helper_infinite_due_to_unroot(snowflake5)
    for _ in itertools.repeat(None, 10):  # prevent infinite loop
        for node in snowflake5.NODES.values():
            node.merging(whole)
        for node in snowflake5.NODES.values():
            node.update_after_merging()
        if not any(node.busy for node in snowflake5.NODES.values()):
            break
    for v in [
        (0, 0),
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1),
        (2, 2),
        (3, 1),
    ]:
        assert snowflake5.NODES[v].cid == 20

def test_no_unroot_pointer_cycle(snowflake5: Snowflake):
    """Test unroot never leads to pointer cycle.
    
    This fails if unroot wave propagates along pointers rather than access.
    """
    helper_infinite_due_to_unroot(snowflake5)
    for _ in itertools.repeat(None, 10):  # prevent infinite loop
        for node in snowflake5.NODES.values():
            node.merging(whole)
        for node in snowflake5.NODES.values():
            node.update_after_merging()
        assert not all((
            snowflake5.NODES[1, 0].pointer == 'E',
            snowflake5.NODES[2, 0].pointer == 'U',
            snowflake5.NODES[2, 1].pointer == 'W',
            snowflake5.NODES[1, 1].pointer == 'D',
        ))
        if not any(node.busy for node in snowflake5.NODES.values()):
            break


def test_no_infinite_loop(snowflake7: Snowflake):
    """Test infinite loop scenario avoided.
    
    The infinite loop tested is a defect bouncing between 2 nodes forever.
    """
    h = snowflake7.CODE.SCHEME.WINDOW_HEIGHT
    snowflake7.decode({(0, h-1), (1, h-1)})
    for _ in itertools.repeat(None, 6):
        snowflake7.decode({(1, h-1)})
    snowflake7.decode({(1, h-1), (3, h-1)})
    snowflake7.decode({(0, h-1), (1, h-1)})
    snowflake7.decode(set())

    # DECODE
    snowflake7.drop(set())
    snowflake7._SCHEDULE.grow()
    for cycle_index in range(1, 11):
        for node in snowflake7.NODES.values():
            node.merging(whole)
        for node in snowflake7.NODES.values():
            node.update_after_merging()
        if cycle_index == 11:
            for v, node in snowflake7.NODES.items():
                # check there's 1 defect left
                if not snowflake7.CODE.is_boundary(v):
                    assert node.defect is (v==(1, 7))
                # check that defect won't bounce between 2 nodes forever
                node_u = snowflake7.NODES[0, 7]
                node_v = snowflake7.NODES[1, 7]
                assert_no_standoff_between(node_u, node_v)
        if not any(node.busy for node in snowflake7.NODES.values()):
            break


class TestLowestEdges:

    @staticmethod
    def _check_edge_indices(decoder: Snowflake, indices: tuple[Edge, ...]):
        for index, edge in zip(indices, decoder._LOWEST_EDGES, strict=True):
            assert edge.INDEX == index


    def test_repetition(self, snowflake3: Snowflake):
        indices = (
            ((0, 0), (0, 1)),
            ((1, 0), (1, 1)),
            ((-1, 0), (0, 0)),
            ((0, 0), (1, 0)),
            ((1, 0), (2, 0)),
        )
        self._check_edge_indices(snowflake3, indices)


    def test_surface(self, surface3_CL_frugal: Surface):
        """Verified `indices` by drawing each edge in a notebook."""
        decoder = Snowflake(surface3_CL_frugal)
        indices = (
            ((0, 0, 0), (0, 0, 1)),
            ((0, 1, 0), (0, 1, 1)),
            ((1, 0, 0), (1, 0, 1)),
            ((1, 1, 0), (1, 1, 1)),
            ((2, 0, 0), (2, 0, 1)),
            ((2, 1, 0), (2, 1, 1)),
            ((0, 0, 1), (1, 0, 0)),
            ((0, 1, 1), (1, 1, 0)),
            ((1, 0, 1), (2, 0, 0)),
            ((1, 1, 1), (2, 1, 0)),
            ((0, 0, 0), (0, 1, 1)),
            ((1, 0, 0), (1, 1, 1)),
            ((2, 0, 0), (2, 1, 1)),
            ((0, 0, 0), (1, 1, 1)),
            ((1, 0, 0), (2, 1, 1)),
            ((0, 0, 0), (1, 0, 0)),
            ((0, 1, 0), (1, 1, 0)),
            ((1, 0, 0), (2, 0, 0)),
            ((1, 1, 0), (2, 1, 0)),
            ((0, -1, 0), (0, 0, 0)),
            ((0, 0, 0), (0, 1, 0)),
            ((0, 1, 0), (0, 2, 0)),
            ((1, -1, 0), (1, 0, 0)),
            ((1, 0, 0), (1, 1, 0)),
            ((1, 1, 0), (1, 2, 0)),
            ((2, -1, 0), (2, 0, 0)),
            ((2, 0, 0), (2, 1, 0)),
            ((2, 1, 0), (2, 2, 0)),
        )
        self._check_edge_indices(decoder, indices)