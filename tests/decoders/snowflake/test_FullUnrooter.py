from localuf.decoders import Snowflake
from localuf.decoders.snowflake import _Node
from localuf.type_aliases import Node
from localuf.decoders.snowflake.constants import RESET
from localuf.decoders._base_uf import direction


def test_start(sfn3: _Node):
    sfn3.pointer = 'D'
    sfn3.UNROOTER.start()
    assert sfn3.cid == RESET
    assert sfn3.pointer == 'C'


def test_flooding(syncing_flooding_objects: tuple[
    Snowflake, tuple[Node, Node, Node], tuple[_Node, _Node, _Node]
]):
    snowflake3, _, (west, center, east) = syncing_flooding_objects
    access: dict[direction, _Node] = {
        'W': west,
        'E': east,
    }

    # TEST FINISH UNROOTING
    center.cid = RESET
    center.next_cid = west.ID
    center.UNROOTER.flooding_whole()
    assert center.busy
    assert center.next_cid == center.ID
    assert center.next_unrooted
    center.reset()

    # TEST START UNROOTING
    west.cid = RESET
    center.access = access
    center.pointer = 'W'
    center.UNROOTER.flooding_whole()
    assert center.busy
    assert center.next_cid == RESET
    assert center.pointer == 'C'
    snowflake3.reset()

    # TEST UPDATE `pointer`, `cid`
    center.access = access
    center.UNROOTER.flooding_half()
    assert center.busy
    assert center.pointer == 'W'
    assert center.next_cid == west.ID