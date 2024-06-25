import pytest

from localuf.decoders.luf import LUF, MacarNode, _Node, Nodes, Macar, Actis
from localuf.constants import Growth
from localuf.decoders._base_uf import direction

class ConcreteNode(_Node):
    """Instantiable version of `_Node`."""
    def advance(self):
        pass

@pytest.fixture
def ufn(macar: Macar):
    v = (0, 0) if macar.NODES.LUF.CODE.DIMENSION==2 else (0, 0, 0)
    return ConcreteNode(macar.NODES, v)

@pytest.fixture(name="ufn3", params=[
    ("sf3F", "v00"),
    ("sf3T", "v000"),
], ids=['code capacity', 'phenomenological'])
def _ufn3(request):
    """Instance of _Node."""
    sf, v = (request.getfixturevalue(s) for s in request.param)
    decoder = Macar(sf)
    vns = Nodes(decoder)
    return ConcreteNode(vns, v)

@pytest.fixture(name="n3", params=[
    ("sf3F", "v00", Macar),
    ("sf3F", "v00", Actis),
    ("sf3T", "v000", Macar),
    ("sf3T", "v000", Actis),
], ids=[
    'CC-vis',
    'CC-invis',
    'ph-vis',
    'ph-invis',
])
def _n3(request):
    """Instances of MacarNode and ActisNode."""
    sf, v = (request.getfixturevalue(s) for s in request.param[:2])
    _, _, decoder_class = request.param
    decoder = decoder_class(sf)
    ns = decoder.NODES
    return ns.dc[v]

@pytest.fixture
def ufn3F(sf3F, v00):
    decoder = Macar(sf3F)
    vns = Nodes(decoder)
    return ConcreteNode(vns, v00)

@pytest.fixture
def ufn3T(sf3T, v000):
    decoder = Macar(sf3T)
    vns = Nodes(decoder)
    return ConcreteNode(vns, v000)

# @pytest.fixture
# def n3F(luf3F: LUF, v00):
#     nodes = luf3F.nodes
#     return nodes.dc[v00]

# @pytest.fixture
# def n3T(vluf5T: LUF, v000):
#     nodes = luf5T.nodes
#     return nodes.dc[v000]

@pytest.fixture
def n3_with_E(n3: _Node):
    """Node (with correct access) and a pointer string."""
    pointer = 'E'
    e, index = n3.NEIGHBORS[pointer]
    n3.access = {pointer: n3.NODES.dc[e[index]]}
    return n3, pointer

def test_nodes_attribute(ufn3: _Node):
    assert type(ufn3.NODES) is Nodes

def test_v_attribute_2D(ufn3F: _Node, v00):
    assert type(ufn3F.INDEX) is tuple
    assert len(ufn3F.INDEX) == 2
    assert ufn3F.INDEX == v00

def test_v_attribute_3D(ufn3T: _Node, v000):
    assert type(ufn3T.INDEX) is tuple
    assert len(ufn3T.INDEX) == 3
    assert ufn3T.INDEX == v000

def test_id_attribute(ufn: _Node):
    assert type(ufn.ID) is int
    power = ufn.NODES.LUF.CODE.DIMENSION - 1
    assert ufn.ID == 2 * ufn.NODES.LUF.CODE.D**power

def test_neighbors_attribute(ufn: _Node, v00, v000):
    assert type(ufn.NEIGHBORS) is dict
    ph_dc = {
        'W': (((0,-1, 0), v000), 0),
        'E': ((v000, (0, 1, 0)), 1),
        'S': ((v000, (1, 0, 0)), 1),
        'U': ((v000, (0, 0, 1)), 1),
    }
    noise = ufn.NODES.LUF.CODE.NOISE
    if str(noise) == 'code capacity':
        assert ufn.NEIGHBORS == {
            'W': (((0,-1), v00), 0),
            'E': ((v00, (0, 1)), 1),
            'S': ((v00, (1, 0)), 1),
        }
    elif str(noise) == 'phenomenological':
        assert ufn.NEIGHBORS == ph_dc
    elif str(noise) == 'circuit-level':
        assert ufn.NEIGHBORS == ph_dc | {
            'EU': ((v000, (0, 1, 1)), 1),
            'SEU': ((v000, (1, 1, 1)), 1),
        }
    else:
        raise ValueError(f"Unknown noise model {noise}")

@pytest.mark.parametrize("prop", [
    "NODES",
    "INDEX",
    "ID",
    "NEIGHBORS",
])
def test_property_attributes(test_property, ufn3: _Node, prop):
    test_property(ufn3, prop)

def test_reset(ufn3: _Node):
    n3_copy = ConcreteNode(ufn3.NODES, ufn3.INDEX)

    ufn3.defect = True
    ufn3.cid = -1
    ufn3.next_cid = -2
    ufn3.anyon = True
    ufn3.next_anyon = True
    ufn3.pointer = 'N'
    ufn3.active = True
    ufn3.busy = True
    pointer = 'E'
    ufn3.access = {pointer: MacarNode(ufn3.NODES, ufn3.INDEX)}

    ufn3.reset()

    assert not any((
        ufn3.defect,
        ufn3.anyon,
        ufn3.next_anyon,
        ufn3.active,
        ufn3.busy,
    ))
    assert ufn3.cid == ufn3.ID
    assert ufn3.next_cid == ufn3.ID
    assert ufn3.pointer == 'C'
    assert ufn3.access == {}
    
    assert dir(ufn3) == dir(n3_copy)

def test_defect_attribute(ufn3: _Node):
    assert type(ufn3.defect) is bool
    assert ufn3.defect == False

def test_cid_attribute(ufn3: _Node):
    assert type(ufn3.cid) is int
    assert ufn3.cid == ufn3.ID

def test_next_cid_attribute(ufn3: _Node):
    assert type(ufn3.next_cid) is int
    assert ufn3.next_cid == ufn3.ID

def test_anyon_attribute(ufn3: _Node):
    assert type(ufn3.anyon) is bool
    assert ufn3.anyon == False

def test_next_anyon_attribute(ufn3: _Node):
    assert type(ufn3.next_anyon) is bool
    assert ufn3.next_anyon == False
    
def test_pointer_attribute(ufn3: _Node):
    assert type(ufn3.pointer) is str
    assert ufn3.pointer == 'C'
    
def test_active_attribute(ufn3: _Node):
    assert type(ufn3.active) is bool
    assert ufn3.active == False
    
def test_busy_attribute(ufn3: _Node):
    assert type(ufn3.busy) is bool
    assert ufn3.busy == False

def test_access_attribute(ufn3: _Node):
    assert type(ufn3.access) is dict
    assert ufn3.access == {}

def test_make_defect(ufn3: _Node):
    ufn3.make_defect()
    assert all((ufn3.defect, ufn3.anyon, ufn3.active))

def test_growing_inactive(ufn: _Node):
    """Test `growing` when node inactive."""
    ufn.growing()
    assert ufn.busy is False
    assert all(val is Growth.UNGROWN
               for val in ufn.NODES.LUF.growth.values())

def test_growing_active(ufn: _Node):
    """Test `growing` when node active."""
    luf = ufn.NODES.LUF
    ufn.active = True
    for growth_value in [Growth.HALF] + 2*[Growth.FULL]:
        ufn.growing()
        assert ufn.busy is False
        for e in luf.CODE.EDGES:
            if e in luf.CODE.INCIDENT_EDGES[ufn.INDEX]:
                assert luf.growth[e] is growth_value
            else:
                assert luf.growth[e] is Growth.UNGROWN

def test_update_access_2D(macar5F: Macar):
    wce = (0, -1), (0, 0), (0, 1)
    update_access_helper(macar5F, wce)

def test_update_access_3D(macar5T: Macar):
    wce = (0, -1, 0), (0, 0, 0), (0, 1, 0)
    update_access_helper(macar5T, wce)

def update_access_helper(luf: LUF, wce):
    w, c, e = wce
    center = luf.NODES.dc[c]
    luf.growth[w, c] = Growth.FULL
    center.update_access()
    assert center.access == {'W': luf.NODES.dc[w]}
    luf.growth[c, e] = Growth.FULL
    center.update_access()
    assert center.access == {
        'W': luf.NODES.dc[w],
        'E': luf.NODES.dc[e],
    }
    luf.growth[w, c] = Growth.UNGROWN
    center.update_access()
    assert center.access == {'E': luf.NODES.dc[e]}

def test_merging_relay(n3_with_E: tuple[_Node, direction]):
    n3, pointer = n3_with_E
    n3.anyon = True
    n3.pointer = pointer
    n3.merging()
    assert n3.busy
    assert n3.access[pointer].next_anyon
    assert not n3.anyon

def test_merging_never_relay_from_boundary(luf3F: LUF):
    boundary_node = luf3F.NODES.dc[1, -1]
    east_node = luf3F.NODES.dc[1, 0]
    boundary_node.access = {'E': east_node} # type: ignore
    boundary_node.pointer = 'E'
    boundary_node.anyon = True
    east_node.cid = 0
    boundary_node.merging()

    assert not east_node.next_anyon
    assert boundary_node.anyon

    assert boundary_node.busy
    assert boundary_node.pointer == 'E'
    assert boundary_node.next_cid == east_node.cid

def test_merging_no_flood(n3_with_E: tuple[_Node, direction]):
    n3, _ = n3_with_E
    n3.merging()
    assert not n3.busy
    assert n3.pointer == 'C'
    assert n3.next_cid == n3.ID

def test_merging_flood_2D(ns3F: Nodes):
    node = ns3F.dc[1, 0]
    north = ns3F.dc[0, 0]
    west = ns3F.dc[1, -1]
    merging_flood_helper(node, north, west)

def test_merging_flood_3D(ns3T: Nodes):
    node = ns3T.dc[1, 0, 0]
    north = ns3T.dc[0, 0, 0]
    west = ns3T.dc[1, -1, 0]
    merging_flood_helper(node, north, west)

def merging_flood_helper(
        node: _Node,
        north: _Node,
        west: _Node,
):
    node.access = {'N': north, 'W': west}
    node.merging()
    assert node.busy
    assert node.pointer == 'W'
    assert node.next_cid == west.ID

def test_update_after_merge_step(ufn3: _Node):
    ufn3.next_cid = 0
    ufn3.next_anyon = True
    ufn3.update_after_merge_step()
    assert ufn3.cid == 0
    assert ufn3.anyon
    assert not ufn3.next_anyon

def test_presyncing(ufn3: _Node):
    
    ufn3.anyon = True
    ufn3.presyncing()
    assert ufn3.active
    assert not ufn3.busy

    ufn3.anyon = False
    ufn3.presyncing()
    assert not ufn3.active
    assert not ufn3.busy

def test_syncing(n3_with_E: tuple[_Node, direction]):
    vn3, pointer = n3_with_E
    vn3.access[pointer].active = True
    vn3.syncing()
    assert vn3.busy

def test_update_after_sync_step(ufn3: _Node):
    # active, busy = 00
    ufn3.update_after_sync_step()
    assert not ufn3.active
    # active, busy = 01
    ufn3.busy = True
    ufn3.update_after_sync_step()
    assert ufn3.active
    # active, busy = 10
    ufn3.busy = False
    ufn3.update_after_sync_step()
    assert ufn3.active
    # active, busy = 11
    ufn3.busy = True
    ufn3.update_after_sync_step()
    assert ufn3.active