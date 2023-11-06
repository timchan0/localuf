from unittest import mock

import pytest

from localuf.decoders.luf import LUF, Controller

def test_CONTROLLER_attribute(luf: LUF):
    assert type(luf.CONTROLLER) is Controller

def test_nodes_attribute(luf: LUF):
    d = luf.CODE.D
    assert type(luf.NODES.dc) is dict
    if luf.CODE.DIMENSION == 2:
        assert len(luf.NODES.dc) == d * (d+1)
    else:
        assert luf.CODE.DIMENSION == 3
        assert len(luf.NODES.dc) == d**2 * (d+1)

@pytest.mark.parametrize("prop", [
    "CONTROLLER",
    "NODES",
    "VISIBLE",
    "_pointer_digraph",
])
def test_property_attributes(test_property, luf: LUF, prop):
    test_property(luf, prop)

@pytest.mark.parametrize("luf, syndrome", [
    ("macar5F", "syndrome5F"),
    ("macar5T", "syndrome5T"),
])
def test_validate_draw_True(luf, syndrome, request):
    """From https://miguendes.me/how-to-use-fixtures-as-arguments-in-pytestmarkparametrize."""
    luf = request.getfixturevalue(luf)
    syndrome = request.getfixturevalue(syndrome)
    n_steps = luf.validate(syndrome, draw=True)
    assert n_steps == len(luf.history)

@pytest.mark.parametrize("sf, syndrome", [
    ("sf5F", "syndrome5F"),
    ("sf5T", "syndrome5T"),
])
def test_validate_draw_consistency(sf, syndrome, request):
    sf = request.getfixturevalue(sf)
    syndrome = request.getfixturevalue(syndrome)
    lufF = LUF(sf)
    lufT = LUF(sf)
    nF = lufF.validate(syndrome)
    nT = lufT.validate(syndrome, draw=True)
    assert nF == nT

def test_advance_physicals(macar: LUF):
    n_nodes = len(macar.CODE.NODES)
    with (
        mock.patch("localuf.decoders.luf.MacarNode.advance") as vn_advance,
        mock.patch("localuf.decoders.luf.Controller.advance") as c_advance,
    ):
        macar._advance()
        assert vn_advance.call_count == n_nodes
        c_advance.assert_called_once_with()

def test_advance_unphysicals_growth(macar: LUF):
    n_nodes = len(macar.CODE.NODES)
    with mock.patch("localuf.decoders.luf.MacarNode.update_accessibles") as ua:
        macar._advance()
        assert ua.call_count == n_nodes

def test_advance_unphysicals_merge(macar: LUF):
    n_nodes = len(macar.CODE.NODES)
    macar._advance()
    with mock.patch("localuf.decoders.luf.MacarNode.update_after_merge_step") as step:
        macar._advance()
        assert step.call_count == n_nodes

def test_advance_unphysicals_sync(macar: LUF):
    n_nodes = len(macar.CODE.NODES)
    for _ in range(3):
        macar._advance()
    with mock.patch("localuf.decoders.luf.MacarNode.update_after_sync_step") as uass:
        macar._advance()
        assert uass.call_count == n_nodes

@pytest.mark.parametrize("sf, syndrome", [
    ("sf5F", "syndrome5F"),
    ("sf5T", "syndrome5T"),
])
def test_reset(sf, syndrome, request):
    sf = request.getfixturevalue(sf)
    syndrome = request.getfixturevalue(syndrome)
    luf1 = LUF(sf)
    luf2 = LUF(sf)
    luf2.validate(syndrome, draw=True)
    luf2.syndrome
    luf2.reset()
    with pytest.raises(AttributeError, match="syndrome"):
        del luf2.syndrome
    assert dir(luf1) == dir(luf2)
    assert luf1.growth == luf2.growth
    assert luf1.syndrome == luf2.syndrome
    assert luf1.CONTROLLER.stage == luf2.CONTROLLER.stage
    assert not hasattr(luf2, 'history')