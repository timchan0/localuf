import itertools
from unittest import mock

import pytest

from localuf.decoders.luf import LUF, Controller, Macar

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

@pytest.mark.parametrize("decoder, syndrome", [
    ("macar5F", "syndrome5F"),
    ("macar5T", "syndrome5T"),
])
def test_validate_draw_True(decoder, syndrome, request):
    """From https://miguendes.me/how-to-use-fixtures-as-arguments-in-pytestmarkparametrize."""
    decoder = request.getfixturevalue(decoder)
    syndrome = request.getfixturevalue(syndrome)
    n_steps = decoder.validate(syndrome, draw=True)
    assert n_steps == len(decoder.history)

@pytest.mark.parametrize("sf, syndrome", [
    ("sf5F", "syndrome5F"),
    ("sf5T", "syndrome5T"),
])
def test_validate_draw_consistency(sf, syndrome, request):
    sf = request.getfixturevalue(sf)
    syndrome = request.getfixturevalue(syndrome)
    decoderF = Macar(sf)
    decoderT = Macar(sf)
    nF = decoderF.validate(syndrome)
    nT = decoderT.validate(syndrome, draw=True)
    assert nF == nT

def test_advance_physicals(macar: Macar):
    n_nodes = len(macar.CODE.NODES)
    with (
        mock.patch("localuf.decoders.luf.MacarNode.advance") as vn_advance,
        mock.patch("localuf.decoders.luf.Controller.advance") as c_advance,
    ):
        macar._advance()
        assert vn_advance.call_count == n_nodes
        c_advance.assert_called_once_with()

def test_advance_unphysicals_growth(macar: Macar):
    n_nodes = len(macar.CODE.NODES)
    with mock.patch("localuf.decoders.luf.MacarNode.update_access") as ua:
        macar._advance()
        assert ua.call_count == n_nodes

def test_advance_unphysicals_merge(macar: Macar):
    n_nodes = len(macar.CODE.NODES)
    macar._advance()
    with mock.patch("localuf.decoders.luf.MacarNode.update_after_merge_step") as step:
        macar._advance()
        assert step.call_count == n_nodes

def test_advance_unphysicals_sync(macar: Macar):
    n_nodes = len(macar.CODE.NODES)
    for _ in itertools.repeat(None, 3):
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
    decoder_1 = Macar(sf)
    decoder_2 = Macar(sf)
    decoder_2.validate(syndrome, draw=True)
    decoder_2.syndrome
    decoder_2.reset()
    with pytest.raises(AttributeError, match="syndrome"):
        del decoder_2.syndrome
    assert dir(decoder_1) == dir(decoder_2)
    assert decoder_1.growth == decoder_2.growth
    assert decoder_1.syndrome == decoder_2.syndrome
    assert decoder_1.CONTROLLER.stage == decoder_2.CONTROLLER.stage
    assert not hasattr(decoder_2, 'history')