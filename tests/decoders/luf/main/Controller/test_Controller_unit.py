"""Unit test the `Controller` class.

These tests use a 2D distance-3 controller fixture.
Need not repeat on a 3D fixture.
"""
from unittest import mock

import pytest

from localuf import Surface
from localuf.decoders.luf import Controller, Macar
from localuf.decoders.luf.constants import Stage

def test_luf_attribute(c3: Controller):
    assert type(c3.LUF) is Macar

@pytest.mark.parametrize("prop", [
    "LUF",
])
def test_property_attributes(test_property, c3: Controller, prop):
    test_property(c3, prop)

def test_init(sf3T: Surface, c3: Controller):
    macar3T = Macar(sf3T)
    with mock.patch("localuf.decoders.luf.Controller.reset") as mock_reset:
        c3.__init__(macar3T)
        assert c3._LUF is macar3T
        mock_reset.assert_called_once()

def test_reset(c3: Controller):
    c3.stage = Stage.MERGING
    c3.reset()
    assert c3.stage is Stage.GROWING

def test_advance(c3: Controller):
    assert c3.advance()
    assert c3.stage is Stage.MERGING
    assert not c3.advance()
    assert c3.stage is Stage.PRESYNCING
    assert not c3.advance()
    assert c3.stage is Stage.SYNCING
    assert not c3.advance()
    assert c3.stage is Stage.BURNING
    assert c3.advance()
    assert c3.stage is Stage.PEELING
    assert c3.advance()
    assert c3.stage is Stage.DONE