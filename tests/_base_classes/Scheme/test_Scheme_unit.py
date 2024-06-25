from unittest import mock

import pytest

from localuf import Repetition, Surface
from localuf._determinants import Determinant
from localuf._schemes import Batch, Forward


@pytest.mark.parametrize("prop", [
    "WINDOW_HEIGHT",
])
def test_property_attributes(test_property, batch3F: Batch, prop: str):
    test_property(batch3F, prop)

def test_CODE_attribute(sf3F: Surface, batch3F: Batch):
    assert batch3F._CODE is sf3F

def test_WINDOW_HEIGHT_attribute(forward_rp: Forward):
    assert forward_rp.WINDOW_HEIGHT == forward_rp._COMMIT_HEIGHT + forward_rp._BUFFER_HEIGHT

def test_DETERMINANT_attribute(batch3F: Batch):
    assert issubclass(type(batch3F._DETERMINANT), Determinant)

def test_get_logical_error(sf5F: Surface):
    leftover0 = {
        ((2, 2), (2, 3)),
        ((2, 3), (2, 4)),
        ((0, 2), (1, 2)),
        ((1, 2), (2, 2)),
        ((0, 2), (0, 3)),
        ((0, 3), (0, 4)),
    }
    leftover1 = {((0, j), (0, j+1)) for j in range(-1, 5-1)}
    assert sf5F.SCHEME.get_logical_error(leftover0) == 0
    assert sf5F.SCHEME.get_logical_error(leftover1) == 1


def test_is_boundary(sf3F: Surface, rp3_forward: Repetition):
    with mock.patch("localuf._determinants.Determinant.is_boundary") as m:
        v, *_ = sf3F.NODES
        sf3F.SCHEME.is_boundary(v)
        m.assert_called_once_with(v)
    with mock.patch("localuf._determinants.SpaceTimeDeterminant.is_boundary") as m:
        v, *_ = rp3_forward.NODES
        rp3_forward.SCHEME.is_boundary(v)
        m.assert_called_once_with(v)