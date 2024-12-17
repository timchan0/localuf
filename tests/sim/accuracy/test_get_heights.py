from collections.abc import Callable
from itertools import product

import pytest

from localuf.sim._height_calculator import get_heights

d = 3
global_batch_slenderness = 100
products = tuple(product((None, lambda d: 2*(d//2), lambda d: 3*(d//2)), repeat=2))


@pytest.mark.parametrize('get_commit_height, get_buffer_height', products)
def test_batch(get_commit_height: None | Callable[[int], int], get_buffer_height: None | Callable[[int], int]):
    w, c, b = get_heights(d, global_batch_slenderness, 'batch', get_commit_height, get_buffer_height)
    assert w is None
    if get_commit_height is None:
        assert c is None
    else:
        assert c == get_commit_height(d)
    if get_buffer_height is None:
        assert b is None
    else:
        assert b == get_buffer_height(d)


@pytest.mark.parametrize('get_commit_height, get_buffer_height', products)
def test_global_batch(get_commit_height: None | Callable[[int], int], get_buffer_height: None | Callable[[int], int]):
    w, c, b = get_heights(d, global_batch_slenderness, 'global batch', get_commit_height, get_buffer_height)
    assert w == d*global_batch_slenderness
    if get_commit_height is None:
        assert c is None
    else:
        assert c == get_commit_height(d)
    if get_buffer_height is None:
        assert b is None
    else:
        assert b == get_buffer_height(d)


@pytest.mark.parametrize('get_commit_height, get_buffer_height', products)
def test_forward(get_commit_height: None | Callable[[int], int], get_buffer_height: None | Callable[[int], int]):
    w, c, b = get_heights(d, global_batch_slenderness, 'forward', get_commit_height, get_buffer_height)
    assert w is None
    if get_commit_height is None:
        assert c is d
    else:
        assert c == get_commit_height(d)
    if get_buffer_height is None:
        assert b is d
    else:
        assert b == get_buffer_height(d)


@pytest.mark.parametrize('get_commit_height, get_buffer_height', products)
def test_frugal(get_commit_height: None | Callable[[int], int], get_buffer_height: None | Callable[[int], int]):
    w, c, b = get_heights(d, global_batch_slenderness, 'frugal', get_commit_height, get_buffer_height)
    assert w is None
    if get_commit_height is None:
        assert c is 1
    else:
        assert c == get_commit_height(d)
    if get_buffer_height is None:
        assert b is 2*(d//2)
    else:
        assert b == get_buffer_height(d)