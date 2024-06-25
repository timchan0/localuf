from itertools import repeat, product

import pytest

from localuf.sim.accuracy import _get_heights

d = 3
n = 100
ns = repeat(n)
products = tuple(product((None, 2, 3), repeat=2))


@pytest.mark.parametrize('commit_multiplier, buffer_multiplier', products)
def test_batch(commit_multiplier: int, buffer_multiplier: int):
    w, c, b = _get_heights(d, ns, 'batch', commit_multiplier, buffer_multiplier)
    assert w is None
    if commit_multiplier is None:
        assert c is None
    else:
        assert c == commit_multiplier*(d//2)
    if buffer_multiplier is None:
        assert b is None
    else:
        assert b == buffer_multiplier*(d//2)


@pytest.mark.parametrize('commit_multiplier, buffer_multiplier', products)
def test_global_batch(commit_multiplier: int, buffer_multiplier: int):
    w, c, b = _get_heights(d, ns, 'global batch', commit_multiplier, buffer_multiplier)
    assert w == d*n
    if commit_multiplier is None:
        assert c is None
    else:
        assert c == commit_multiplier*(d//2)
    if buffer_multiplier is None:
        assert b is None
    else:
        assert b == buffer_multiplier*(d//2)


@pytest.mark.parametrize('commit_multiplier, buffer_multiplier', products)
def test_forward(commit_multiplier: int, buffer_multiplier: int):
    w, c, b = _get_heights(d, ns, 'forward', commit_multiplier, buffer_multiplier)
    assert w is None
    if commit_multiplier is None:
        assert c is d
    else:
        assert c == commit_multiplier*(d//2)
    if buffer_multiplier is None:
        assert b is d
    else:
        assert b == buffer_multiplier*(d//2)


@pytest.mark.parametrize('commit_multiplier, buffer_multiplier', products)
def test_frugal(commit_multiplier: int, buffer_multiplier: int):
    w, c, b = _get_heights(d, ns, 'frugal', commit_multiplier, buffer_multiplier)
    assert w is None
    if commit_multiplier is None:
        assert c is 1
    else:
        assert c == commit_multiplier*(d//2)
    if buffer_multiplier is None:
        assert b is 2*(d//2)
    else:
        assert b == buffer_multiplier*(d//2)