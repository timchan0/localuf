from localuf._schemes import Forward
from localuf.decoders import UF

def test_get_syndrome(forward_rp: Forward):
    syndrome = forward_rp._get_syndrome(set(forward_rp._COMMIT_EDGES), set())
    assert syndrome == {v for v in forward_rp._CODE.NODES
                             if v[forward_rp._CODE.TIME_AXIS] == 0
                             and not forward_rp._CODE.is_boundary(v)}


def test_run(forward_rp: Forward):
    uf = UF(forward_rp._CODE)
    for n in range(1, 4):
        assert forward_rp.run(uf, 0, n) == (0, n+2)
        a, b = forward_rp.run(uf, 1, n)
        assert a != 0
        assert b == n+2


def test_make_error(forward3F: Forward):
    error = forward3F._make_error(set(), 0)
    assert error == set()
    error = forward3F._make_error(set(), 1)
    assert error == set(forward3F._BUFFER_EDGES)
    error = forward3F._make_error(set(forward3F._BUFFER_EDGES), 0)
    assert error == set(forward3F._COMMIT_EDGES)
    error = forward3F._make_error(set(forward3F._BUFFER_EDGES), 1)
    assert error == set(forward3F._CODE.EDGES)