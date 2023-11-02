from localuf import Surface


def test_is_boundary(sf: Surface):
    n_boundaries = sum(sf.is_boundary(v) for v in sf.NODES)
    power = sf.DIMENSION - 1
    assert n_boundaries == 2 * sf.D**power