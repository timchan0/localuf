import numpy as np

from localuf.noise import CircuitLevel

def test_make_multiplicities_both(
        assert_these_multiplicities_unchanged,
        diagonals,
):
    m = CircuitLevel._make_multiplicities(demolition=True, monolingual=True)
    assert (m['S'] == np.array((4, 2, 4, 0))).all()
    assert (m['E westmost'] == np.array((1, 0, 5, 0))).all()
    assert (m['E bulk'] == np.array((2, 0, 4, 0))).all()
    assert (m['E eastmost'] == np.array((1, 0, 4, 0))).all()
    assert (m['U 3'] == np.array((3, 0, 3, 2))).all()
    assert (m['U 4'] == np.array((4, 0, 2, 2))).all()
    assert_these_multiplicities_unchanged(m, diagonals)