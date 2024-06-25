from localuf import Repetition


def test_is_boundary(rp_forward: Repetition):
    n_boundaries = sum(rp_forward.is_boundary(v) for v in rp_forward.NODES)
    assert n_boundaries == 2 * rp_forward.SCHEME.WINDOW_HEIGHT + rp_forward.D - 1