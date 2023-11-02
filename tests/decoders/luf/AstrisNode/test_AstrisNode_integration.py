from localuf import Surface
from localuf.decoders.luf import LUF

def test_merging_never_relay_from_boundary(sf3F: Surface):
    luf = LUF(sf3F)
    syndrome = {
        (0, 0),
        (2, 0),
    }
    n_steps = luf.validate(syndrome)
    assert n_steps == 12
    capital_I = {
        ((0, -1), (0, 0)),
        ((0, 0), (0, 1)),
        ((0, 0), (1, 0)),
        ((1, 0), (2, 0)),
        ((2, -1), (2, 0)),
        ((2, 0), (2, 1)),
    }
    assert luf.erasure == capital_I