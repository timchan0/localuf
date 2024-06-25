from localuf import Surface
from localuf.decoders.luf import Macar

def test_merging_never_relay_from_boundary(sf3F: Surface):
    decoder = Macar(sf3F)
    syndrome = {
        (0, 0),
        (2, 0),
    }
    n_steps = decoder.validate(syndrome)
    assert n_steps == 12
    capital_I = {
        ((0, -1), (0, 0)),
        ((0, 0), (0, 1)),
        ((0, 0), (1, 0)),
        ((1, 0), (2, 0)),
        ((2, -1), (2, 0)),
        ((2, 0), (2, 1)),
    }
    assert decoder.erasure == capital_I