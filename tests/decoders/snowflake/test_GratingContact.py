from localuf.decoders import Snowflake
from localuf._pairs import Pairs


def test_drop(snowflake3: Snowflake):
    pairs: Pairs = snowflake3.CODE.SCHEME.pairs # type: ignore

    u, v = (0, 0), (1, 0)
    edge = snowflake3.EDGES[u, v]

    edge.CONTACT.drop()
    assert pairs._dc == {}

    edge.correction = True
    edge.CONTACT.drop()
    assert edge.correction is True
    assert pairs._dc == {u: v, v: u}

    edge.CONTACT.drop()
    assert pairs._dc == {}