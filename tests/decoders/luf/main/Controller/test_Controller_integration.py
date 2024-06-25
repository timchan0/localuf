from localuf.decoders.luf import Controller
from localuf.decoders.luf.constants import Stage


def test_stage_symbol(c3: Controller):
    assert str(c3.stage) == 'G'
    c3.stage = Stage.MERGING
    assert str(c3.stage) == 'M'