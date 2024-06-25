import pytest

from localuf.decoders.luf.constants import Stage

def test_str():
    assert str(Stage.GROWING) == 'G'
    assert str(Stage.MERGING) == 'M'
    assert str(Stage.PRESYNCING) == 'PS'
    assert str(Stage.SYNCING) == 'S'
    assert str(Stage.BURNING) == 'B'
    assert str(Stage.PEELING) == 'P'
    assert str(Stage.DONE) == 'D'

def test_iadd():
    x = Stage.GROWING
    x += Stage.INCREMENT
    assert x is Stage.MERGING
    x += Stage.INCREMENT
    assert x is Stage.PRESYNCING
    x += Stage.INCREMENT
    assert x is Stage.SYNCING
    x += Stage.INCREMENT
    assert x is Stage.BURNING
    x += Stage.INCREMENT
    assert x is Stage.PEELING
    x += Stage.INCREMENT
    assert x is Stage.DONE
    with pytest.raises(ValueError, match="7 is not a valid Stage"):
        x += Stage.INCREMENT

def test_imod():
    x = Stage.BURNING
    x %= Stage.SV_STAGE_COUNT
    assert x is Stage.GROWING