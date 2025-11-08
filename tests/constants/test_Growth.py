from localuf.constants import Growth

def test_iadd():
    """Test the __iadd__ method of Growth enum."""
    assert Growth.BURNT + Growth.INCREMENT == Growth.UNGROWN
    assert Growth.UNGROWN + Growth.INCREMENT == Growth.HALF
    assert Growth.HALF + Growth.INCREMENT == Growth.FULL

def test_as_float():
    """Test the as_float property of Growth enum."""
    burnt = Growth.BURNT
    assert burnt.as_float == 1.0
    ungrown = Growth.UNGROWN
    assert ungrown.as_float == 0.0
    half = Growth.HALF
    assert half.as_float == 0.5
    full = Growth.FULL
    assert full.as_float == 1.0