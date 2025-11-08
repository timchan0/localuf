from localuf import Surface

def test_init():
    """Calls `Surface._substitute`.
    
    Fails if `Surface._substitute` calls `self._NOISE` before this is defined.
    """
    code = Surface(3, 'circuit-level', merge_equivalent_boundary_nodes=True)
    assert code.DIMENSION == 3