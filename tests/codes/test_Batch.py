from localuf.codes import _Batch, SpaceDeterminant

def test_DETERMINANT_attribute(batch3F: _Batch):
    assert type(batch3F._DETERMINANT) is SpaceDeterminant