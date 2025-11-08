import pytest

from localuf.decoders.snowflake.main import _Repetition


def test_syndrome_vector_to_set():
    converter = _Repetition(3, 2)
    assert converter._syndrome_vector_to_set('00') == set()
    assert converter._syndrome_vector_to_set('10') == {(0, 1)}
    assert converter._syndrome_vector_to_set('01') == {(1, 1)}
    assert converter._syndrome_vector_to_set('11') == {(0, 1), (1, 1)}
    with pytest.raises(ValueError):
        converter._syndrome_vector_to_set('10010')


@pytest.mark.parametrize("window_height", [2, 3])
def test_set_to_syndrome_vector(window_height: int):
    converter = _Repetition(3, window_height)
    assert converter._set_to_syndrome_vector({(0, window_height-1), (1, window_height-1)}) == '11'
    assert converter._set_to_syndrome_vector(set()) == '00'