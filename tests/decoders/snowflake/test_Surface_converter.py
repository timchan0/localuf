import pytest

from localuf.decoders.snowflake.main import _Surface


def test_syndrome_vector_to_set():
    converter = _Surface(3, 2)
    assert converter._syndrome_vector_to_set('100100') == {(2, 0, 1), (1, 1, 1)}
    assert converter._syndrome_vector_to_set('010010') == {(2, 1, 1), (0, 0, 1)}
    assert converter._syndrome_vector_to_set('001001') == {(1, 0, 1), (0, 1, 1)}
    assert converter._syndrome_vector_to_set('000000') == set()
    with pytest.raises(ValueError):
        converter._syndrome_vector_to_set('10010')


@pytest.mark.parametrize("window_height", [1, 2, 3])
def test_bit_position_to_defect(window_height: int):
    converter = _Surface(3, window_height)
    assert converter._bit_position_to_defect(0) == (2, 0, window_height-1)
    assert converter._bit_position_to_defect(1) == (2, 1, window_height-1)
    assert converter._bit_position_to_defect(2) == (1, 0, window_height-1)
    assert converter._bit_position_to_defect(3) == (1, 1, window_height-1)
    assert converter._bit_position_to_defect(4) == (0, 0, window_height-1)
    assert converter._bit_position_to_defect(5) == (0, 1, window_height-1)


@pytest.mark.parametrize("window_height", [1, 2, 3])
def test_defect_to_bit_position(window_height: int):
    converter = _Surface(3, window_height)
    assert converter._defect_to_bit_position((2, 0, window_height-1)) == 0
    assert converter._defect_to_bit_position((2, 1, window_height-1)) == 1
    assert converter._defect_to_bit_position((1, 0, window_height-1)) == 2
    assert converter._defect_to_bit_position((1, 1, window_height-1)) == 3
    assert converter._defect_to_bit_position((0, 0, window_height-1)) == 4
    assert converter._defect_to_bit_position((0, 1, window_height-1)) == 5
    with pytest.raises(ValueError):
        converter._defect_to_bit_position((3, 0, window_height))


@pytest.mark.parametrize("window_height", [2, 3])
def test_set_to_syndrome_vector(window_height: int):
    converter = _Surface(3, window_height)
    assert converter._set_to_syndrome_vector({(2, 0, window_height-1), (1, 1, window_height-1)}) == '100100'
    assert converter._set_to_syndrome_vector({(2, 1, window_height-1), (0, 0, window_height-1)}) == '010010'
    assert converter._set_to_syndrome_vector({(1, 0, window_height-1), (0, 1, window_height-1)}) == '001001'
    assert converter._set_to_syndrome_vector(set()) == '000000'