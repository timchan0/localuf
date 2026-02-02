from itertools import repeat
from unittest.mock import patch, call

from pandas import DataFrame, MultiIndex

from localuf.sim import runtime


def test_validate_only():
    stages = ('SV',)
    ds = [1, 2]
    noise_levels = [0, 1]
    n = 2
    call_count = len(ds) * len(noise_levels) * n
    with (
        patch('localuf.Surface.make_error', return_value=set()) as mock_me,
        patch('localuf.Surface.get_syndrome', return_value=set()) as mock_gs,
        patch('localuf.decoders.luf.main.LUF.validate', return_value=0) as mock_validate,
        patch('localuf.decoders.luf.main.LUF.reset') as mock_reset,
    ):
        data = runtime.batch(
            ds=ds,
            noise_levels=noise_levels,
            n=n,
            noise='code capacity',
        )
        assert mock_me.call_args_list == len(ds) * [
            call(noise_level) for noise_level in noise_levels for _ in repeat(None, times=n)
        ]
        assert mock_gs.call_args_list == call_count * [call(set())]
        assert mock_validate.call_args_list == call_count * [call(set())]
        assert mock_reset.call_args_list == call_count * [call()]
    assert type(data) is DataFrame
    assert data.shape == (n, len(ds) * len(noise_levels))
    assert data.columns.names == ['stage', 'd', 'p']
    assert (data.columns == MultiIndex.from_product([stages, ds, noise_levels])).all()


def test_full():
    stages = ('BP', 'SV')
    ds = [1, 2]
    noise_levels = [0, 1]
    n = 2
    call_count = len(ds) * len(noise_levels) * n
    with (
        patch('localuf.Surface.make_error', return_value=set()) as mock_me,
        patch('localuf.Surface.get_syndrome', return_value=set()) as mock_gs,
        patch('localuf.decoders.luf.main.LUF.decode', return_value=(0, 0)) as mock_decode,
        patch('localuf.decoders.luf.main.LUF.reset') as mock_reset,
    ):
        data = runtime.batch(
            ds=ds,
            noise_levels=noise_levels,
            n=n,
            noise='code capacity',
            validate_only=False,
        )
        assert mock_me.call_args_list == len(ds) * [
            call(noise_level) for noise_level in noise_levels for _ in repeat(None, times=n)
        ]
        assert mock_gs.call_args_list == call_count * [call(set())]
        assert mock_decode.call_args_list == call_count * [call(set())]
        assert mock_reset.call_args_list == call_count * [call()]
    assert type(data) is DataFrame
    assert data.shape == (n, len(stages) * len(ds) * len(noise_levels))
    assert data.columns.names == ['stage', 'd', 'p']
    assert (data.columns == MultiIndex.from_product([stages, ds, noise_levels])).all()