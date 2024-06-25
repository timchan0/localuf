from typing import Type
from unittest.mock import patch

from pandas import DataFrame, MultiIndex
import pytest

from localuf import Repetition, Surface
from localuf.decoders.snowflake import Snowflake
from localuf.sim import runtime
from localuf._base_classes import Code
from localuf._schemes import Frugal


@pytest.mark.parametrize('code_class', (Repetition, Surface))
def test_frugal(code_class: Type[Code]):
    ds = [1, 2]
    ps = [0, 1]
    n = 2
    call_count = len(ds) * len(ps)

    def make_step_counts(
            decoder: Snowflake,
            p: float,
            n: int,
            **_,
    ):
        frugal: Frugal = decoder.CODE.SCHEME # type: ignore
        frugal.step_counts = (n-1)*[int(p)]

    with patch(
        'localuf._schemes.Frugal.run',
        side_effect=make_step_counts,
    ) as mock_run:
        data = runtime.frugal(
            ds=ds,
            ps=ps,
            n=n,
            code_class=code_class,
            noise='phenomenological',
        )
        assert mock_run.call_count == call_count
    assert type(data) is DataFrame
    assert data.shape == (n-1, len(ds) * len(ps))
    assert data.columns.names == ['d', 'p']
    assert (data.columns == MultiIndex.from_product([ds, ps])).all()