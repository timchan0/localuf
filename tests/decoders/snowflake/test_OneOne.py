from localuf.decoders import Snowflake
from tests.decoders.snowflake.Snowflake.test_Snowflake_unit import tds


from unittest import mock


def test_grow(snowflake_one_one: Snowflake):
    node_calls = [mock.call()] * len(snowflake_one_one.NODES)
    with (
        mock.patch(f"{tds}._Node.grow") as mock_growing,
        mock.patch(f"{tds}._Node.update_access") as mock_ua,
    ):
        snowflake_one_one._SCHEDULE.grow()
        assert mock_growing.call_args_list == node_calls
        assert mock_ua.call_args_list == node_calls