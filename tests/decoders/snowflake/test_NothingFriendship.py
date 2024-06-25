from unittest import mock

from localuf.decoders.snowflake import _Node


def test_find_broken_pointers(sfn3: _Node):
    for pointer in ('D', 'SD', 'WD', 'NWD'):
        with mock.patch("localuf.decoders.snowflake.main._FullUnrooter.start") as mock_start:
            sfn3.pointer = pointer
            sfn3.FRIENDSHIP.find_broken_pointers()
            mock_start.assert_called_once()